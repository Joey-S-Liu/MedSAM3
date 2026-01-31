
#!/usr/bin/env python3
"""
SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters

from torchvision.transforms import v2
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format

# Note: Evaluation modules (mAP, cgF1, NMS) are in validate_sam3_lora.py
# Training only computes validation loss, following SAM3's approach


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # Load COCO annotations
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # Convert from COCO [x, y, w, h] to normalized [cx, cy, w, h] (CxCyWH)
            # SAM3 internally expects boxes in CxCyWH format normalized to [0, 1]
            x, y, w, h = bbox_coco
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Scale to resolution and normalize to [0, 1]
            box_tensor = torch.tensor([
                cx * scale_w / self.resolution,
                cy * scale_h / self.resolution,
                w * scale_w / self.resolution,
                h * scale_h / self.resolution,
            ], dtype=torch.float32)

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # Check if it's RLE format (dict) or polygon format (list)
                    if isinstance(segmentation, dict):
                        # RLE format: {"counts": "...", "size": [h, w]}
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        print(f"Warning: Unknown segmentation format: {type(segmentation)}")
                        segment = None
                        continue

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Query
        # Use category names from COCO annotations for category-aware training
        object_ids = [obj.object_id for obj in objects]

        # Determine query text based on categories present in the image
        if len(object_class_names) > 0:
            unique_classes = list(set(object_class_names))
            if len(unique_classes) == 1:
                # Single category: use its name (lowercase)
                query_text = unique_classes[0].lower()
            else:
                # Multiple categories: use most common one
                # This helps the model learn category-specific features
                from collections import Counter
                most_common = Counter(object_class_names).most_common(1)[0][0]
                query_text = most_common.lower()
        else:
            # No annotations: use generic term
            query_text = "object"

        query = FindQueryLoaded(
            query_text=query_text,
            image_id=0,
            object_ids_output=object_ids,
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=img_id,
                original_image_id=img_id,
                original_category_id=0,
                original_size=(orig_h, orig_w),
                object_id=-1,
                frame_index=-1
            )
        )

        return Datapoint(
            find_queries=[query],
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    """
    Merge overlapping masks that likely represent the same object.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.3)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format for evaluation.

    OPTIMIZATION: Keep masks at native model output resolution (288×288)
    GT is downsampled to match, so no upsampling needed!

    Args:
        predictions_list: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        resolution: Mask resolution for evaluation (default: 288, model's native output)
        score_threshold: Minimum score threshold for predictions
        merge_overlaps: Whether to merge overlapping predictions (default: True)
        iou_threshold: IoU threshold for merging overlaps (default: 0.3)
        debug: Print debug information

    Returns:
        List of prediction dictionaries in COCO format
    """
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Extract predictions
        logits = preds['pred_logits']  # [num_queries, 1]
        boxes = preds['pred_boxes']    # [num_queries, 4]
        masks = preds['pred_masks']    # [num_queries, H, W]

        scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:  # Debug first image only
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        # Convert masks to binary (apply sigmoid first, then threshold)
        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        # Merge overlapping predictions to avoid over-segmentation penalty
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        # Encode masks to RLE (at native resolution - much faster!)
        if len(binary_masks) > 0:
            # Check if masks have content
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to [x, y, w, h] in pixel coordinates
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    'image_id': int(img_id),
                    'category_id': 1,  # Single category for instance segmentation
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                })
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from SimpleSAM3Dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    scale_factor = mask_resolution / dataset.resolution

    for idx in indices:
        # Add image entry at mask resolution
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True  # Required for cgF1 evaluation
        })

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at mask_resolution
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free GPU memory immediately after upsampling
        del masks_sigmoid, masks_upsampled
        torch.cuda.empty_cache()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


class SAM3TrainerNative:
    def __init__(self, config_path, multi_gpu=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print_rank0("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,  # Tries to download from HF if checkpoint_path is None
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False
        )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )
        
    def train(self):
        # Get data directory from config (should point to directory containing train/valid folders)
        data_dir = self.config["training"]["data_dir"]

        # Load datasets using COCO format
        print_rank0(f"\nLoading training data from {data_dir}...")
        train_ds = COCOSegmentDataset(data_dir=data_dir, split="train")

        # Check if validation data exists
        has_validation = False
        val_ds = None

        try:
            print_rank0(f"\nLoading validation data from {data_dir}...")
            val_ds = COCOSegmentDataset(data_dir=data_dir, split="valid")
            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} images")
            else:
                print_rank0(f"Validation dataset is empty.")
                val_ds = None
        except Exception as e:
            print_rank0(f"Could not load validation data: {e}")
            val_ds = None

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        epochs = self.config["training"]["num_epochs"]
        best_val_loss = float('inf')
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}")

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch
            train_losses = []

            # Only show progress bar on rank 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for batch_dict in pbar:
                input_batch = batch_dict["input"]

                # Move to device
                input_batch = move_to_device(input_batch, self.device)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Track training loss
                train_losses.append(total_loss.item())
                pbar.set_postfix({"loss": total_loss.item()})

            # Calculate average training loss for this epoch
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation", disable=not is_main_process())

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss
                        }) + "\n")

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                print(f"   python validate_sam3_lora.py \\")
                print(f"     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir <data_dir>/valid")
                print(f"{'='*80}")
            else:
                # If no validation, copy last to best
                import shutil
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\nℹ️  No validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],  # This script
        "--config", args.config,
        "--device", *map(str, devices),
        "--_launched_by_torchrun"  # Internal flag to indicate we're in subprocess
    ]

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
             "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)"
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, multi_gpu=multi_gpu)
        trainer.train()
