#!/usr/bin/env python3
"""
Inference script for SAM3 with LoRA weights.

This script loads the SAM3 model, applies LoRA configuration,
loads trained LoRA weights, and runs inference on images.

Usage:
    python3 inference_lora.py \
        --config configs/full_lora_config.yaml \
        --weights outputs/sam3_lora_full/lora_weights.pt \
        --image path/to/image.jpg \
        --prompt "object to segment" \
        --output output.png
"""

import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image as PILImage
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

from torchvision.transforms import v2


class SAM3LoRAInference:
    def __init__(self, config_path, weights_path):
        """
        Initialize SAM3 LoRA inference.

        Args:
            config_path: Path to YAML config file used for training
            weights_path: Path to saved LoRA weights (.pt file)
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.resolution = 1008

        # Build Model
        print("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True  # Set to eval mode for inference
        )

        # Apply LoRA (with same config as training)
        print("Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,  # No dropout during inference
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load LoRA weights
        print(f"Loading LoRA weights from {weights_path}...")
        load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup image transform
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print("✅ Model ready for inference!")

    def prepare_image(self, image_path):
        """
        Load and prepare image for inference.

        Args:
            image_path: Path to input image

        Returns:
            Prepared datapoint for SAM3
        """
        # Load image
        pil_image = PILImage.open(image_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image_resized = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image_resized)

        # Create Image object
        image_obj = Image(
            data=image_tensor,
            objects=[],  # No objects for inference
            size=(self.resolution, self.resolution)
        )

        # Create query
        query = FindQueryLoaded(
            query_text="object",  # Generic query
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=0,
                original_size=(orig_h, orig_w),
                object_id=-1,
                frame_index=-1
            )
        )

        return Datapoint(
            find_queries=[query],
            images=[image_obj],
            raw_images=[pil_image_resized]
        ), pil_image, (orig_w, orig_h)

    @torch.no_grad()
    def predict(self, image_path, text_prompt=None):
        """
        Run inference on an image with optional text prompt.

        Args:
            image_path: Path to input image
            text_prompt: Optional text prompt to guide segmentation (e.g., "yellow school bus", "person wearing hat")
                        If None, uses generic "object" query.

        Returns:
            Dictionary containing predictions:
                - boxes: Predicted bounding boxes [N, 4] in normalized coords
                - scores: Confidence scores [N, num_classes]
                - masks: Segmentation masks [N, H, W] (if available)
                - original_size: Original image size (width, height)
                - image: PIL Image object
        """
        # Prepare input
        datapoint, original_image, (orig_w, orig_h) = self.prepare_image(image_path)

        # Override text prompt if provided
        if text_prompt:
            print(f"Using text prompt: '{text_prompt}'")
            datapoint.find_queries[0].query_text = text_prompt
        else:
            print("Using generic 'object' query (consider adding --prompt for better results)")

        # Collate into batch
        batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
        input_batch = batch_dict["input"]

        # Move to device
        input_batch = self._move_to_device(input_batch, self.device)

        # Forward pass
        print("Running inference...")
        outputs_list = self.model(input_batch)
        outputs = outputs_list[-1]

        # Extract predictions
        pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4]
        pred_masks = outputs.get('pred_masks', None)  # [batch, num_queries, H, W]

        # Get confidence scores (sigmoid of logits)
        out_probs = pred_logits.sigmoid()

        # Convert to numpy for easier handling
        scores = out_probs.cpu().numpy()
        pred_boxes = pred_boxes.cpu().numpy()
        if pred_masks is not None:
            pred_masks = pred_masks.cpu().numpy()

        # Debug: Print box value ranges
        boxes_0 = pred_boxes[0]
        print(f"\n📊 Box statistics:")
        print(f"  cx range: [{boxes_0[:, 0].min():.3f}, {boxes_0[:, 0].max():.3f}]")
        print(f"  cy range: [{boxes_0[:, 1].min():.3f}, {boxes_0[:, 1].max():.3f}]")
        print(f"  w range: [{boxes_0[:, 2].min():.3f}, {boxes_0[:, 2].max():.3f}]")
        print(f"  h range: [{boxes_0[:, 3].min():.3f}, {boxes_0[:, 3].max():.3f}]")

        return {
            'boxes': pred_boxes[0],  # Remove batch dimension
            'scores': scores[0],
            'masks': pred_masks[0] if pred_masks is not None else None,
            'original_size': (orig_w, orig_h),
            'image': original_image
        }

    def _move_to_device(self, obj, device):
        """Helper to move nested structures to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [self._move_to_device(x, device) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(x, device) for x in obj)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        elif hasattr(obj, "__dataclass_fields__"):
            for field in obj.__dataclass_fields__:
                val = getattr(obj, field)
                setattr(obj, field, self._move_to_device(val, device))
            return obj
        return obj

    def visualize_predictions(self, predictions, output_path, confidence_threshold=0.5, text_prompt=None, nms_iou_threshold=0.5):
        """
        Visualize predictions on the image.

        Args:
            predictions: Dictionary from predict()
            output_path: Where to save the visualization
            confidence_threshold: Minimum confidence to show predictions
            text_prompt: Optional text prompt to display in title
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
        """
        image = predictions['image']
        boxes = predictions['boxes']
        scores = predictions['scores']
        masks = predictions['masks']
        orig_w, orig_h = predictions['original_size']

        # Filter by confidence first
        max_scores = scores.max(axis=1)
        valid_mask = max_scores > confidence_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            print(f"⚠️ No predictions above confidence threshold {confidence_threshold}")
            return

        # Apply NMS to remove overlapping boxes
        # Convert boxes from cxcywh to xyxy for NMS
        valid_boxes = boxes[valid_indices]  # [N, 4] in cxcywh normalized
        valid_scores_flat = max_scores[valid_indices]

        # Convert cxcywh to xyxy (still normalized)
        cx, cy, w, h = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        boxes_tensor = torch.from_numpy(boxes_xyxy).float()
        scores_tensor = torch.from_numpy(valid_scores_flat).float()
        keep_nms = nms(boxes_tensor, scores_tensor, nms_iou_threshold)
        keep_nms = keep_nms.numpy()

        # Update valid_indices to only keep NMS survivors
        valid_indices = valid_indices[keep_nms]

        print(f"📦 NMS: {len(valid_mask.nonzero()[0])} → {len(valid_indices)} boxes (IoU threshold: {nms_iou_threshold})")

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # Draw boxes and masks
        for idx in valid_indices:
            box = boxes[idx]  # [cx, cy, w, h] normalized [0, 1]
            score = scores[idx].max()

            # Convert from [cx, cy, w, h] to [x1, y1, x2, y2] (still normalized)
            cx, cy, w, h = box
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Clamp normalized coordinates to [0, 1] range
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # Scale to original image size
            orig_w, orig_h = predictions['original_size']
            x1 = x1 * orig_w
            y1 = y1 * orig_h
            x2 = x2 * orig_w
            y2 = y2 * orig_h

            # Get width and height for rectangle
            width = x2 - x1
            height = y2 - y1

            # Draw box
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x1, y1 - 5,
                f'{score:.2f}',
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=10, color='white'
            )

            # Draw mask if available
            if masks is not None:
                mask = masks[idx] > 0.5  # Threshold mask
                # Resize mask to original image size
                from scipy.ndimage import zoom
                mask_resized = zoom(mask, (orig_h / mask.shape[0], orig_w / mask.shape[1]), order=0)

                # Overlay mask
                colored_mask = np.zeros((*mask_resized.shape, 4))
                colored_mask[mask_resized] = [1, 0, 0, 0.3]  # Red with alpha
                ax.imshow(colored_mask)

        ax.axis('off')

        # Add title with text prompt if provided
        if text_prompt:
            plt.suptitle(f'Text Prompt: "{text_prompt}"', fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"✅ Saved visualization to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="SAM3 LoRA Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file used for training"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained LoRA weights (.pt file). If not specified, uses best_lora_weights.pt from config's output_dir"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='Text prompt to guide segmentation (e.g., "yellow school bus", "person with red hat", "car"). Improves accuracy for specific objects.'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for showing predictions"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5, lower = fewer boxes)"
    )

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    # Auto-detect weights path if not provided
    if args.weights is None:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        output_dir = config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
        args.weights = os.path.join(output_dir, 'best_lora_weights.pt')
        print(f"ℹ️  Using best model: {args.weights}")

    if not os.path.exists(args.weights):
        print(f"❌ Weights file not found: {args.weights}")
        print(f"   Available: best_lora_weights.pt or last_lora_weights.pt")
        return
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return

    # Initialize inference
    inferencer = SAM3LoRAInference(args.config, args.weights)

    # Run prediction
    predictions = inferencer.predict(args.image, args.prompt)

    # Visualize results with NMS
    inferencer.visualize_predictions(predictions, args.output, args.threshold, text_prompt=args.prompt, nms_iou_threshold=args.nms_iou)

    # Print summary
    print("\n📊 Prediction Summary:")
    if args.prompt:
        print(f"  Text prompt: '{args.prompt}'")
    print(f"  Detected objects: {(predictions['scores'].max(axis=1) > args.threshold).sum()}")
    print(f"  Max confidence: {predictions['scores'].max():.3f}")
    print(f"  Output saved to: {args.output}")


if __name__ == "__main__":
    main()
