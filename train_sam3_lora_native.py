
import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters

from torchvision.transforms import v2

class SimpleSAM3Dataset(Dataset):
    def __init__(self, root_dir, image_set="train"):
        self.root_dir = Path(root_dir) / image_set
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                  list(self.images_dir.glob("*.png")))
        print(f"Loaded {len(self.image_files)} images from {self.images_dir}")
        
        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann_path = self.annotations_dir / f"{img_path.stem}.json"
        
        # Load image
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size
        
        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        
        # Transform to tensor
        image_tensor = self.transform(pil_image)
        
        # Load annotation
        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        objects = []

        # Handle the actual annotation format: {"annotations": [{"bbox": [...], "segmentation": {...}}]}
        annotations = ann_data.get("annotations", [])

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Convert from COCO [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox_coco
            box_tensor = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)

            # Scale box to resolution
            box_tensor[0] *= scale_w
            box_tensor[2] *= scale_w
            box_tensor[1] *= scale_h
            box_tensor[3] *= scale_h

            # IMPORTANT: Normalize boxes to [0, 1] range (required by SAM3 loss functions)
            box_tensor /= self.resolution

            # Handle segmentation mask (RLE encoded)
            segment = None
            segmentation = ann.get("segmentation", None)
            if segmentation and isinstance(segmentation, dict):
                # RLE format: {"counts": "...", "size": [h, w]}
                try:
                    from pycocotools import mask as mask_utils
                    # Decode RLE to binary mask
                    mask_np = mask_utils.decode(segmentation)  # Returns [H, W] binary mask

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor
                except ImportError:
                    print("Warning: pycocotools not installed, skipping mask")
                    segment = None
                except Exception as e:
                    print(f"Warning: Error decoding mask: {e}")
                    segment = None
            
            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2]-box_tensor[0])*(box_tensor[3]-box_tensor[1]),
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
        # We create a single FindQuery that targets all objects
        object_ids = [obj.object_id for obj in objects]
        
        query = FindQueryLoaded(
            query_text="object", # Generic text prompt
            image_id=0, # Relative to this datapoint (only 1 image)
            object_ids_output=object_ids,
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx,
                original_image_id=idx,
                original_category_id=0,
                original_size=(orig_h, orig_w), # Keep original size here? Or resized? Usually original.
                object_id=-1,
                frame_index=-1
            )
        )
        
        return Datapoint(
            find_queries=[query],
            images=[image_obj],
            raw_images=[pil_image] # Keep raw image as PIL (resized or original? Collator might merge them)
        )


class SAM3TrainerNative:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build Model
        print("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True, # Tries to download from HF if checkpoint_path is None
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False
        )
        
        # Apply LoRA
        print("Applying LoRA...")
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
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")
        
        self.model.to(self.device)
        
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
        train_ds = SimpleSAM3Dataset("data", image_set="train")

        # Check if validation data exists (try both "valid" and "val")
        has_validation = False
        val_ds = None

        for val_name in ["valid", "val"]:
            try:
                val_ds = SimpleSAM3Dataset("data", image_set=val_name)
                if len(val_ds) > 0:
                    has_validation = True
                    print(f"Found validation data in data/{val_name}/")
                    break
            except:
                continue

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0 # Simplified
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
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
        print(f"Starting training for {epochs} epochs...")

        if has_validation:
            print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print(f"Training samples: {len(train_ds)}")
            print("⚠️  No validation data found - training without validation")

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

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_dict in pbar:
                input_batch = batch_dict["input"]
                
                # Move to device
                input_batch = move_to_device(input_batch, self.device)
                
                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self.model.back_convert(target) for target in input_batch.find_targets]

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # We need to add "indices" to each output dict (main + aux_outputs)
                for stage_outputs in outputs_list:
                    for outputs in stage_outputs:
                        # Compute indices for main output
                        outputs["indices"] = self.matcher(outputs, find_targets[0])

                        # Also add indices to auxiliary outputs if they exist
                        if "aux_outputs" in outputs:
                            for aux_out in outputs["aux_outputs"]:
                                aux_out["indices"] = self.matcher(aux_out, find_targets[0])

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]
                
                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({"loss": total_loss.item()})

            # Validation (only if validation data exists)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation")
                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self.model.back_convert(target) for target in input_batch.find_targets]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        for stage_outputs in outputs_list:
                            for outputs in stage_outputs:
                                outputs["indices"] = self.matcher(outputs, find_targets[0])
                                if "aux_outputs" in outputs:
                                    for aux_out in outputs["aux_outputs"]:
                                        aux_out["indices"] = self.matcher(aux_out, find_targets[0])

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.6f}")

                # Save best model
                out_dir = Path(self.config["output"]["output_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_lora_weights(self.model, str(out_dir / "best_lora_weights.pt"))
                    print(f"✓ Saved best model (val_loss: {avg_val_loss:.6f})")

                # Save latest model
                save_lora_weights(self.model, str(out_dir / "last_lora_weights.pt"))

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch
                out_dir = Path(self.config["output"]["output_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                save_lora_weights(self.model, str(out_dir / "last_lora_weights.pt"))

        # Final save
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        if has_validation:
            print(f"\n✅ Training complete!")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Models saved to {out_dir}:")
            print(f"  - best_lora_weights.pt (best validation)")
            print(f"  - last_lora_weights.pt (last epoch)")
        else:
            # If no validation, copy last to best
            import shutil
            last_path = out_dir / "last_lora_weights.pt"
            best_path = out_dir / "best_lora_weights.pt"
            if last_path.exists():
                shutil.copy(last_path, best_path)

            print(f"\n✅ Training complete!")
            print(f"Models saved to {out_dir}:")
            print(f"  - best_lora_weights.pt (copy of last epoch)")
            print(f"  - last_lora_weights.pt (last epoch)")
            print(f"\nℹ️  No validation data - consider adding data/valid/ for better model selection")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM3 with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    trainer = SAM3TrainerNative(args.config)
    trainer.train()
