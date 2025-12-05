
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
from sam3.train.matcher import BinaryHungarianMatcherV2
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
        bboxes = ann_data.get("bboxes", [])
        masks = ann_data.get("masks", [])
        
        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h
        
        if not bboxes:
             # Dummy object to avoid crash?
             pass

        for i, bbox in enumerate(bboxes):
            # Assuming bbox is [x1, y1, x2, y2]
            box_tensor = torch.tensor(bbox, dtype=torch.float32)
            
            # Scale box
            box_tensor[0] *= scale_w
            box_tensor[2] *= scale_w
            box_tensor[1] *= scale_h
            box_tensor[3] *= scale_h
            
            segment = None
            if i < len(masks):
                mask_np = np.array(masks[i], dtype=bool)
                # Resize mask
                # Mask resizing is tricky. For now we skip or assume mask is on original image.
                # If we pass mask to object, it should match image size?
                # Sam3Image expects mask? 
                # The loss needs valid masks. 
                # We need to resize mask to 1008x1008.
                # Use torch interpolation.
                mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0) # 1,1,H,W
                mask_t = torch.nn.functional.interpolate(mask_t, size=(self.resolution, self.resolution), mode="nearest")
                segment = mask_t.squeeze() > 0.5
            
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
        
        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }
        
        self.criterion_cls = IABCEMdetr(pos_weight=1.0, weight_dict=weight_dict, pos_focal=True)
        self.criterion_box = Boxes(weight_dict=weight_dict)
        self.criterion_mask = Masks(weight_dict=weight_dict)
        
    def train(self):
        train_ds = SimpleSAM3Dataset("data", image_set="train")

        # Check if validation data exists
        try:
            val_ds = SimpleSAM3Dataset("data", image_set="valid")
            has_validation = len(val_ds) > 0
        except:
            val_ds = None
            has_validation = False

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
                
                # Forward
                # forward() expects BatchedDatapoint
                # We need to make sure find_targets are on device?
                # Sam3Image.forward constructs backbone_out.
                
                outputs_list = self.model(input_batch)
                # outputs_list is SAM3Output with iter_mode=LAST_STEP_PER_STAGE
                # So outputs_list[-1] returns the last step of the last stage, which is the output dict.
                outputs = outputs_list[-1]
                
                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget
                targets_raw = input_batch.find_targets[0] # Stage 0
                targets = self.model.back_convert(targets_raw)
                
                # Move targets to device
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        targets[k] = v.to(self.device)
                
                # Compute Matcher Indices
                # outputs has 'pred_logits', 'pred_boxes'
                # targets has 'boxes', 'labels' etc.
                
                # We need indices for loss
                # The loss wrapper usually calls matcher.
                # We do it manually here.
                indices = self.matcher(outputs, targets)
                
                # Compute Losses
                losses = {}
                
                l_cls = self.criterion_cls.get_loss(outputs, targets, indices, num_boxes=1)
                l_box = self.criterion_box.get_loss(outputs, targets, indices, num_boxes=1)
                l_mask = self.criterion_mask.get_loss(outputs, targets, indices, num_boxes=1)
                
                losses.update(l_cls)
                losses.update(l_box)
                losses.update(l_mask)
                
                # Reduce
                total_loss = 0
                for k, v in losses.items():
                    if k in weight_dict:
                        total_loss += v * weight_dict[k]
                
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

                        # Forward
                        outputs_list = self.model(input_batch)
                        outputs = outputs_list[-1]

                        # Prepare targets
                        targets_raw = input_batch.find_targets[0]
                        targets = self.model.back_convert(targets_raw)

                        for k, v in targets.items():
                            if isinstance(v, torch.Tensor):
                                targets[k] = v.to(self.device)

                        # Compute losses
                        indices = self.matcher(outputs, targets)
                        losses = {}

                        l_cls = self.criterion_cls.get_loss(outputs, targets, indices, num_boxes=1)
                        l_box = self.criterion_box.get_loss(outputs, targets, indices, num_boxes=1)
                        l_mask = self.criterion_mask.get_loss(outputs, targets, indices, num_boxes=1)

                        losses.update(l_cls)
                        losses.update(l_box)
                        losses.update(l_mask)

                        total_loss = 0
                        for k, v in losses.items():
                            if k in weight_dict:
                                total_loss += v * weight_dict[k]

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
