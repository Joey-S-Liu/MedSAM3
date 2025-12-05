#!/usr/bin/env python3
"""
SAM3 LoRA Training with PROPER Category Support

This version:
- Reads category information from COCO file
- Uses actual class names as text prompts during training
- Supports multiple classes properly
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import numpy as np
from torchvision.transforms import v2
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from sam3.train.loss.sam3_loss import SAM3Loss

from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters


class SAM3DatasetWithCategories(Dataset):
    """Dataset that properly uses category names from COCO annotations"""

    def __init__(self, root_dir, coco_file_path=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"

        # Load COCO file to get category mappings
        if coco_file_path is None:
            coco_file_path = self.root_dir / "_annotations.coco.json"

        with open(coco_file_path, 'r') as f:
            self.coco_data = json.load(f)

        # Build category mapping: category_id -> category_name
        self.categories = {
            cat['id']: cat['name']
            for cat in self.coco_data['categories']
        }
        print(f"📂 Loaded {len(self.categories)} categories:")
        for cat_id, cat_name in self.categories.items():
            print(f"   - ID {cat_id}: '{cat_name}'")

        # Build mapping: image_filename -> list of (bbox, mask, category_id)
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']

            # Find image filename
            image_info = next((img for img in self.coco_data['images'] if img['id'] == image_id), None)
            if not image_info:
                continue

            filename = image_info['file_name']

            if filename not in self.image_annotations:
                self.image_annotations[filename] = []

            self.image_annotations[filename].append({
                'bbox': ann['bbox'],  # [x, y, width, height] in COCO format
                'segmentation': ann.get('segmentation'),
                'category_id': ann.get('category_id', 1),
                'area': ann.get('area', 0)
            })

        # Get image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) +
                                  list(self.images_dir.glob("*.png")))
        print(f"📷 Loaded {len(self.image_files)} images from {self.images_dir}")

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

        # Load image
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations from COCO data
        filename = img_path.name
        annotations = self.image_annotations.get(filename, [])

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        objects = []
        category_ids = []

        for i, ann in enumerate(annotations):
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Convert to [x1, y1, x2, y2] and scale
            box_tensor = torch.tensor([
                x * scale_w,
                y * scale_h,
                (x + w) * scale_w,
                (y + h) * scale_h
            ], dtype=torch.float32)

            # Handle segmentation (simplified - would need proper RLE decoding for production)
            segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2]-box_tensor[0])*(box_tensor[3]-box_tensor[1]),
                object_id=i,
                segment=segment
            )
            objects.append(obj)
            category_ids.append(ann['category_id'])

        # If no annotations, create dummy
        if not objects:
            objects = []
            category_ids = []

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Determine which category to use as text prompt
        # Strategy: use the most common category in this image
        if category_ids:
            most_common_cat_id = max(set(category_ids), key=category_ids.count)
            query_text = self.categories.get(most_common_cat_id, "object")
        else:
            query_text = "object"

        # Construct Query with ACTUAL category name
        object_ids = [obj.object_id for obj in objects]

        query = FindQueryLoaded(
            query_text=query_text,  # ✅ Uses actual category name!
            image_id=0,
            object_ids_output=object_ids,
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx,
                original_image_id=idx,
                original_category_id=most_common_cat_id if category_ids else 0,
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


class SAM3TrainerWithCategories:
    """Trainer that properly uses category names"""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )

        # Apply LoRA
        print("Applying LoRA configuration...")
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

        # Print parameter count
        stats = count_parameters(self.model)
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {stats['total_parameters']:,}")
        print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
        print(f"  Trainable percentage: {stats['trainable_percentage']:.2f}%")

        self.model.to(self.device)

        # Setup datasets with category support
        train_cfg = self.config["training"]
        train_path = Path(train_cfg["train_data_path"])

        print(f"\n📁 Loading datasets...")
        self.train_dataset = SAM3DatasetWithCategories(
            root_dir=train_path,
            coco_file_path=train_path / "_annotations.coco.json"
        )

        # Validation dataset
        val_path = Path(train_cfg.get("val_data_path", "data/valid"))
        if val_path.exists() and (val_path / "_annotations.coco.json").exists():
            self.val_dataset = SAM3DatasetWithCategories(
                root_dir=val_path,
                coco_file_path=val_path / "_annotations.coco.json"
            )
            print(f"✅ Validation data loaded: {len(self.val_dataset)} images")
        else:
            self.val_dataset = None
            print("⚠️ No validation data found")

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True),
            pin_memory=True
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=train_cfg.get("num_workers", 4),
                collate_fn=lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True),
                pin_memory=True
            )
        else:
            self.val_loader = None

        # Loss
        self.loss_fn = SAM3Loss()

        # Optimizer (only LoRA parameters)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
            betas=(train_cfg.get("adam_beta1", 0.9), train_cfg.get("adam_beta2", 0.999)),
            eps=train_cfg.get("adam_epsilon", 1e-8)
        )

        # LR Scheduler
        total_steps = len(self.train_loader) * train_cfg["num_epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )

        # Output directory
        self.output_dir = Path(self.config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = train_cfg["num_epochs"]
        self.best_val_loss = float('inf')

    def train(self):
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        print(f"📊 Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"📊 Validation samples: {len(self.val_dataset)}")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            if self.val_loader:
                val_loss = self.validate_epoch(epoch)
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = self.output_dir / "best_lora_weights.pt"
                    save_lora_weights(self.model, best_path)
                    print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}")

            # Save last model
            last_path = self.output_dir / "last_lora_weights.pt"
            save_lora_weights(self.model, last_path)

        # Copy last to best if no validation
        if not self.val_loader:
            import shutil
            shutil.copy(last_path, self.output_dir / "best_lora_weights.pt")
            print(f"ℹ️ No validation - copied last epoch as best")

        print(f"\n✅ Training complete! Weights saved to {self.output_dir}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            input_batch = batch["input"]

            # Move to device
            input_batch = self._move_to_device(input_batch, self.device)

            # Forward
            outputs = self.model(input_batch)

            # Compute loss
            loss_dict = self.loss_fn(outputs, input_batch)
            loss = loss_dict["loss"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_batch = batch["input"]
                input_batch = self._move_to_device(input_batch, self.device)

                outputs = self.model(input_batch)
                loss_dict = self.loss_fn(outputs, input_batch)
                loss = loss_dict["loss"]

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _move_to_device(self, obj, device):
        """Recursively move nested structures to device"""
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/crack_detection_config.yaml")
    args = parser.parse_args()

    trainer = SAM3TrainerWithCategories(args.config)
    trainer.train()
