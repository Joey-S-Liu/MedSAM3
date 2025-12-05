#!/usr/bin/env python3
"""
Debug inference script to check raw model outputs
"""

import yaml
import torch
import numpy as np
from PIL import Image as PILImage
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from torchvision.transforms import v2

# Load config
config_path = "configs/full_lora_config.yaml"
weights_path = "outputs/sam3_lora_full/best_lora_weights.pt"
image_path = "/workspace/sam3_lora/data/test/05274_jpg.rf.49df85e9e38c9cb57651cb63f161309f.jpg"

print("=" * 80)
print("DEBUG INFERENCE")
print("=" * 80)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resolution = 1008

# Build Model
print("\n1. Building SAM3 model...")
model = build_sam3_image_model(
    device=device.type,
    compile=False,
    load_from_HF=True,
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    eval_mode=True
)

# Apply LoRA
print("\n2. Applying LoRA configuration...")
lora_cfg = config["lora"]
lora_config = LoRAConfig(
    rank=lora_cfg["rank"],
    alpha=lora_cfg["alpha"],
    dropout=0.0,
    target_modules=lora_cfg["target_modules"],
    apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
    apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
    apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
    apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
    apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
    apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
)
model = apply_lora_to_model(model, lora_config)

# Load LoRA weights
print(f"\n3. Loading LoRA weights from {weights_path}...")
load_lora_weights(model, weights_path)

model.to(device)
model.eval()

# Prepare image
print(f"\n4. Loading image: {image_path}")
pil_image = PILImage.open(image_path).convert("RGB")
orig_w, orig_h = pil_image.size
print(f"   Original size: {orig_w} x {orig_h}")

pil_image_resized = pil_image.resize((resolution, resolution), PILImage.BILINEAR)
print(f"   Resized to: {resolution} x {resolution}")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image_tensor = transform(pil_image_resized)

# Create Image object
image_obj = Image(
    data=image_tensor,
    objects=[],
    size=(resolution, resolution)
)

# Try different text prompts
test_prompts = ["crack", "object", "defect", "damage"]

for text_prompt in test_prompts:
    print(f"\n{'='*80}")
    print(f"5. Testing with text prompt: '{text_prompt}'")
    print(f"{'='*80}")

    # Create query
    query = FindQueryLoaded(
        query_text=text_prompt,
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

    datapoint = Datapoint(
        find_queries=[query],
        images=[image_obj],
        raw_images=[pil_image_resized]
    )

    # Collate into batch
    batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    input_batch = batch_dict["input"]

    # Move to device
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

    input_batch = move_to_device(input_batch, device)

    # Forward pass
    print("   Running inference...")
    with torch.no_grad():
        outputs_list = model(input_batch)

    print(f"   Number of output layers: {len(outputs_list)}")
    outputs = outputs_list[-1]

    print(f"\n   Output keys: {list(outputs.keys())}")

    # Check each output
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"     min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")
        else:
            print(f"   - {key}: type={type(value)}")

    # Examine logits and scores
    if 'pred_logits' in outputs:
        pred_logits = outputs['pred_logits']
        print(f"\n   Logits stats:")
        print(f"   - Shape: {pred_logits.shape}")
        print(f"   - Min: {pred_logits.min().item():.6f}")
        print(f"   - Max: {pred_logits.max().item():.6f}")
        print(f"   - Mean: {pred_logits.mean().item():.6f}")
        print(f"   - Std: {pred_logits.std().item():.6f}")

        # Check individual values
        print(f"\n   First 10 logit values (batch 0, query 0):")
        print(f"   {pred_logits[0, 0, :10].cpu().numpy()}")

        # Convert to scores (sigmoid)
        scores = torch.sigmoid(pred_logits)
        print(f"\n   Scores (after sigmoid):")
        print(f"   - Min: {scores.min().item():.6f}")
        print(f"   - Max: {scores.max().item():.6f}")
        print(f"   - Mean: {scores.mean().item():.6f}")
        print(f"   - Above 0.1: {(scores > 0.1).sum().item()}")
        print(f"   - Above 0.5: {(scores > 0.5).sum().item()}")

    if 'pred_boxes' in outputs:
        pred_boxes = outputs['pred_boxes']
        print(f"\n   Boxes stats:")
        print(f"   - Shape: {pred_boxes.shape}")
        print(f"   - First box (batch 0, query 0): {pred_boxes[0, 0].cpu().numpy()}")

    if 'pred_masks' in outputs:
        pred_masks = outputs['pred_masks']
        print(f"\n   Masks stats:")
        print(f"   - Shape: {pred_masks.shape}")
        print(f"   - Min: {pred_masks.min().item():.6f}")
        print(f"   - Max: {pred_masks.max().item():.6f}")
        print(f"   - Mean: {pred_masks.mean().item():.6f}")

print(f"\n{'='*80}")
print("DEBUG COMPLETE")
print(f"{'='*80}")
