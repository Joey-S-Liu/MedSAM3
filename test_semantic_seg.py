#!/usr/bin/env python3
"""
Test semantic segmentation output
"""

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from torchvision.transforms import v2

config_path = "configs/full_lora_config.yaml"
weights_path = "outputs/sam3_lora_full/best_lora_weights.pt"
image_path = "/workspace/sam3_lora/data/test/05274_jpg.rf.49df85e9e38c9cb57651cb63f161309f.jpg"

print("Testing Semantic Segmentation Output")
print("=" * 80)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resolution = 1008

# Build and load model (same as before)
model = build_sam3_image_model(
    device=device.type,
    compile=False,
    load_from_HF=True,
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    eval_mode=True
)

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
load_lora_weights(model, weights_path)
model.to(device)
model.eval()

# Load image
pil_image = PILImage.open(image_path).convert("RGB")
orig_w, orig_h = pil_image.size
pil_image_resized = pil_image.resize((resolution, resolution), PILImage.BILINEAR)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image_tensor = transform(pil_image_resized)
image_obj = Image(data=image_tensor, objects=[], size=(resolution, resolution))

query = FindQueryLoaded(
    query_text="crack",
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

# Run inference
batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
input_batch = batch_dict["input"]

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

with torch.no_grad():
    outputs_list = model(input_batch)

outputs = outputs_list[-1]

# Get semantic segmentation
semantic_seg = outputs['semantic_seg']  # [1, 1, H, W]
print(f"Semantic seg shape: {semantic_seg.shape}")
print(f"Semantic seg range: {semantic_seg.min().item():.2f} to {semantic_seg.max().item():.2f}")

# Apply sigmoid to get probabilities
semantic_prob = torch.sigmoid(semantic_seg)
print(f"Semantic prob range: {semantic_prob.min().item():.4f} to {semantic_prob.max().item():.4f}")

# Convert to numpy and threshold
semantic_mask = semantic_prob[0, 0].cpu().numpy()
binary_mask = semantic_mask > 0.5

print(f"Pixels above 0.5: {binary_mask.sum()} / {binary_mask.size} ({100 * binary_mask.sum() / binary_mask.size:.2f}%)")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(pil_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(semantic_mask, cmap="jet")
axes[1].set_title("Semantic Segmentation (Raw)")
axes[1].axis("off")
axes[1].colorbar = plt.colorbar(axes[1].imshow(semantic_mask, cmap="jet"), ax=axes[1])

axes[2].imshow(binary_mask, cmap="gray")
axes[2].set_title("Binary Mask (threshold=0.5)")
axes[2].axis("off")

axes[3].imshow(pil_image_resized)
axes[3].imshow(binary_mask, alpha=0.5, cmap="Reds")
axes[3].set_title("Overlay")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("semantic_seg_test.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved visualization to semantic_seg_test.png")
plt.close()
