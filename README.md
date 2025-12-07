# SAM3-LoRA: Efficient Fine-Tuning with Low-Rank Adaptation

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Train SAM3 segmentation models with 99% fewer trainable parameters**

[Quick Start](#quick-start) • [Architecture](#architecture) • [Training](#training) • [Inference](#inference) • [Examples](#real-world-example-concrete-crack-detection) • [Configuration](#configuration)

</div>

---

## Overview

Fine-tune the SAM3 (Segment Anything Model 3) using **LoRA (Low-Rank Adaptation)** - a parameter-efficient method that reduces trainable parameters from 100% to ~1% while maintaining performance.

### Recent Updates

**2025-12-07**:
- Fixed missing `rle_encode` import in `train_sam3_lora_native.py`
- Verified category-aware text prompts work correctly in both training and validation
- All mask processing operations now working properly without errors

### Why Use This?

- ✅ **Train on Consumer GPUs**: 16GB VRAM instead of 80GB
- ✅ **Tiny Checkpoints**: 10-50MB LoRA weights vs 3GB full model
- ✅ **Fast Iterations**: Less memory = faster training
- ✅ **Easy to Use**: YAML configs + simple CLI
- ✅ **Production Ready**: Complete train + inference pipeline
- ✅ **Real Applications**: Crack detection, defect inspection, and more

### What is LoRA?

Instead of fine-tuning all model weights, LoRA injects small trainable matrices:
```
W' = W_frozen + B×A  (where rank << model_dim)
```

**Result**: Only ~1% of parameters need training!

### Architecture

SAM3-LoRA applies Low-Rank Adaptation to key components of the SAM3 architecture:

<div align="center">
<img src="asset/Screenshot 2568-12-06 at 07.00.16.png" alt="SAM3 Architecture with LoRA" width="900">
<br>
<em>SAM3 Model Architecture with Full LoRA Adaptation</em>
</div>

<br>

**LoRA Adapters Applied To:**

| Component | Description | LoRA Impact |
|-----------|-------------|-------------|
| **Vision Encoder (ViT)** | Extracts visual features from input images | High - Primary feature learning |
| **Text Encoder** | Processes text prompts for guided segmentation | Medium - Semantic understanding |
| **Geometry Encoder** | Handles geometric prompts (boxes, points) | Medium - Spatial reasoning |
| **DETR Encoder** | Transformer encoder for object detection | High - Scene understanding |
| **DETR Decoder** | Transformer decoder for object queries | High - Object localization |
| **Mask Decoder** | Generates segmentation masks | High - Fine-grained segmentation |

**Data Flow:**
1. **Input**: Image + Text/Geometric prompts
2. **Encoding**: Multiple encoders process different modalities
3. **Transformation**: DETR encoder-decoder refines representations
4. **Output**: High-quality segmentation masks

**LoRA Benefits:**
- ✅ Only ~1% parameters trainable (frozen base + small adapters)
- ✅ Adapters can be swapped for different tasks
- ✅ Original model weights preserved
- ✅ Efficient storage (10-50MB vs 3GB full model)

---

## Installation

### Prerequisites

Before installing, you need to:

1. **Request SAM3 Access on Hugging Face**
   - Go to [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3)
   - Click "Request Access" and accept the license terms
   - Wait for approval (usually instant to a few hours)

2. **Get Your Hugging Face Token**
   - Go to [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens)
   - Create a new token or use existing one
   - Copy the token (you'll need it in the next step)

### Install

```bash
# Clone repository
git clone https://github.com/yourusername/sam3_lora.git
cd SAM3_LoRA

# Install dependencies
pip install -e .

# Login to Hugging Face
huggingface-cli login
# Paste your token when prompted
```

**Alternative login method:**
```bash
# Or set token as environment variable
export HF_TOKEN="your_token_here"
```

**Requirements**: Python 3.8+, PyTorch 2.0+, CUDA (optional), Hugging Face account with SAM3 access

### Verification

Verify your setup is complete:

```bash
# Test Hugging Face login
huggingface-cli whoami

# Test SAM3 access (should not give access error)
python3 -c "from transformers import AutoModel; print('✓ SAM3 accessible')"
```

If you see errors, review the [Troubleshooting](#troubleshooting) section.

---

## Quick Start

> **⚠️ Important**: Make sure you've completed the [Installation](#installation) steps, including Hugging Face login, before proceeding.

**Example Result**: Train a model to detect concrete cracks with just ~1% trainable parameters!

<div align="center">
<img src="asset/output.png" alt="Example: Concrete Crack Detection" width="600">
<br>
<em>Detection: "concrete crack" with 0.32 confidence • Precise segmentation mask</em>
</div>

<br>

### 1. Prepare Your Data

Organize your dataset in **COCO format** with a single annotation file per split:

```
data/
├── train/                    # Required
│   ├── img001.jpg
│   ├── img002.jpg
│   └── _annotations.coco.json
├── valid/                    # Optional but recommended
│   ├── img001.jpg
│   ├── img002.jpg
│   └── _annotations.coco.json
└── test/                     # Optional
    ├── img001.jpg
    └── _annotations.coco.json
```

> **Note**: Validation data (`data/valid/`) is **optional** but strongly recommended for monitoring training progress and preventing overfitting.

**COCO Annotation Format** (`_annotations.coco.json`):
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "img001.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "defect"}
  ]
}
```

**Supported Segmentation Formats:**
- **Polygon**: `"segmentation": [[x1, y1, x2, y2, ...]]` (list of polygons)
- **RLE**: `"segmentation": {"counts": "...", "size": [h, w]}` (run-length encoded)

### 2. Train Your Model

```bash
# Train with default config
python3 train_sam3_lora_native.py

# Or specify custom config
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

**Expected output:**
```
Building SAM3 model...
Applying LoRA...
Applied LoRA to 64 modules
Trainable params: 11,796,480 (1.38%)

Loading training data from /workspace/data2...
Loaded COCO dataset: train split
  Images: 778
  Annotations: 1631
  Categories: {0: 'CRACKS', 1: 'CRACKS', 2: 'JOINT', 3: 'LOCATION', 4: 'MARKING'}

Loading validation data from /workspace/data2...
Loaded COCO dataset: valid split
  Images: 152
  Annotations: 298
Found validation data: 152 images
Starting training for 200 epochs...
Training samples: 778, Validation samples: 152

Epoch 1: 100%|████████| 98/98 [07:47<00:00, loss=140]
Validation: 100%|████████| 19/19 [00:32<00:00, val_loss=23.7]

Epoch 1/200 - Train Loss: 156.234567, Val Loss: 17.032280
✓ New best model saved (val_loss: 17.032280)

Epoch 2: 100%|████████| 98/98 [07:24<00:00, loss=167]
Validation: 100%|████████| 19/19 [00:31<00:00, val_loss=20.1]

Epoch 2/200 - Train Loss: 142.891234, Val Loss: 15.641912
✓ New best model saved (val_loss: 15.641912)
...
```

**Validation Strategy (Following SAM3):**
- **During training**: Only validation **loss** is computed (fast, no NMS or metrics)
- **After training**: Run `validate_sam3_lora.py` for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training while still monitoring overfitting via validation loss

### 3. Run Inference

```bash
# Basic inference (automatically uses best model)
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --output predictions.png

# With text prompt for better accuracy
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --prompt "yellow school bus" \
  --output predictions.png

# Multiple prompts to detect different objects
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --prompt "crack" "defect" "damage" \
  --output predictions.png
```

---

## Training

### Basic Training

```bash
# Use default configuration
python3 train_sam3_lora_native.py
```

### Custom Configuration

Create a config file (e.g., `configs/my_config.yaml`):

```yaml
lora:
  rank: 16                    # LoRA rank (higher = more capacity)
  alpha: 32                   # Scaling factor (typically 2×rank)
  dropout: 0.1                # Dropout for regularization
  target_modules:             # Which layers to adapt
    - "q_proj"                # Query projection
    - "k_proj"                # Key projection
    - "v_proj"                # Value projection
    - "fc1"                   # MLP layer 1
    - "fc2"                   # MLP layer 2

  # Which model components to apply LoRA to
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  apply_to_detr_encoder: false
  apply_to_detr_decoder: false

training:
  data_dir: "/path/to/data"   # Root directory with train/valid/test folders
  batch_size: 8               # Adjust based on GPU memory
  num_epochs: 200             # Training epochs
  learning_rate: 1e-5         # Learning rate (lower for stability)
  weight_decay: 0.01          # Weight decay

output:
  output_dir: "outputs/my_model"
```

**Important Notes:**
- Use **generic text prompts** like `"object"` for best results (SAM models work better with simple terms)
- The dataset automatically extracts category names from COCO annotations
- Text prompts during training are class-agnostic to leverage pre-trained knowledge

Then train:
```bash
python3 train_sam3_lora_native.py --config configs/my_config.yaml
```

### Model Checkpointing

During training, two models are automatically saved:
- **`best_lora_weights.pt`**: Best model based on validation loss (saved only when validation loss improves)
- **`last_lora_weights.pt`**: Model from the last epoch (saved after every epoch)

**With validation data**: Training monitors validation **loss only** (fast). Best model is saved when validation loss decreases.

**Without validation data**: Training continues normally but saves the last epoch as both files. You'll see:
```
⚠️  No validation data found - training without validation
...
ℹ️  No validation data - consider adding data/valid/ for better model selection
```

### Full Evaluation After Training

**Important**: Training only computes validation loss (no metrics). After training completes, run the standalone evaluation script to get full metrics:

```bash
# Evaluate best model with full metrics (mAP, cgF1) and NMS
python3 validate_sam3_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --data_dir /workspace/data2 \
  --split valid

# Evaluate on test set
python3 validate_sam3_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --data_dir /workspace/data2 \
  --split test
```

**This will compute:**
- COCO mAP metrics (IoU 0.50:0.95, 0.50, 0.75)
- Category-agnostic F1 scores (cgF1, cgF1@50, cgF1@75)
- Applies SAM3's NMS filtering (prob_threshold=0.3, iou_threshold=0.7)
- Generates detailed predictions

**Why separate evaluation?**
- ⚡ **Faster training**: No expensive metric computation during training
- 📊 **Better monitoring**: Validation loss is sufficient to detect overfitting
- 🎯 **Accurate metrics**: Full evaluation with proper NMS and post-processing

### Training Tips

**Starting Out:**
- Use `rank: 4` or `rank: 8` for quick experiments
- Set `num_epochs: 5` for initial tests
- Monitor that trainable params are ~0.5-2%
- Watch validation loss - it should decrease over epochs

**Production Training:**
- Increase to `rank: 16` or `rank: 32` for better performance
- Use `num_epochs: 20-50` depending on dataset size
- Enable more components (DETR encoder/decoder) if needed
- Use early stopping if validation loss stops improving

**Troubleshooting:**
- **Loss too low (< 0.001)**: Model might be overfitting, reduce rank or add regularization
- **Val loss > Train loss**: Normal, indicates some overfitting
- **Val loss increasing**: Overfitting! Reduce rank, add dropout, or stop training
- **Loss not decreasing**: Increase learning rate or rank
- **OOM errors**: Reduce batch size or rank
- **63% trainable params**: Bug! Should be ~1% - make sure base model is frozen

---

## Inference

Run inference on new images using your trained LoRA model. The new `infer_sam.py` script is based on official SAM3 patterns and supports **multiple text prompts** for better accuracy.

### Command Line

```bash
# Basic inference (automatically uses best model)
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --output predictions.png

# With text prompt (recommended for better accuracy)
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "yellow school bus" \
  --output predictions.png

# Multiple prompts to detect different object types
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image street_scene.jpg \
  --prompt "car" "person" "bus" \
  --output segmentation.png

# Use last epoch model instead
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/last_lora_weights.pt \
  --image path/to/image.jpg \
  --prompt "person with red backpack" \
  --output predictions.png

# With custom confidence threshold
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "building" \
  --threshold 0.3 \
  --output predictions.png

# Mask-only visualization (no boxes)
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "crack" \
  --no-boxes \
  --output masks_only.png
```

### Text Prompts

Text prompts help guide the model to segment specific objects more accurately. **New feature**: You can now use multiple prompts in a single command!

**Single prompt examples:**
- `"yellow school bus"` - Specific color and object type
- `"person wearing red hat"` - Object with distinctive features
- `"car"` - Simple, clear object type
- `"crack"` - For defect detection
- `"building with glass windows"` - Object with distinguishing features

**Multiple prompt examples:**
```bash
# Detect different defect types
--prompt "crack" "spalling" "corrosion"

# Detect multiple objects in street scenes
--prompt "car" "person" "traffic sign"
```

**Tips for better prompts:**
- Be specific but concise
- Include distinctive colors or features when relevant
- Use natural language descriptions
- For multiple prompts, order from most to least important
- Match the vocabulary to your training data

### Inference Parameters

| Parameter | Description | Example | Default |
|-----------|-------------|---------|---------|
| `--config` | Path to training config file | `configs/full_lora_config.yaml` | Required |
| `--weights` | Path to LoRA weights (optional) | `outputs/sam3_lora_full/best_lora_weights.pt` | Auto-detected |
| `--image` | Input image path | `test_image.jpg` | Required |
| `--prompt` | One or more text prompts | `"crack"` or `"crack" "defect"` | `"object"` |
| `--output` | Output visualization path | `predictions.png` | `output.png` |
| `--threshold` | Confidence threshold (0.0-1.0) | `0.3` | `0.5` |
| `--resolution` | Input resolution | `1008` | `1008` |
| `--no-boxes` | Don't show bounding boxes | - | False |
| `--no-masks` | Don't show segmentation masks | - | False |

### Python API

```python
from infer_sam import SAM3LoRAInference

# Initialize inference engine
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
    detection_threshold=0.5
)

# Run prediction with single text prompt
predictions = inferencer.predict(
    image_path="image.jpg",
    text_prompts=["yellow school bus"]
)

# Run prediction with multiple text prompts
predictions = inferencer.predict(
    image_path="image.jpg",
    text_prompts=["crack", "defect", "damage"]
)

# Visualize results
inferencer.visualize(
    predictions,
    output_path="output.png",
    show_boxes=True,
    show_masks=True
)

# Access predictions for each prompt
for idx, prompt in enumerate(["crack", "defect"]):
    result = predictions[idx]
    print(f"Prompt '{result['prompt']}':")
    print(f"  Detections: {result['num_detections']}")
    if result['num_detections'] > 0:
        print(f"  Boxes: {result['boxes'].shape}")      # [N, 4] in xyxy format
        print(f"  Scores: {result['scores'].shape}")    # [N]
        print(f"  Masks: {result['masks'].shape}")      # [N, H, W]
```

---

## Configuration

### LoRA Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `rank` | LoRA rank (bottleneck dimension) | 4, 8, 16, 32 |
| `alpha` | Scaling factor | 2×rank (e.g., 16 for rank=8) |
| `dropout` | Dropout probability | 0.0 - 0.1 |
| `target_modules` | Which layer types to adapt | q_proj, k_proj, v_proj, fc1, fc2 |

### Component Flags

| Flag | Description | When to Enable |
|------|-------------|----------------|
| `apply_to_vision_encoder` | Vision backbone | Always (main feature extractor) |
| `apply_to_mask_decoder` | Mask generation | Recommended for segmentation |
| `apply_to_detr_encoder` | Object detection encoder | For complex scenes |
| `apply_to_detr_decoder` | Object detection decoder | For complex scenes |
| `apply_to_text_encoder` | Text understanding | For text-based prompts |

### Preset Configurations

**Minimal (Fastest, Lowest Memory)**
```yaml
lora:
  rank: 4
  alpha: 8
  target_modules: ["q_proj", "v_proj"]
  apply_to_vision_encoder: true
  # All others: false
```

**Balanced (Recommended)**
```yaml
lora:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  # Others: false
```

**Maximum (Best Performance)**
```yaml
lora:
  rank: 32
  alpha: 64
  target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true
```

---

## Real-World Example: Concrete Crack Detection

SAM3-LoRA excels at detecting structural defects like cracks in concrete. Here's a real example:

<div align="center">
<img src="asset/output.png" alt="Concrete Crack Detection" width="800">
</div>

**Detection Results:**
- **Prompt**: "concrete crack"
- **Confidence**: 0.32 (using threshold 0.3)
- **Segmentation**: Precise mask following the crack pattern
- **Application**: Infrastructure inspection, structural health monitoring

**Run this example:**
```bash
# Detect cracks in concrete structures
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/concrete.jpg \
  --prompt "concrete crack" \
  --threshold 0.3 \
  --output crack_detection.png

# Detect multiple defect types
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/concrete.jpg \
  --prompt "crack" "spalling" "corrosion" \
  --threshold 0.3 \
  --output defect_analysis.png
```

**Use Cases:**
- 🏗️ Civil engineering inspection
- 🌉 Bridge and infrastructure monitoring
- 🏢 Building maintenance
- 🛣️ Road surface analysis
- 🏭 Industrial facility assessment

---

## Examples

### Example 1: Quick Test (5 Epochs)

```bash
# Create minimal config
cat > configs/quick_test.yaml << EOF
lora:
  rank: 4
  alpha: 8
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  apply_to_vision_encoder: true
  apply_to_mask_decoder: false

training:
  batch_size: 1
  num_epochs: 5
  learning_rate: 1e-4
  weight_decay: 0.01

output:
  output_dir: "outputs/quick_test"
EOF

# Train
python3 train_sam3_lora_native.py --config configs/quick_test.yaml

# Inference with text prompt
python3 infer_sam.py \
  --config configs/quick_test.yaml \
  --weights outputs/quick_test/best_lora_weights.pt \
  --image test.jpg \
  --prompt "car" \
  --output result.png

# Multiple prompts
python3 infer_sam.py \
  --config configs/quick_test.yaml \
  --image test.jpg \
  --prompt "car" "person" "bus" \
  --output result.png
```

### Example 2: Production Training

```bash
# Create production config
cat > configs/production.yaml << EOF
lora:
  rank: 32
  alpha: 64
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true

training:
  batch_size: 2
  num_epochs: 50
  learning_rate: 3e-5
  weight_decay: 0.01

output:
  output_dir: "outputs/production"
EOF

# Train
python3 train_sam3_lora_native.py --config configs/production.yaml
```

### Example 3: Programmatic Training

```python
from train_sam3_lora_native import SAM3TrainerNative

# Create trainer
trainer = SAM3TrainerNative("configs/full_lora_config.yaml")

# Train
trainer.train()

# Weights saved to: outputs/sam3_lora_full/lora_weights.pt
```

### Example 4: Batch Inference with Text Prompts

```python
from infer_sam import SAM3LoRAInference
from pathlib import Path

# Initialize once
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    weights_path="outputs/sam3_lora_full/best_lora_weights.pt"
)

# Process multiple images with same prompt
image_dir = Path("test_images")
output_dir = Path("predictions")
output_dir.mkdir(exist_ok=True)

for img_path in image_dir.glob("*.jpg"):
    predictions = inferencer.predict(
        str(img_path),
        text_prompts=["car"]
    )

    output_path = output_dir / f"{img_path.stem}_pred.png"
    inferencer.visualize(
        predictions,
        str(output_path)
    )

    print(f"✓ Processed {img_path.name}")

# Process with multiple prompts per image
for img_path in image_dir.glob("*.jpg"):
    # Detect multiple object types at once
    predictions = inferencer.predict(
        str(img_path),
        text_prompts=["crack", "defect", "damage"]
    )

    output_path = output_dir / f"{img_path.stem}_multi.png"
    inferencer.visualize(predictions, str(output_path))

    # Print summary
    for idx in range(3):
        result = predictions[idx]
        print(f"  {result['prompt']}: {result['num_detections']} detections")
```

---

## Advanced Usage

### Apply LoRA to Custom Models

```python
from lora_layers import LoRAConfig, apply_lora_to_model, count_parameters
import torch.nn as nn

# Your PyTorch model
model = YourModel()

# Configure LoRA
lora_config = LoRAConfig(
    rank=8,
    alpha=16,
    dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=False,
    apply_to_detr_decoder=False,
    apply_to_mask_decoder=False,
)

# Apply LoRA (automatically freezes base model)
model = apply_lora_to_model(model, lora_config)

# Check trainable parameters
stats = count_parameters(model)
print(f"Trainable: {stats['trainable_parameters']:,} / {stats['total_parameters']:,}")
print(f"Percentage: {stats['trainable_percentage']:.2f}%")

# Train normally
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

### Save and Load LoRA Weights

```python
from lora_layers import save_lora_weights, load_lora_weights

# Save only LoRA parameters (small file!)
save_lora_weights(model, "my_lora_weights.pt")

# Load into new model
load_lora_weights(model, "my_lora_weights.pt")
```

---

## Project Structure

```
sam3_lora/
├── configs/
│   └── full_lora_config.yaml      # Default training config
├── data/                          # COCO format dataset
│   ├── train/
│   │   ├── img001.jpg             # Training images
│   │   ├── img002.jpg
│   │   └── _annotations.coco.json # COCO annotations
│   ├── valid/
│   │   ├── img001.jpg             # Validation images
│   │   ├── img002.jpg
│   │   └── _annotations.coco.json # COCO annotations
│   └── test/
│       ├── img001.jpg             # Test images (optional)
│       └── _annotations.coco.json # COCO annotations
├── outputs/
│   └── sam3_lora_full/
│       ├── best_lora_weights.pt   # Best model (lowest val loss)
│       └── last_lora_weights.pt   # Last epoch model
├── sam3/                          # SAM3 model library
├── lora_layers.py                 # LoRA implementation
├── train_sam3_lora_native.py      # Training script (computes validation loss only)
├── validate_sam3_lora.py          # Full evaluation script (mAP, cgF1, NMS)
├── validate_single_image.py       # Single image validation with visualization
├── infer_sam.py                   # Inference script (recommended)
├── inference_lora.py              # Legacy inference script
├── README_INFERENCE.md            # Detailed inference guide
└── README.md                      # This file
```

---

## Troubleshooting

### Common Issues

**1. Hugging Face Authentication Error**
```
Error: Access denied to facebook/sam3
```
**Solution:**
- Make sure you've requested access at https://huggingface.co/facebook/sam3
- Wait for approval (check your email)
- Run `huggingface-cli login` and paste your token
- Or set: `export HF_TOKEN="your_token"`

**2. Import Errors**
```bash
# Make sure package is installed
pip install -e .
```

**3. CUDA Out of Memory**
```yaml
# Reduce batch size and rank in config
training:
  batch_size: 1

lora:
  rank: 4
```

**4. Very Low Loss (< 0.001)**
- Model may be overfitting
- Reduce LoRA rank
- Add more dropout
- Check if base model is properly frozen

**5. Loss Not Decreasing**
- Increase learning rate
- Increase LoRA rank
- Train for more epochs
- Check data quality

**6. Wrong Number of Trainable Parameters**
```
Expected: ~0.5-2% (for rank 4-16)
If you see 63%: Base model not frozen (bug fixed in latest version)
```

**7. No Validation Data**
```
⚠️ No validation data found - training without validation
```
**Solution:**
- Create `data/valid/` directory with same structure as `data/train/`
- Split your data: ~80% train, ~20% validation
- Training will work without validation but you won't see validation metrics

**8. Annotation Format Errors**
```
FileNotFoundError: COCO annotation file not found: /path/to/data/train/_annotations.coco.json
```
**Solution:**
- Ensure your data is in COCO format with `_annotations.coco.json` in each split folder
- Each split (train/valid/test) needs its own annotation file
- Images should be in the same directory as the annotation file
- Supported segmentation formats: polygon lists or RLE dictionaries

**9. Want to See mAP/cgF1 During Training?**
**Solution:**
- Training only computes validation loss (fast, following SAM3's approach)
- After training, run `validate_sam3_lora.py` for full metrics with NMS
- This approach significantly speeds up training while still monitoring overfitting
- Validation loss is sufficient to detect overfitting and select best model

### Performance Benchmarks

| Configuration | Trainable Params | Checkpoint Size | GPU Memory | Speed |
|---------------|------------------|-----------------|------------|-------|
| Minimal (r=4) | ~0.2% | ~10 MB | 8 GB | Fast |
| Balanced (r=8) | ~0.5% | ~20 MB | 12 GB | Medium |
| Full (r=16) | ~1.0% | ~40 MB | 16 GB | Slower |
| Maximum (r=32) | ~2.0% | ~80 MB | 20 GB | Slowest |

*Benchmarks on NVIDIA RTX 3090*

---

## Citation

If you use this work, please cite:

```bibtex
@software{sam3_lora,
  title = {SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning},
  author = {AI Research Group, KMUTT},
  year = {2025},
  organization = {King Mongkut's University of Technology Thonburi},
  url = {https://github.com/yourusername/sam3_lora}
}
```

### References

- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685) - "LoRA: Low-Rank Adaptation of Large Language Models"
- **SAM**: [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) - "Segment Anything"
- **SAM3**: Meta AI Research

---

## Credits

**Made by AI Research Group, KMUTT**
*King Mongkut's University of Technology Thonburi*

---

## License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Version**: 1.0.0
**Python**: 3.8+
**PyTorch**: 2.0+

Built with ❤️ for the research community

[⬆ Back to Top](#sam3-lora-efficient-fine-tuning-with-low-rank-adaptation)

</div>
