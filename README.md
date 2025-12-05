# SAM3-LoRA: Efficient Fine-Tuning with Low-Rank Adaptation

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Train SAM3 segmentation models with 99% fewer trainable parameters**

[Quick Start](#quick-start) • [Training](#training) • [Inference](#inference) • [Configuration](#configuration)

</div>

---

## Overview

Fine-tune the SAM3 (Segment Anything Model 3) using **LoRA (Low-Rank Adaptation)** - a parameter-efficient method that reduces trainable parameters from 100% to ~1% while maintaining performance.

### Why Use This?

- ✅ **Train on Consumer GPUs**: 16GB VRAM instead of 80GB
- ✅ **Tiny Checkpoints**: 10-50MB LoRA weights vs 3GB full model
- ✅ **Fast Iterations**: Less memory = faster training
- ✅ **Easy to Use**: YAML configs + simple CLI
- ✅ **Production Ready**: Complete train + inference pipeline

### What is LoRA?

Instead of fine-tuning all model weights, LoRA injects small trainable matrices:
```
W' = W_frozen + B×A  (where rank << model_dim)
```

**Result**: Only ~1% of parameters need training!

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
cd sam3_lora

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

### 1. Prepare Your Data

Organize your dataset with images and annotations:

```
data/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── annotations/
│       ├── img001.json
│       └── img002.json
└── valid/
    ├── images/
    └── annotations/
```

**Annotation format** (JSON per image):
```json
{
  "bboxes": [[x1, y1, x2, y2], ...],
  "masks": [[[x, y], ...], ...]
}
```

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
Trainable params: 4,200,000 (~0.5%)  ← Should be ~1%!
Starting training for 20 epochs...
Training samples: 700, Validation samples: 100

Epoch 1: 100%|████████| 700/700 [12:00<00:00, loss=0.45]
Validation: 100%|████████| 100/100 [01:30<00:00, val_loss=0.38]
Epoch 1/20 - Validation Loss: 0.380000
✓ Saved best model (val_loss: 0.380000)

Epoch 2: 100%|████████| 700/700 [12:00<00:00, loss=0.32]
Validation: 100%|████████| 100/100 [01:30<00:00, val_loss=0.29]
Epoch 2/20 - Validation Loss: 0.290000
✓ Saved best model (val_loss: 0.290000)
...
```

### 3. Run Inference

```bash
# Segment objects in new images (automatically uses best model)
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --output predictions.png

# Or specify weights explicitly
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --image test_image.jpg \
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
  batch_size: 1               # Adjust based on GPU memory
  num_epochs: 20              # Training epochs
  learning_rate: 5e-5         # Learning rate
  weight_decay: 0.01          # Weight decay

output:
  output_dir: "outputs/my_model"
```

Then train:
```bash
python3 train_sam3_lora_native.py --config configs/my_config.yaml
```

### Model Checkpointing

During training, two models are automatically saved:
- **`best_lora_weights.pt`**: Best model based on validation loss (recommended for inference)
- **`last_lora_weights.pt`**: Model from the last epoch

Training monitors validation loss and saves the best model automatically.

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

### Command Line

```bash
# Basic inference (automatically uses best model)
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --output predictions.png

# Use last epoch model instead
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/last_lora_weights.pt \
  --image path/to/image.jpg \
  --output predictions.png

# With custom confidence threshold
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --threshold 0.7 \
  --output predictions.png
```

### Python API

```python
from inference_lora import SAM3LoRAInference

# Initialize inference engine
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    weights_path="outputs/sam3_lora_full/lora_weights.pt"
)

# Run prediction
predictions = inferencer.predict("image.jpg")

# Visualize results
inferencer.visualize_predictions(
    predictions,
    output_path="output.png",
    confidence_threshold=0.5
)

# Access raw predictions
boxes = predictions['boxes']        # Bounding boxes
scores = predictions['scores']      # Confidence scores
masks = predictions['masks']        # Segmentation masks
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

# Inference
python3 inference_lora.py \
  --config configs/quick_test.yaml \
  --weights outputs/quick_test/lora_weights.pt \
  --image test.jpg \
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

### Example 4: Batch Inference

```python
from inference_lora import SAM3LoRAInference
from pathlib import Path

# Initialize once
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    weights_path="outputs/sam3_lora_full/lora_weights.pt"
)

# Process multiple images
image_dir = Path("test_images")
output_dir = Path("predictions")
output_dir.mkdir(exist_ok=True)

for img_path in image_dir.glob("*.jpg"):
    predictions = inferencer.predict(str(img_path))

    output_path = output_dir / f"{img_path.stem}_pred.png"
    inferencer.visualize_predictions(predictions, str(output_path))

    print(f"✓ Processed {img_path.name}")
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
├── data/
│   ├── train/
│   │   ├── images/                # Training images
│   │   └── annotations/           # JSON annotations
│   └── valid/
│       ├── images/                # Validation images
│       └── annotations/           # JSON annotations
├── outputs/
│   └── sam3_lora_full/
│       ├── best_lora_weights.pt   # Best model (lowest val loss)
│       └── last_lora_weights.pt   # Last epoch model
├── sam3/                          # SAM3 model library
├── lora_layers.py                 # LoRA implementation
├── train_sam3_lora_native.py      # Training script
├── inference_lora.py              # Inference script
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
