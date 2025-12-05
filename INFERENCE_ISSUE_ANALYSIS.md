# Inference Issue Analysis & Solution

## Problem Summary

Your inference is producing **zero detections** with max confidence of 0.000.

## Root Cause Analysis

After debugging, I found two critical issues:

### 1. Text Encoder NOT Trained ❌

In `configs/full_lora_config.yaml`:
```yaml
apply_to_text_encoder: false  # ← This is the problem!
```

**Impact**: The model cannot understand text prompts like "crack" because the text encoder wasn't adapted during training. Text prompts are ignored.

### 2. Model Outputs Indicate Poor Training

Debug results show:
- `pred_logits`: -10 to -11 (should be closer to 0)
- Scores after sigmoid: 0.00001 - 0.00003 (should be > 0.5 for detections)
- Semantic segmentation: also near-zero probabilities

**This suggests the model didn't train properly or didn't converge.**

## Solution: Retrain with Correct Configuration

### Step 1: Use the Fixed Config

I've created `configs/crack_detection_config.yaml` with:
- ✅ `apply_to_text_encoder: true` - Enables text prompt support
- ✅ Higher rank (16) for better adaptation
- ✅ 50 epochs for better convergence
- ✅ Optimized learning rate (1e-4)

### Step 2: Retrain the Model

```bash
# Retrain with fixed configuration
python3 train_sam3_lora_native.py --config configs/crack_detection_config.yaml
```

Expected output:
```
Applied LoRA to XX modules...
Trainable params: ~2-3% (should be low but >1% due to text encoder)
Starting training for 50 epochs...

Epoch 1: loss should decrease from ~2.0 to <1.0
Epoch 5: loss should be <0.5
Epoch 10+: loss should stabilize around 0.1-0.3
```

###  Step 3: Monitor Training

Watch for:
- ✅ Loss decreasing over epochs
- ✅ Validation loss improving
- ✅ Training completing without errors

### Step 4: Test Inference

```bash
python3 inference_lora.py \
  --config configs/crack_detection_config.yaml \
  --image /workspace/sam3_lora/data/test/05274_jpg.rf.49df85e9e38c9cb57651cb63f161309f.jpg \
  --prompt "crack" \
  --output crack_detection.png \
  --threshold 0.3
```

Expected: Detections with confidence > 0.3

## Why the Original Training Failed

1. **Text encoder disabled**: Model couldn't use "crack" prompt
2. **Possibly insufficient epochs**: 20 epochs may not be enough
3. **Data format issues**: Training data might need verification

## Verification Checklist

After retraining, check:

- [ ] Training completed all epochs without errors
- [ ] Final training loss < 0.5
- [ ] Validation loss decreasing
- [ ] Inference produces detections with confidence > 0.3
- [ ] Text prompts like "crack", "defect" work correctly

## Alternative: Try Without Text Prompts

If you need immediate results, try automatic segmentation (though results will be poor with current model):

```bash
# Lower threshold to see anything
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --image test.jpg \
  --threshold 0.0001 \
  --output test_output.png
```

But I **strongly recommend retraining** with the fixed config.

## Quick Test Before Full Retraining

Test if your data loads correctly:

```bash
python3 -c "
from pathlib import Path
import json

# Check train data
train_imgs = list(Path('data/train/images').glob('*.jpg'))
train_annots = list(Path('data/train/annotations').glob('*.json'))

print(f'Train images: {len(train_imgs)}')
print(f'Train annotations: {len(train_annots)}')

# Check one annotation
if train_annots:
    with open(train_annots[0]) as f:
        data = json.load(f)
    print(f'\\nSample annotation keys: {list(data.keys())}')
    print(f'Annotations count: {len(data.get(\"annotations\", []))}')
"
```

## Contact

If retraining still doesn't work, check:
1. Training logs for errors
2. Data annotation format
3. GPU memory (reduce batch_size if OOM)

---

**Created**: 2025-12-05
**Issue**: Zero detections in inference
**Status**: Waiting for retrain with fixed config
