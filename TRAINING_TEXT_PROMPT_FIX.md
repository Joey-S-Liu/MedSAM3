# How Text Prompts Work in Your Training

## Current Situation

### Your Data:
- COCO category: **"object"** (not "crack")
- All cracks are labeled as category "object"

### Training Code (line 119):
```python
query_text="object"  # Hardcoded generic prompt
```

## What Happens with `apply_to_text_encoder: false`

**During Training:**
1. ✅ Text prompt "object" IS used
2. ✅ Text encoder processes "object" → creates embeddings
3. ❌ Text encoder is FROZEN (weights don't change)
4. ✅ Vision encoder learns: "when I see 'object' embeddings, segment cracks"

**During Inference with "crack":**
1. ❌ Text encoder (frozen) produces DIFFERENT embeddings for "crack"
2. ❌ Model doesn't recognize because it learned "object" embeddings, not "crack"
3. ❌ No detections!

## What Happens with `apply_to_text_encoder: true`

**During Training:**
1. ✅ Text prompt "object" is used
2. ✅ Text encoder processes "object"
3. ✅ Text encoder ADAPTS to understand "object" better for your data
4. ✅ Vision encoder aligns with adapted "object" embeddings

**During Inference with "crack":**
- ⚠️ Slightly better, but still problematic
- The adapted text encoder might produce better embeddings
- But model was never trained to respond to "crack" specifically

---

## Solutions

### ✅ Solution 1: Use "object" at Inference (Easiest)

Since your model was trained with "object", use it at inference:

```bash
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --image test.jpg \
  --prompt "object" \    # ← Match training!
  --output result.png
```

### ✅ Solution 2: Change COCO Category to "crack"

Update your COCO annotations:

```python
import json

# Load COCO file
with open('data/train/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Change category name
for cat in coco_data['categories']:
    if cat['name'] == 'object':
        cat['name'] = 'crack'

# Save
with open('data/train/_annotations.coco.json', 'w') as f:
    json.dump(coco_data, f)
```

Then modify training code to use the category name instead of hardcoded "object".

### ✅ Solution 3: Train with Multiple Prompts (Best)

Modify training to randomly use different prompts:

```python
# In train_sam3_lora_native.py, line 119
import random
prompts = ["crack", "object", "defect", "damage", "concrete crack"]
query_text = random.choice(prompts)  # Random prompt each time
```

This teaches the model to respond to various terms.

### ✅ Solution 4: Enable Text Encoder + Use "crack" Consistently

1. Change COCO category to "crack"
2. Enable `apply_to_text_encoder: true`
3. Modify training to use category name
4. Retrain

Then at inference, "crack" will work properly.

---

## Recommendation

**For immediate results:**
- Use `--prompt "object"` at inference (matches your training)

**For production use:**
- Retrain with `apply_to_text_encoder: true`
- Use multiple varied prompts during training
- This makes model robust to different phrasings

---

## Key Insight

> **The text prompt used at inference MUST match (or be similar to) the prompts used during training!**

If you train with "object" but inference with "crack":
- Frozen text encoder: ❌ Completely different embeddings → no detection
- Trainable text encoder: ⚠️ Slightly better but still mismatched → poor detection

**Solution:** Match your inference prompts to training prompts, OR train with diverse prompts.
