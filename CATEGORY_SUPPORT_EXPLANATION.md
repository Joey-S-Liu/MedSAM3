# Category Support in SAM3 LoRA Training

## Your Question

**"Do class names get used during training even with `apply_to_text_encoder: false`?"**

## Short Answer

### ❌ **NO - The current training code does NOT use class names!**

The training is **hardcoded** to use "object" for ALL samples, regardless of actual categories in your data.

---

## Detailed Explanation

### Current Training Code Analysis

In `train_sam3_lora_native.py` (lines 119, 127):

```python
query = FindQueryLoaded(
    query_text="object",  # ← HARDCODED! Always "object"
    ...
    inference_metadata=InferenceMetadata(
        ...
        original_category_id=0,  # ← HARDCODED to 0!
        ...
    )
)
```

### What This Means:

| Data Has | Training Uses | Result |
|----------|---------------|--------|
| "crack" | "object" | ❌ Ignores "crack" |
| "damage" | "object" | ❌ Ignores "damage" |
| "spalling" | "object" | ❌ Ignores "spalling" |
| Multiple classes | "object" for all | ❌ Can't distinguish classes |

---

## Why This Happens

### 1. Annotation Format Issue

Your individual annotation files (`*.json`) **don't have category information**:

```json
{
  "annotations": [{
    "id": 44,
    "bbox": [98, 0, 71, 226],
    "area": 16046,
    "segmentation": {...}
    // ← NO category_id field!
  }]
}
```

### 2. COCO File Has Categories, But Isn't Used

The `_annotations.coco.json` file HAS categories:

```json
{
  "categories": [
    {"id": 1, "name": "object"}  // or "crack", etc.
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,  // ← This exists!
      "bbox": [...]
    }
  ]
}
```

**BUT:** The training code reads individual `.json` files, NOT the COCO file!

### 3. Training Code Hardcoded

Even if category info was available, the code ignores it and uses "object".

---

## Impact on Training

### Scenario 1: Single Class Dataset (like yours - only "crack")

**Your COCO file:**
```json
"categories": [{"id": 1, "name": "object"}]
```

**Impact:**
- ✅ Not a big problem (only one class anyway)
- ❌ But you should train with "crack" text, not "object"
- ❌ At inference, using "crack" won't work because model trained with "object"

### Scenario 2: Multi-Class Dataset

**Example COCO file:**
```json
"categories": [
  {"id": 1, "name": "crack"},
  {"id": 2, "name": "spalling"},
  {"id": 3, "name": "corrosion"}
]
```

**Impact with current code:**
- ❌ ALL classes trained as "object"
- ❌ Model can't distinguish between crack/spalling/corrosion
- ❌ Text prompts "crack", "spalling", "corrosion" won't work
- ❌ Model loses semantic understanding

---

## The `apply_to_text_encoder` Question

### With `apply_to_text_encoder: false` (current):

```
Training:
  Input text: "object" → Frozen Text Encoder → Fixed embeddings
  Vision encoder learns to align with these fixed "object" embeddings

Inference with "crack":
  Input text: "crack" → Frozen Text Encoder → DIFFERENT fixed embeddings
  Vision encoder doesn't recognize these embeddings → No detection
```

### With `apply_to_text_encoder: true`:

```
Training:
  Input text: "object" → Trainable Text Encoder → Learned embeddings
  Both text and vision encoders adapt together

Inference with "crack":
  Input text: "crack" → Trained Text Encoder → DIFFERENT embeddings
  Still doesn't work well (trained on "object", not "crack")
```

**Key Point:** Even with trainable text encoder, if you train with "object" and inference with "crack", there's a mismatch!

---

## Solutions

### ✅ Solution 1: Use "object" at Inference (Quick Fix)

Match your inference to training:

```bash
python3 inference_lora.py \
  --prompt "object" \  # ← Same as training
  --image test.jpg
```

**Pros:** Works with current model
**Cons:** Not semantically meaningful

### ✅ Solution 2: Fix Training to Use Actual Categories

Use the new training script I created: `train_sam3_lora_with_categories.py`

**Changes:**
- ✅ Reads categories from COCO file
- ✅ Maps image_id → category_id → category_name
- ✅ Uses actual category name as `query_text`
- ✅ Supports multiple classes properly

**Usage:**
```bash
python3 train_sam3_lora_with_categories.py --config configs/crack_detection_config.yaml
```

Then at inference:
```bash
python3 inference_lora.py \
  --prompt "crack" \  # ← Now this will work!
  --image test.jpg
```

### ✅ Solution 3: Change Your COCO Category to "crack"

If you want to keep using the original training script:

```python
import json

# Load COCO file
with open('data/train/_annotations.coco.json', 'r') as f:
    coco = json.load(f)

# Change category name
for cat in coco['categories']:
    cat['name'] = 'crack'  # Change from "object" to "crack"

# Save
with open('data/train/_annotations.coco.json', 'w') as f:
    json.dump(coco, f)

# Repeat for validation data
with open('data/valid/_annotations.coco.json', 'r') as f:
    coco = json.load(f)
for cat in coco['categories']:
    cat['name'] = 'crack'
with open('data/valid/_annotations.coco.json', 'w') as f:
    json.dump(coco, f)
```

Then manually edit `train_sam3_lora_native.py` line 119:
```python
query_text="crack",  # Changed from "object"
```

### ✅ Solution 4: Train with Multiple Prompts (Best for Robustness)

Modify training to use varied prompts:

```python
# In train_sam3_lora_native.py, line 119
import random
query_text = random.choice([
    "crack",
    "concrete crack",
    "damage",
    "defect",
    "structural crack"
])
```

This makes the model robust to different phrasings.

---

## Recommendation

For your crack detection use case:

1. **Immediate fix:** Use `--prompt "object"` at inference
2. **Proper fix:** Use `train_sam3_lora_with_categories.py` with updated COCO categories
3. **Enable:** `apply_to_text_encoder: true` in config
4. **Retrain:** With actual "crack" category names

---

## Summary Table

| Approach | Training Text | Inference Text | Works? |
|----------|---------------|----------------|--------|
| **Current (broken)** | "object" | "crack" | ❌ No |
| **Quick fix** | "object" | "object" | ⚠️ Yes, but not semantic |
| **Proper fix** | "crack" (from COCO) | "crack" | ✅ Yes! |
| **Best** | ["crack", "damage", ...] | any variant | ✅ Robust! |

---

## Testing

After fixing, verify category support:

```bash
# Train with categories
python3 train_sam3_lora_with_categories.py --config configs/crack_detection_config.yaml

# Test with correct category name
python3 inference_lora.py \
  --config configs/crack_detection_config.yaml \
  --prompt "crack" \
  --image test.jpg \
  --output result.png
```

You should see detections with confidence > 0.3 if training worked.

---

**Created:** 2025-12-05
**Issue:** Training doesn't use class names
**Status:** Fixed training script provided
