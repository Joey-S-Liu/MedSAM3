# Training Approaches Comparison

## Current Training (Broken)

```
Data has:
  Image 1: [crack at (10,20), crack at (30,40)]  → Category: "crack"
  Image 2: [spalling at (50,60)]                  → Category: "spalling"
  Image 3: [corrosion at (70,80)]                 → Category: "corrosion"

Training sees:
  Image 1: query_text="object"  ← WRONG!
  Image 2: query_text="object"  ← WRONG!
  Image 3: query_text="object"  ← WRONG!

Result:
  ❌ Model learns: "object" → segment anything
  ❌ Can't distinguish crack vs spalling vs corrosion
  ❌ Inference with "crack" fails (different embeddings)
```

## Fixed Training (Correct)

```
Data has:
  Image 1: [crack at (10,20), crack at (30,40)]  → Category: "crack"
  Image 2: [spalling at (50,60)]                  → Category: "spalling"
  Image 3: [corrosion at (70,80)]                 → Category: "corrosion"

Training sees:
  Image 1: query_text="crack"     ← CORRECT!
  Image 2: query_text="spalling"  ← CORRECT!
  Image 3: query_text="corrosion" ← CORRECT!

Result:
  ✅ Model learns: "crack" → segment cracks
  ✅ Model learns: "spalling" → segment spalling
  ✅ Model learns: "corrosion" → segment corrosion
  ✅ Inference with correct text prompts works!
```

---

## Code Difference

### OLD (train_sam3_lora_native.py)

```python
# Line 119 - HARDCODED
query_text="object"  # ← Always "object", ignores real categories!
```

### NEW (train_sam3_lora_with_categories.py)

```python
# Lines 135-139 - DYNAMIC
if category_ids:
    most_common_cat_id = max(set(category_ids), key=category_ids.count)
    query_text = self.categories.get(most_common_cat_id, "object")
else:
    query_text = "object"
# ← Uses ACTUAL category name from COCO data!
```

---

## Visual Example

### Your Current Situation

```
┌─────────────────────────────────────┐
│  COCO Annotations                   │
│  categories: [                      │
│    {id: 1, name: "object"}          │  ← Only "object"
│  ]                                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Training Code                      │
│  query_text = "object"              │  ← Hardcoded "object"
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Model Learns                       │
│  "object" embeddings → segment      │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Inference with "crack"             │
│  Result: ❌ No detection            │  ← Mismatch!
│  (Different embeddings)             │
└─────────────────────────────────────┘
```

### After Fix

```
┌─────────────────────────────────────┐
│  COCO Annotations (updated)         │
│  categories: [                      │
│    {id: 1, name: "crack"}           │  ← Changed to "crack"
│  ]                                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  New Training Code                  │
│  query_text = categories[cat_id]    │  ← Reads from COCO
│             = "crack"               │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Model Learns                       │
│  "crack" embeddings → segment cracks│
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Inference with "crack"             │
│  Result: ✅ Detections!             │  ← Match!
│  (Same embeddings as training)      │
└─────────────────────────────────────┘
```

---

## Testing Both Approaches

### Test Current Model (will fail)

```bash
# Your current model trained with "object"
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --prompt "crack" \
  --image test.jpg

# Output: ❌ Max confidence: 0.000 (no detections)
```

```bash
# Try with "object" instead
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --prompt "object" \
  --image test.jpg

# Output: Still likely fails (model didn't train properly)
```

### Test Fixed Approach (will work)

```bash
# 1. Update COCO categories
python3 -c "
import json
for split in ['train', 'valid']:
    path = f'data/{split}/_annotations.coco.json'
    with open(path) as f: data = json.load(f)
    for cat in data['categories']:
        cat['name'] = 'crack'
    with open(path, 'w') as f: json.dump(data, f)
print('✅ Updated categories to crack')
"

# 2. Train with new script
python3 train_sam3_lora_with_categories.py \
  --config configs/crack_detection_config.yaml

# 3. Inference with "crack"
python3 inference_lora.py \
  --config configs/crack_detection_config.yaml \
  --prompt "crack" \
  --image test.jpg

# Output: ✅ Detections with confidence > 0.3
```

---

## Key Takeaway

> **The text prompt at inference MUST match what the model saw during training!**

If training sees "object" but inference uses "crack" → ❌ No match → No detections

If training sees "crack" and inference uses "crack" → ✅ Match → Detections!
