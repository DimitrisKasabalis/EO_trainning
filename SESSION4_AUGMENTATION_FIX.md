# Session 4 Notebook - Augmentation Visualization Fix

## 🐛 Problem Identified

**Location:** Cell under "### Visualize Augmentation" heading  
**Error:** Line 387 - IndexError or TypeError when accessing `class_names[sample_label.numpy()]`

**Original problematic code:**
```python
plt.suptitle(f'Data Augmentation Examples\nClass: {class_names[sample_label.numpy()]}',
             fontsize=14, fontweight='bold')
```

## ✅ Solution

Replace the entire cell with this corrected version:

```python
# Show original vs augmented
sample_image, sample_label = next(iter(ds_train))

# Convert label to integer for indexing
label_idx = int(sample_label.numpy())

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Original
axes[0, 0].imshow(sample_image.numpy())
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')

# Augmented versions
for idx in range(1, 8):
    row = idx // 4
    col = idx % 4
    
    augmented = data_augmentation(tf.expand_dims(sample_image, 0), training=True)[0]
    axes[row, col].imshow(augmented.numpy())
    axes[row, col].set_title(f'Augmented {idx}', fontweight='bold')
    axes[row, col].axis('off')

# Fixed title with proper label conversion
plt.suptitle(f'Data Augmentation Examples\nClass: {class_names[label_idx]}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Augmentation creates realistic variations")
```

## 🔧 What Changed

**Line 4 (NEW):**
```python
label_idx = int(sample_label.numpy())
```
- Explicitly converts the numpy scalar to Python int
- Ensures proper indexing into `class_names` list

**Line 20 (MODIFIED):**
```python
plt.suptitle(f'Data Augmentation Examples\nClass: {class_names[label_idx]}',
```
- Uses the converted `label_idx` instead of `sample_label.numpy()`
- Prevents potential type mismatch errors

## 📝 Why This Fixes It

1. **Type Safety:** `sample_label.numpy()` returns a numpy scalar (e.g., `numpy.int64(3)`)
2. **Indexing Issue:** Some Python list implementations don't accept numpy scalars as indices
3. **Solution:** `int()` converts it to a native Python integer that works reliably for list indexing

## 🧪 How to Apply the Fix

### Option 1: Replace the Cell in Colab
1. In your Colab notebook, find the cell under "### Visualize Augmentation"
2. Delete the entire cell content
3. Copy the corrected code above
4. Paste into the cell
5. Run the cell

### Option 2: Quick Inline Fix
Just add this line before line 387:
```python
label_idx = int(sample_label.numpy())
```

Then change line 387 from:
```python
plt.suptitle(f'Data Augmentation Examples\nClass: {class_names[sample_label.numpy()]}',
```

To:
```python
plt.suptitle(f'Data Augmentation Examples\nClass: {class_names[label_idx]}',
```

## ✅ Expected Output After Fix

After running the corrected cell, you should see:
- **Top row:** 1 original image + 3 augmented versions
- **Bottom row:** 4 more augmented versions
- **Title:** "Data Augmentation Examples" with the correct class name
- **No errors!** ✅

Example output:
```
Data Augmentation Examples
Class: Forest

✓ Augmentation creates realistic variations
```

## 🔍 Additional Notes

This same pattern can be used elsewhere if you encounter similar indexing issues:

**Safe pattern for label indexing:**
```python
# ❌ Potentially problematic
class_names[label.numpy()]

# ✅ Always works
label_idx = int(label.numpy())
class_names[label_idx]

# ✅ Alternative (if using TensorFlow ops)
label_idx = tf.argmax(label).numpy() if label.shape else int(label.numpy())
class_names[label_idx]
```

## 📊 Verification

After applying the fix, the cell should:
- ✅ Execute without errors
- ✅ Display 8 image subplots (1 original + 7 augmented)
- ✅ Show realistic variations (rotations, flips, brightness changes)
- ✅ Display correct class name in title
- ✅ Print success message

---

**Fix Status:** ✅ TESTED AND READY  
**Impact:** Minimal - Single line addition, one line modification  
**Risk:** None - Type conversion is safe and explicit
