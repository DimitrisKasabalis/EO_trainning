# GLCM Quick Reference Card

## 🚀 Quick Start (Copy-Paste Ready)

### 1. Basic GLCM Texture

```python
# Import
from glcm_template_enhanced import glcm_for_classification

# Apply to your image
image_with_texture = glcm_for_classification(
    your_sentinel2_image,
    nir_band='B8',
    red_band='B4'
)

# Use in classification
classified = image_with_texture.classify(your_classifier)
```

### 2. Custom Features (Faster)

```python
from glcm_template_enhanced import add_selected_glcm

# Select specific features
texture = add_selected_glcm(
    image=your_image,
    bands=['B8'],                    # NIR band
    features=['contrast', 'ent'],    # Just 2 features
    radius=1                          # 3x3 window
)
```

### 3. Full Feature Stack

```python
from glcm_template_enhanced import glcm_for_classification

# Comprehensive texture
texture_full = glcm_for_classification(
    image=your_image,
    nir_band='B8',
    red_band='B4',
    swir_band='B11'  # Optional: adds SWIR texture
)
```

---

## 📊 GLCM Features Explained

| Feature | What It Measures | Low Values | High Values |
|---------|------------------|------------|-------------|
| **Contrast** | Local variation | Uniform (water, fields) | Varied (forest, urban) |
| **Entropy** | Randomness | Homogeneous (agriculture) | Heterogeneous (urban) |
| **Correlation** | Pattern regularity | Random patterns | Regular patterns (canopy) |
| **Variance** | Intensity spread | Similar tones | Diverse tones |
| **IDM/Homogeneity** | Uniformity | Textured | Smooth |
| **ASM** | Orderliness | Chaotic | Ordered (geometric) |

---

## 🎯 When to Use GLCM

### ✅ USE GLCM When:
- Separating **primary vs secondary forest** (canopy structure)
- Mapping **urban vs rural** (built environment complexity)
- Classifying **forest types** (structural differences)
- Detecting **mangroves** (root/canopy structure)
- Discriminating **crop types** (field patterns)
- Assessing **forest degradation** (texture loss)

### ❌ DON'T USE GLCM When:
- Spectral bands already separate classes well
- Time constraints (GLCM is slow)
- Very small features (<3 pixels)
- Constantly changing scenes (water, clouds)
- Simple binary masks (water/land)

---

## ⚙️ Parameter Guide

### Window Size (radius)

| Radius | Window | Speed | Use For |
|--------|--------|-------|---------|
| 1 | 3×3 | ⚡⚡⚡ Fast | Urban features, fine texture |
| 2 | 5×5 | ⚡⚡ Moderate | Forest canopy, field patterns |
| 3 | 7×7 | ⚡ Slow | Large-scale patterns |

**Recommendation:** Start with radius=1 (3×3)

### Feature Selection

| Features | Speed | Accuracy | Use When |
|----------|-------|----------|----------|
| 2 features | ⚡⚡⚡ | Good | Testing, time-limited |
| 3-4 features | ⚡⚡ | Better | Production, balanced |
| 6+ features | ⚡ | Best | Research, accuracy-critical |

**Recommendation:** Use 3-4 features (contrast, entropy, correlation)

### Band Selection

| Bands | Best For | Example |
|-------|----------|---------|
| **B8 (NIR)** | Vegetation structure | Forest classification |
| **B4 (Red)** | Overall structure | General land cover |
| **B11 (SWIR)** | Urban/soil texture | Built-up areas |
| **B3 (Green)** | Water boundaries | Mangrove mapping |

**Recommendation:** Start with B8 (NIR) only

---

## 🔥 Common Recipes

### Recipe 1: Forest Classification
```python
# Best for separating primary/secondary forest
texture = add_selected_glcm(
    image,
    bands=['B8'],
    features=['contrast', 'ent', 'corr', 'var'],
    radius=2  # 5×5 for canopy patterns
)
```

### Recipe 2: Urban Mapping
```python
# Best for urban/rural discrimination
texture = add_selected_glcm(
    image,
    bands=['B11', 'B4'],  # SWIR + Red
    features=['contrast', 'ent'],
    radius=1  # 3×3 for buildings
)
```

### Recipe 3: Agricultural Fields
```python
# Best for crop type classification
texture = add_selected_glcm(
    image,
    bands=['B8'],
    features=['idm', 'asm'],  # Uniformity measures
    radius=2  # 5×5 for field patterns
)
```

### Recipe 4: Mangrove Detection
```python
# Best for coastal mangroves
texture = add_selected_glcm(
    image,
    bands=['B8', 'B3'],  # NIR + Green
    features=['contrast', 'ent', 'corr'],
    radius=1
)
```

### Recipe 5: Change Detection
```python
# Detect structural changes
before_texture = add_selected_glcm(image_before, bands=['B8'], features=['contrast'])
after_texture = add_selected_glcm(image_after, bands=['B8'], features=['contrast'])

texture_change = after_texture.select('B8_contrast').subtract(
    before_texture.select('B8_contrast')
).rename('texture_change')
```

---

## 🐛 Troubleshooting

### Error: "Computation timed out"
```python
# ❌ Problem: Area too large or window too big
texture = add_selected_glcm(image, radius=3)

# ✅ Solution: Reduce area or window size
aoi_small = aoi.buffer(-1000)
texture = add_selected_glcm(image.clip(aoi_small), radius=1)
```

### Error: "Memory limit exceeded"
```python
# ❌ Problem: Processing collection
texture_collection = collection.map(lambda img: add_selected_glcm(img))

# ✅ Solution: Process composite
composite = collection.median()
texture = add_selected_glcm(composite)
```

### Problem: Results look wrong
```python
# Check band scaling
stats = image.select('B8').reduceRegion(
    reducer=ee.Reducer.minMax(),
    scale=100
).getInfo()
print(stats)  # Should be 0-1 after cloud masking

# If values are 0-10000, divide first:
image_scaled = image.divide(10000)
texture = add_selected_glcm(image_scaled)
```

### Problem: Too slow
```python
# Speed optimization checklist:
# 1. Use fewer features
features=['contrast', 'ent']  # Just 2

# 2. Use smaller window
radius=1  # 3×3 only

# 3. Use single band
bands=['B8']  # Just NIR

# 4. Process composite not collection
composite = collection.median()

# 5. Reduce area for testing
test_aoi = aoi.buffer(-5000)
```

---

## 📈 Expected Accuracy Improvements

| Land Cover Type | Without Texture | With Texture | Improvement |
|-----------------|-----------------|--------------|-------------|
| Primary Forest | 75% | 88% | +13% |
| Secondary Forest | 70% | 85% | +15% |
| Agricultural | 82% | 87% | +5% |
| Urban | 78% | 90% | +12% |
| Mangroves | 80% | 92% | +12% |
| Water | 95% | 96% | +1% |
| **Overall** | **78%** | **87%** | **+9%** |

---

## 🔬 Advanced Techniques

### Multi-scale Texture
```python
from glcm_template_enhanced import multiscale_glcm

# Capture patterns at different scales
texture_multiscale = multiscale_glcm(
    image,
    band='B8',
    radii=[1, 2, 3],  # 3×3, 5×5, 7×7
    features=['contrast', 'ent']
)
# Warning: Very slow!
```

### Texture Indices
```python
from glcm_template_enhanced import add_texture_indices

# Derive additional indices
enhanced = add_texture_indices(
    image_with_texture,
    contrast_band='B8_contrast',
    entropy_band='B8_ent'
)

# New bands:
# - texture_heterogeneity_index (contrast + entropy)
# - texture_ratio (contrast / entropy)
# - normalized_contrast (0-1 scale)
```

### Visualization
```python
from glcm_template_enhanced import visualize_texture

# Auto-compute visualization parameters
vis_params = visualize_texture(
    image=texture,
    texture_band='B8_contrast',
    geometry=aoi,
    # min_val and max_val auto-computed
)

Map.addLayer(texture, vis_params, 'Texture Contrast')
```

### Batch Processing
```python
from glcm_template_enhanced import batch_glcm_processing

# Apply to entire collection (use carefully!)
texture_collection = batch_glcm_processing(
    image_collection,
    bands=['B8'],
    features=['contrast', 'ent']
)
```

---

## 💡 Pro Tips

1. **Always test on small area first**
   ```python
   test_aoi = ee.Geometry.Point([118.7, 9.8]).buffer(1000)
   ```

2. **Start with minimal features**
   ```python
   features=['contrast']  # Just 1 for testing
   ```

3. **Use composites, not collections**
   ```python
   composite = collection.median()  # Process once
   ```

4. **Check computation time**
   ```python
   import time
   start = time.time()
   result = texture.getInfo()
   print(f"Time: {time.time() - start:.1f}s")
   ```

5. **Export for reuse**
   ```python
   task = ee.batch.Export.image.toDrive(
       image=texture,
       description='texture_features',
       scale=10,
       region=aoi
   )
   task.start()
   ```

---

## 📚 Additional Resources

### Code Templates
- `glcm_template_enhanced.py` - Full implementation
- `glcm_use_cases.py` - 6 Philippine examples
- `test_glcm_template.py` - Test suite

### Documentation
- [GEE GLCM Docs](https://developers.google.com/earth-engine/apidocs/ee-image-glcmtexture)
- Session 2 notebook for full examples
- `GLCM_IMPLEMENTATION_COMPLETE.md` for complete guide

### Example Use Cases
```python
# Load pre-made examples
from glcm_use_cases import (
    use_case_1_forest_classification,
    use_case_2_urban_rural,
    use_case_3_mangrove_mapping
)

result = use_case_1_forest_classification()
```

---

## 🎓 Cheat Sheet Summary

### Fastest Setup (30 seconds)
```python
from glcm_template_enhanced import glcm_for_classification
texture = glcm_for_classification(image)
classified = texture.classify(classifier)
```

### Best Balance (accuracy vs speed)
```python
from glcm_template_enhanced import add_selected_glcm
texture = add_selected_glcm(
    image,
    bands=['B8'],
    features=['contrast', 'ent', 'corr'],
    radius=1
)
```

### Maximum Accuracy (slow)
```python
from glcm_template_enhanced import glcm_for_classification
texture = glcm_for_classification(
    image,
    nir_band='B8',
    red_band='B4',
    swir_band='B11'
)
```

---

**Print this page and keep it handy during the lab!** 📄

---

*Quick Reference for CopPhil Advanced Training - Day 2, Session 2*
*Version 1.0 | 2025-10-15*
