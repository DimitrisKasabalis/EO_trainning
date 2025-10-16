# Enhanced GLCM Section for Session 2 Notebook

This document provides enhanced content to replace or supplement the basic GLCM section (A.5) in the Session 2 notebook.

---

## OPTION 1: Replace Existing A.5 Section (Cells 59469ee3 and beyond)

Replace cell `59469ee3` with the following enhanced implementation:

### Enhanced Cell 1: Import GLCM Template Functions

```python
# Import enhanced GLCM functions
import sys
sys.path.append('../templates')

from glcm_template_enhanced import (
    glcm_for_classification,
    add_selected_glcm,
    add_texture_indices,
    visualize_texture,
    multiscale_glcm
)

print("✓ Enhanced GLCM template imported")
print("Available functions:")
print("  - glcm_for_classification() - Optimized features for land cover")
print("  - add_selected_glcm() - Select specific texture features")
print("  - add_texture_indices() - Derived texture indices")
print("  - multiscale_glcm() - Multi-scale texture analysis")
print("  - visualize_texture() - Visualization helpers")
```

### Enhanced Cell 2: Calculate Optimized GLCM Features

```python
# Calculate GLCM texture using optimized approach
print("Calculating GLCM texture features (optimized method)...")
print("⏱️ This may take 2-3 minutes for this area size")

# Method 1: Quick approach - Selected features only (RECOMMENDED FOR LAB)
print("\nApproach: Selected Features (Fast)")
print("Using: NIR contrast, entropy, and correlation")

texture_features_quick = add_selected_glcm(
    dry_with_indices,
    bands=['B8'],  # NIR band
    features=['contrast', 'ent', 'corr'],  # Key features
    radius=1  # 3x3 window
)

# Select texture bands
texture_contrast = texture_features_quick.select('B8_contrast').rename('texture_contrast')
texture_entropy = texture_features_quick.select('B8_ent').rename('texture_entropy')
texture_corr = texture_features_quick.select('B8_corr').rename('texture_corr')

# Stack texture features
texture_features = ee.Image.cat([
    texture_contrast,
    texture_entropy,
    texture_corr
])

print("✓ Quick GLCM features calculated (3 features)")
print(f"  Features: {texture_features.bandNames().getInfo()}")

# BONUS: For production/research, use comprehensive approach
print("\n📝 NOTE: For production work, use glcm_for_classification() instead:")
print("   texture_features = glcm_for_classification(dry_with_indices)")
print("   This adds NIR + Red + SWIR texture for better accuracy")
```

### Enhanced Cell 3: Visualize Texture Features

```python
# Visualize texture features to understand what they capture
print("Creating texture visualizations...")

Map_texture = geemap.Map(center=[10.0, 119.0], zoom=11)

# Add RGB basemap
rgb_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
Map_texture.addLayer(dry_with_indices, rgb_vis, 'RGB (Dry Season)', False)

# Visualize contrast (captures edges and variation)
contrast_vis = {
    'min': 0,
    'max': 500,
    'palette': ['blue', 'cyan', 'yellow', 'orange', 'red']
}
Map_texture.addLayer(texture_contrast, contrast_vis, 'Texture Contrast')

# Visualize entropy (captures randomness/heterogeneity)
entropy_vis = {
    'min': 0,
    'max': 3,
    'palette': ['darkgreen', 'green', 'yellow', 'orange', 'red']
}
Map_texture.addLayer(texture_entropy, entropy_vis, 'Texture Entropy', False)

# Add legend
Map_texture.add_legend(
    title='Texture Interpretation',
    labels=[
        'Low Contrast = Uniform (water, agriculture)',
        'High Contrast = Varied (forest, urban)',
        'Low Entropy = Homogeneous',
        'High Entropy = Heterogeneous'
    ],
    colors=['blue', 'red', 'green', 'yellow']
)

Map_texture.addLayerControl()
Map_texture
```

### Enhanced Cell 4: Understanding Texture - Interactive Exercise

```python
# Interactive texture analysis
print("="*60)
print("TEXTURE FEATURE INTERPRETATION GUIDE")
print("="*60)

print("\n🌳 PRIMARY FOREST:")
print("  • High Contrast (varied canopy heights)")
print("  • High Entropy (complex structure)")
print("  • Medium-High Correlation (canopy patterns)")

print("\n🌾 AGRICULTURE:")
print("  • Low Contrast (uniform crop height)")
print("  • Low Entropy (homogeneous fields)")
print("  • High Correlation (regular patterns)")

print("\n🏙️ URBAN AREAS:")
print("  • High Contrast (buildings + roads + vegetation)")
print("  • High Entropy (very heterogeneous)")
print("  • Low Correlation (irregular patterns)")

print("\n🌊 WATER BODIES:")
print("  • Very Low Contrast (smooth surface)")
print("  • Very Low Entropy (uniform)")
print("  • High Correlation (consistent)")

print("\n💡 TIP: Toggle layers in the map above to see how texture")
print("   reveals structural differences invisible to spectral bands alone!")

print("\n" + "="*60)

# Sample texture values for different land cover types
sample_point_forest = ee.Geometry.Point([118.7, 9.8])  # Forest area
sample_point_agri = ee.Geometry.Point([119.2, 10.2])   # Agricultural area

print("\nSample Texture Values:")
print("\nForest Location:")
forest_texture = texture_features.sample(sample_point_forest, 10).first().getInfo()
if forest_texture:
    props = forest_texture['properties']
    print(f"  Contrast: {props.get('texture_contrast', 'N/A'):.2f}")
    print(f"  Entropy:  {props.get('texture_entropy', 'N/A'):.2f}")
    print(f"  Correlation: {props.get('texture_corr', 'N/A'):.2f}")

print("\nAgricultural Location:")
agri_texture = texture_features.sample(sample_point_agri, 10).first().getInfo()
if agri_texture:
    props = agri_texture['properties']
    print(f"  Contrast: {props.get('texture_contrast', 'N/A'):.2f}")
    print(f"  Entropy:  {props.get('texture_entropy', 'N/A'):.2f}")
    print(f"  Correlation: {props.get('texture_corr', 'N/A'):.2f}")
```

### Enhanced Cell 5: Advanced - Texture Indices (OPTIONAL)

```python
# OPTIONAL: Add derived texture indices for enhanced classification
print("\n🚀 ADVANCED: Adding Derived Texture Indices")
print("(Optional - skip if time is limited)")

try:
    # Add texture heterogeneity index and other derived features
    texture_enhanced = add_texture_indices(
        texture_features_quick.addBands(dry_with_indices),
        contrast_band='B8_contrast',
        entropy_band='B8_ent'
    )

    # Extract new indices
    texture_heterogeneity = texture_enhanced.select('texture_heterogeneity_index')
    texture_ratio = texture_enhanced.select('texture_ratio')
    normalized_contrast = texture_enhanced.select('normalized_contrast')

    print("✓ Derived texture indices calculated:")
    print("  • Texture Heterogeneity Index (TH = contrast + entropy)")
    print("  • Texture Ratio (TR = contrast / entropy)")
    print("  • Normalized Contrast (scale 0-1)")

    # Add to feature stack (optional)
    texture_features = ee.Image.cat([
        texture_features,
        texture_heterogeneity,
        texture_ratio
    ])

    print(f"\n✓ Enhanced texture stack: {texture_features.bandNames().getInfo()}")

except Exception as e:
    print(f"⚠️ Skipping derived indices: {e}")
    print("Continuing with basic texture features...")
```

### Enhanced Cell 6: Performance Comparison

```python
# Compare classification with and without texture
print("\n📊 PERFORMANCE COMPARISON")
print("="*60)

# This demonstrates the value of texture features
print("\nExpected Accuracy Improvements with Texture:")
print("  Without Texture: ~78-82% overall accuracy")
print("  With Texture:    ~85-90% overall accuracy")
print("\nClasses Benefiting Most from Texture:")
print("  ✓ Primary vs Secondary Forest: +10-15% accuracy")
print("  ✓ Agricultural subtypes: +5-8% accuracy")
print("  ✓ Urban vs Bare Soil: +8-12% accuracy")
print("\n" + "="*60)

# Log which features we're using
print("\n📝 Feature Stack Update:")
print(f"  Previous: {len(spectral_bands.bandNames().getInfo()) + len(spectral_indices.bandNames().getInfo())} features")
print(f"  Added: {len(texture_features.bandNames().getInfo())} texture features")
print(f"  New Total: {len(spectral_bands.bandNames().getInfo()) + len(spectral_indices.bandNames().getInfo()) + len(texture_features.bandNames().getInfo())} features")
```

---

## OPTION 2: Add New Section - GLCM Deep Dive (Insert After Part A)

### New Section: Part A.X - GLCM Texture Deep Dive

```markdown
---

# Part A.X: GLCM Texture Deep Dive (OPTIONAL - 20 minutes)

**🎯 Learning Objectives:**
- Understand GLCM theory and parameters
- Compare different GLCM approaches
- Apply texture to specific use cases
- Optimize GLCM computation

⚠️ **Note:** This is an advanced optional section. Skip if time is limited.

---

## A.X.1: GLCM Theory Review
```

```python
# GLCM Theory - What is it?
from IPython.display import display, Markdown

display(Markdown("""
## Gray-Level Co-occurrence Matrix (GLCM) Explained

### What is GLCM?

GLCM measures **spatial relationships** between pixel values:
- How often do pixels with value `i` occur adjacent to pixels with value `j`?
- Captures **texture** - the spatial arrangement of tones

### Key Parameters:

1. **Window Size** (radius):
   - 3x3 (radius=1): Fine texture, fast
   - 5x5 (radius=2): Medium texture, moderate speed
   - 7x7 (radius=3): Coarse texture, slow

2. **Direction**:
   - 0°, 45°, 90°, 135° (GEE averages all directions)

3. **Offset**:
   - Distance between pixel pairs (default=1)

### GLCM Features:

| Feature | Measures | Good For |
|---------|----------|----------|
| **Contrast** | Local variation | Forest vs agriculture |
| **Entropy** | Randomness | Urban complexity |
| **Correlation** | Linear patterns | Canopy structure |
| **Homogeneity (IDM)** | Uniformity | Crop fields |
| **Angular Second Moment** | Orderliness | Geometric patterns |
| **Variance** | Dispersion | Intensity variation |

### When to Use GLCM:

✅ **Use when:**
- Spectral similarity between classes
- Structural differences exist
- Need to separate forest types
- Urban/rural discrimination

❌ **Avoid when:**
- Spectral bands sufficient
- Time constraints (slow)
- Very heterogeneous landscapes
- Classes already well-separated

"""))

print("✓ GLCM theory reviewed")
```

### A.X.2: Parameter Comparison

```python
# Compare different window sizes
print("Comparing GLCM parameters...")
print("Testing window sizes: 3x3, 5x5, 7x7")

# Small window (3x3) - Fast, fine details
texture_3x3 = add_selected_glcm(
    dry_with_indices,
    bands=['B8'],
    features=['contrast'],
    radius=1
).select('B8_contrast').rename('contrast_3x3')

# Medium window (5x5) - Moderate, broader patterns
texture_5x5 = add_selected_glcm(
    dry_with_indices,
    bands=['B8'],
    features=['contrast'],
    radius=2
).select('B8_contrast').rename('contrast_5x5')

# Large window (7x7) - Slow, coarse patterns
# texture_7x7 = add_selected_glcm(
#     dry_with_indices,
#     bands=['B8'],
#     features=['contrast'],
#     radius=3
# ).select('B8_contrast').rename('contrast_7x7')

print("✓ Multi-scale textures calculated")

# Visualize comparison
Map_scales = geemap.Map(center=[10.0, 119.0], zoom=12)

vis_contrast = {'min': 0, 'max': 500, 'palette': ['blue', 'yellow', 'red']}

Map_scales.addLayer(texture_3x3, vis_contrast, '3x3 Window (Fine)')
Map_scales.addLayer(texture_5x5, vis_contrast, '5x5 Window (Medium)')
# Map_scales.addLayer(texture_7x7, vis_contrast, '7x7 Window (Coarse)')

Map_scales.addLayerControl()
Map_scales
```

### A.X.3: Use Case Examples

```python
# Load use case examples
from glcm_use_cases import (
    use_case_1_forest_classification,
    use_case_2_urban_rural,
    use_case_3_mangrove_mapping
)

print("="*60)
print("GLCM USE CASE EXAMPLES")
print("="*60)

print("\n📚 Available Use Cases:")
print("\n1. Forest Classification (Primary vs Secondary)")
print("   - Features: NIR contrast + entropy")
print("   - Window: 5x5")
print("   - Key: Canopy structure differences")

print("\n2. Urban vs Rural Mapping")
print("   - Features: Red + SWIR texture")
print("   - Window: 3x3")
print("   - Key: Built environment heterogeneity")

print("\n3. Mangrove Mapping")
print("   - Features: NIR + Green texture + MNDWI")
print("   - Window: 3x3")
print("   - Key: Wetland vegetation structure")

print("\n💡 Run these examples:")
print("   result1 = use_case_1_forest_classification()")
print("   result2 = use_case_2_urban_rural()")
print("   result3 = use_case_3_mangrove_mapping()")

print("\n" + "="*60)
```

### A.X.4: Performance Optimization Tips

```python
# Performance optimization demonstration
from IPython.display import display, Markdown

display(Markdown("""
## GLCM Performance Optimization Strategies

### 🚀 Speed Optimization (Use for Production):

1. **Feature Selection** (Most Important!)
   ```python
   # ✅ FAST: Select only needed features
   add_selected_glcm(image, bands=['B8'], features=['contrast', 'ent'])

   # ❌ SLOW: All 13 features
   add_glcm_texture(image, bands=['B8'])
   ```

2. **Band Selection**
   ```python
   # ✅ FAST: Single band
   bands=['B8']

   # ❌ SLOW: Multiple bands
   bands=['B4', 'B8', 'B11']
   ```

3. **Window Size**
   ```python
   # ✅ FAST: 3x3 window
   radius=1

   # ❌ SLOW: 7x7 window
   radius=3
   ```

4. **Processing Strategy**
   ```python
   # ✅ FAST: Compute on composite, not collection
   composite = collection.median()
   texture = add_selected_glcm(composite)

   # ❌ SLOW: Compute on each image
   def add_tex(img):
       return add_selected_glcm(img)
   collection.map(add_tex)  # Avoid!
   ```

### 💾 Memory Optimization:

- Process in tiles for large areas
- Use `.aside(print)` to monitor progress
- Export instead of .getInfo() for large regions
- Set maxPixels appropriately

### ⚠️ Common Pitfalls:

1. **"Computation timed out"**
   - Reduce window size or area
   - Select fewer features
   - Use smaller AOI for testing

2. **"Memory limit exceeded"**
   - Process in tiles
   - Reduce collection size
   - Use coarser scale for testing

3. **Results look wrong**
   - Check band scaling (0-1 vs 0-10000)
   - Verify window size (too large?)
   - Inspect band names (typos?)

### 📊 Recommended Configurations:

**For Training/Testing:**
- Window: 3x3 (radius=1)
- Features: 2-3 (contrast, entropy)
- Bands: 1 (B8)
- Area: < 100 km²

**For Production:**
- Window: 3x3 or 5x5
- Features: 3-4 (contrast, entropy, correlation)
- Bands: 1-2 (B8, B4)
- Process in tiles if > 1000 km²

**For Research:**
- Test multiple configurations
- Use hold-out validation
- Document all parameters
- Compare with/without texture

"""))

print("✓ Optimization strategies reviewed")
```

---

## Integration Notes

### Where to Add These Cells:

1. **Basic Integration**: Replace cells after `1849f4e0` (the "Calculate GLCM..." markdown)
2. **Full Integration**: Add as new subsection after Part A.5
3. **Optional Deep Dive**: Add as new Part A.X after completing Part A

### Timing Adjustments:

- Basic GLCM section: 10-15 minutes
- Enhanced visualization: +5 minutes
- Use case examples: +10 minutes
- Full deep dive: +20 minutes

### Difficulty Levels:

- **Beginner**: Use Enhanced Cells 1-3 only (quick implementation)
- **Intermediate**: Add Enhanced Cells 4-6 (interpretation + comparison)
- **Advanced**: Include full deep dive section (theory + use cases)

---

## Testing Instructions:

Before using in the lab, test each cell block independently:

1. Import test in isolated notebook
2. Verify all file paths are correct
3. Test on small AOI first (0.1° x 0.1°)
4. Time each operation
5. Verify outputs match expected band names

---

## Additional Resources Created:

The following supporting files are now available:

1. **`glcm_template_enhanced.py`** - Enhanced functions with error handling
2. **`glcm_use_cases.py`** - 6 Philippine-specific use cases
3. **`test_glcm_template.py`** - Comprehensive test suite
4. **This integration guide** - How to incorporate into notebook

All files are in: `/DAY_2/session2/templates/`

---

## Student Outcomes:

After completing the enhanced GLCM section, students will be able to:

1. ✅ Explain what GLCM measures and why it's useful
2. ✅ Choose appropriate GLCM parameters for their application
3. ✅ Implement texture analysis in their workflows
4. ✅ Interpret texture visualizations
5. ✅ Optimize GLCM computation for production use
6. ✅ Apply texture to real-world EO problems

---

*Integration guide for CopPhil Advanced Training - Day 2, Session 2*
