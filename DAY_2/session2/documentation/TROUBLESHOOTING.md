# Session 2 Troubleshooting Guide
**Advanced Palawan Land Cover Classification**

---

## Common Issues and Solutions

This guide covers the most frequent issues encountered in Session 2 and their solutions.

---

## Table of Contents

1. [Google Earth Engine Issues](#gee-issues)
2. [Computation & Performance](#computation-performance)
3. [Classification Problems](#classification-problems)
4. [Data & Feature Issues](#data-feature-issues)
5. [Visualization Problems](#visualization-problems)
6. [Export Issues](#export-issues)

---

## <a name="gee-issues"></a>1. Google Earth Engine Issues

### ❌ Error: "Earth Engine not initialized"

**Symptom:**
```
EEException: Earth Engine not initialized
```

**Solution:**
```python
# Re-initialize Earth Engine
import ee
ee.Authenticate()  # Follow prompts to login
ee.Initialize()
```

If problems persist:
```python
# Clear credentials and re-authenticate
!earthengine authenticate --force
import ee
ee.Initialize()
```

---

### ❌ Error: "User memory limit exceeded"

**Symptom:**
```
Computation timed out. (Error code: 3)
```
or
```
User memory limit exceeded
```

**Causes:**
- Study area too large
- Too many features
- GLCM with large kernel
- High-resolution exports

**Solutions:**

**Option 1: Reduce study area**
```python
# Use smaller AOI for testing
small_aoi = ee.Geometry.Rectangle([118.5, 9.5, 119.0, 10.0])  # Half size
```

**Option 2: Process in tiles**
```python
# Split into tiles
def create_tiles(bbox, n_tiles=4):
    # Split bounding box into n_tiles x n_tiles grid
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / n_tiles
    lat_step = (max_lat - min_lat) / n_tiles
    
    tiles = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            tile = ee.Geometry.Rectangle([
                min_lon + i * lon_step,
                min_lat + j * lat_step,
                min_lon + (i+1) * lon_step,
                min_lat + (j+1) * lat_step
            ])
            tiles.append(tile)
    
    return tiles

# Process each tile separately
tiles = create_tiles([118.5, 9.5, 119.5, 10.5], n_tiles=2)
for i, tile in enumerate(tiles):
    result = feature_stack.clip(tile).classify(classifier)
    # Export each tile...
```

**Option 3: Reduce GLCM kernel size**
```python
# Use smaller texture window
glcm = nir_band.glcmTexture(size=3)  # Instead of size=5 or 7
```

**Option 4: Reduce feature count**
```python
# Use only most important features
essential_features = feature_stack.select([
    'B4', 'B8', 'B11',  # Key spectral bands
    'NDVI', 'NDWI',     # Key indices
    'texture_contrast',  # Key texture
    'NDVI_diff'         # Key temporal
])
```

---

### ❌ Error: "Computation timed out"

**Symptom:**
```
Computation timed out. (Error code: 3)
```

**Solutions:**

1. **Use .aside() sparingly**
   ```python
   # BAD: Too many .aside() calls in loops
   for i in range(100):
       value = image.reduceRegion(...).aside(print)
   
   # GOOD: Only check final result
   result = image.reduceRegion(...)
   print(result.getInfo())  # Only when needed
   ```

2. **Limit temporal range**
   ```python
   # If using multi-year composites
   collection = ee.ImageCollection('COPERNICUS/S2_SR') \
       .filterDate('2024-01-01', '2024-03-31')  # 3 months instead of 12
   ```

3. **Increase scale parameter**
   ```python
   # Use larger pixels for faster computation
   stats = image.reduceRegion(
       reducer=ee.Reducer.mean(),
       geometry=aoi,
       scale=30,  # Instead of 10
       maxPixels=1e10
   )
   ```

---

## <a name="computation-performance"></a>2. Computation & Performance

### ⏱️ Issue: GLCM computation is very slow

**Expected:** 2-5 minutes for GLCM on moderate area  
**Problem:** >10 minutes or timing out

**Solutions:**

1. **Use smaller kernel**
   ```python
   # Fast: 3x3 window
   glcm = nir_band.glcmTexture(size=3)
   
   # Slow: 7x7 window
   # glcm = nir_band.glcmTexture(size=7)  # Avoid
   ```

2. **Compute on composite, not collection**
   ```python
   # GOOD: Compute texture once on median composite
   composite = collection.median()
   glcm = composite.select('B8').glcmTexture(size=3)
   
   # BAD: Compute texture on each image in collection
   # collection_glcm = collection.map(lambda img: img.select('B8').glcmTexture(size=3))
   ```

3. **Select only needed texture features**
   ```python
   # Use specific features instead of all
   texture = glcm.select(['B8_contrast', 'B8_ent', 'B8_corr'])
   # Instead of using all 18 GLCM bands
   ```

---

### ⏱️ Issue: Classification takes too long

**Solutions:**

1. **Reduce number of trees during testing**
   ```python
   # For testing: 50 trees
   test_classifier = ee.Classifier.smileRandomForest(numberOfTrees=50)
   
   # For production: 200 trees
   final_classifier = ee.Classifier.smileRandomForest(numberOfTrees=200)
   ```

2. **Sample training data more efficiently**
   ```python
   # Limit samples per class if you have many
   training_sampled = feature_stack.sampleRegions(
       collection=training_polygons,
       properties=['class_id'],
       scale=10,
       tileScale=4,  # Helps with large areas
       geometries=False  # Don't keep geometries
   )
   ```

---

## <a name="classification-problems"></a>3. Classification Problems

### 📉 Issue: Low classification accuracy (<80%)

**Diagnosis checklist:**

1. **Check training data quality**
   ```python
   # How many samples per class?
   class_counts = training.aggregate_histogram('class_id').getInfo()
   print(class_counts)
   
   # Should have at least 100-200 samples per class
   ```

2. **Visualize training locations**
   ```python
   # Are training polygons in representative locations?
   Map = geemap.Map()
   Map.addLayer(training_polygons, {}, 'Training Data')
   Map.addLayer(dry_season, {'bands': ['B4', 'B3', 'B2'], 'max': 0.3}, 'Sentinel-2')
   Map
   ```

3. **Check feature separability**
   ```python
   # Sample features for each class
   class_samples = {}
   for class_id in range(1, 9):
       class_data = training.filter(ee.Filter.eq('class_id', class_id))
       class_samples[class_id] = class_data
   
   # Check if features differ between classes
   ```

**Solutions:**

1. **Add more/better training samples**
   - Focus on confused classes
   - Ensure geographic distribution
   - Include class variability

2. **Add more features**
   - Try texture features if not using
   - Add temporal features
   - Include topographic data

3. **Check for mislabeled samples**
   - Visually inspect training polygons
   - Remove ambiguous samples
   - Ensure class definitions are clear

---

### 🔀 Issue: Specific classes are confused

**Common confusions:**

#### Primary ↔ Secondary Forest

**Solution:** Add texture features
```python
# Primary forest has more uniform canopy (lower texture variance)
texture_var = glcm.select('B8_var')
feature_stack = feature_stack.addBands(texture_var)
```

#### Mangroves ↔ Agriculture (wet season)

**Solution:** Use temporal NDVI difference
```python
# Mangroves are evergreen (low NDVI_diff)
# Agriculture is seasonal (high NDVI_diff)
ndvi_diff = ndvi_wet.subtract(ndvi_dry).rename('NDVI_diff')
```

#### Urban ↔ Bare Soil

**Solution:** Add NDBI and texture
```python
# Urban has higher heterogeneity (texture entropy)
# Bare soil is more uniform
texture_ent = glcm.select('B8_ent')
ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
```

---

### 🎯 Issue: Classification looks "noisy" (salt-and-pepper)

**Symptom:** Many small isolated pixels of different classes

**Solutions:**

1. **Apply majority filter**
   ```python
   # 3x3 focal mode filter
   classified_smooth = classified.focal_mode(radius=1, kernelType='square')
   ```

2. **Increase minimum mapping unit**
   ```python
   # Remove patches smaller than X pixels
   def remove_small_patches(classified, min_pixels=5):
       # Connected component labeling
       connected = classified.connectedPixelCount(min_pixels)
       classified_filtered = classified.updateMask(connected.gte(min_pixels))
       return classified_filtered
   
   cleaned = remove_small_patches(classified, min_pixels=10)
   ```

3. **Use better training data**
   - Avoid very small or linear training polygons
   - Use compact, representative polygons

---

## <a name="data-feature-issues"></a>4. Data & Feature Issues

### 📁 Error: "Cannot load training data"

**Symptom:**
```
FileNotFoundError: palawan_training_polygons.geojson not found
```

**Solution:**
```python
# Check file path
import os
data_path = '../../../DAY_2/session1/data/palawan_training_polygons.geojson'

if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    print(f"Current directory: {os.getcwd()}")
    # Adjust path as needed

# Alternative: Use absolute path
data_path = '/Users/your_username/Projects/.../palawan_training_polygons.geojson'
training_polygons = geemap.geojson_to_ee(data_path)
```

---

### ❌ Error: "No images found in collection"

**Symptom:**
```
Collection is empty after filtering
```

**Solutions:**

1. **Check date range and cloud cover**
   ```python
   # See how many images available
   collection = ee.ImageCollection('COPERNICUS/S2_SR') \
       .filterBounds(aoi) \
       .filterDate('2024-01-01', '2024-05-31')
   
   count = collection.size().getInfo()
   print(f"Images found: {count}")
   
   # If zero, relax cloud filter
   collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))  # Instead of 20
   ```

2. **Verify AOI is correct**
   ```python
   # Check AOI coordinates
   print(aoi.bounds().getInfo())
   
   # Visualize
   Map = geemap.Map()
   Map.addLayer(aoi, {'color': 'red'}, 'AOI')
   Map.centerObject(aoi)
   Map
   ```

---

### ⚠️ Issue: Features contain null/NaN values

**Symptom:**
```
Error: Sample contains null values
```

**Solution:**
```python
# Check for null values
def check_nulls(image):
    null_mask = image.reduce(ee.Reducer.allNonZero())
    null_count = null_mask.Not().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100
    )
    return null_count.getInfo()

nulls = check_nulls(feature_stack)
print(f"Null pixels: {nulls}")

# Remove null values before sampling
feature_stack_clean = feature_stack.unmask(-9999).updateMask(
    feature_stack.reduce(ee.Reducer.allNonZero())
)
```

---

## <a name="visualization-problems"></a>5. Visualization Problems

### 🗺️ Issue: geemap map not displaying

**Solutions:**

1. **Check Jupyter setup**
   ```python
   # In Jupyter Notebook
   %matplotlib inline
   import geemap
   
   # If using JupyterLab
   # !jupyter labextension install @jupyter-widgets/jupyterlab-manager
   ```

2. **Use alternative rendering**
   ```python
   # If interactive map fails, use static map
   from geemap import geemap
   Map = geemap.Map()
   Map.addLayer(image, vis_params, 'Layer')
   Map.to_html('map.html')  # Save as HTML and open in browser
   ```

---

### 🎨 Issue: Classification colors don't match legend

**Solution:**
```python
# Ensure class IDs match palette indices
# Class 1 → palette[0], Class 2 → palette[1], etc.

class_colors = [
    '#0A5F0A',  # Class 1: Primary Forest
    '#4CAF50',  # Class 2: Secondary Forest
    '#009688',  # Class 3: Mangroves
    '#FFC107',  # Class 4: Agricultural
    '#FFEB3B',  # Class 5: Grassland
    '#2196F3',  # Class 6: Water
    '#F44336',  # Class 7: Urban
    '#795548'   # Class 8: Bare Soil
]

# Ensure min/max match class range
vis = {'min': 1, 'max': 8, 'palette': class_colors}
Map.addLayer(classified, vis, 'Classification')
```

---

## <a name="export-issues"></a>6. Export Issues

### 💾 Error: "Export failed" or "Export too large"

**Solutions:**

1. **Export in tiles**
   ```python
   # Split large exports
   tiles = create_tiles(aoi.bounds().getInfo()['coordinates'][0], n_tiles=2)
   
   for i, tile in enumerate(tiles):
       task = ee.batch.Export.image.toDrive(
           image=classified.clip(tile),
           description=f'palawan_tile_{i}',
           scale=10,
           region=tile,
           maxPixels=1e13
       )
       task.start()
   ```

2. **Reduce resolution**
   ```python
   # Export at lower resolution
   task = ee.batch.Export.image.toDrive(
       image=classified,
       description='palawan_classification',
       scale=30,  # Instead of 10
       region=aoi,
       maxPixels=1e13
   )
   ```

3. **Export to Asset first**
   ```python
   # For very large exports
   task = ee.batch.Export.image.toAsset(
       image=classified,
       description='palawan_to_asset',
       assetId='users/your_username/palawan_2024',
       scale=10,
       region=aoi,
       maxPixels=1e13
   )
   task.start()
   
   # Then download from Asset
   ```

---

## Quick Reference: Error Messages

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| "Earth Engine not initialized" | Authentication issue | `ee.Authenticate(); ee.Initialize()` |
| "Computation timed out" | Area too large / complex | Reduce AOI or simplify features |
| "User memory limit exceeded" | Too much data in memory | Process in tiles or reduce resolution |
| "Collection is empty" | No images match filters | Relax date/cloud filters |
| "FileNotFoundError" | Wrong file path | Check path with `os.path.exists()` |
| "Sample contains null values" | Missing data in features | Use `.unmask()` or filter nulls |
| "Export failed" | Export too large | Export in tiles or reduce resolution |

---

## Getting Additional Help

1. **Check GEE Status**
   - [https://status.earthengine.google.com/](https://status.earthengine.google.com/)

2. **GEE Community Forum**
   - [https://groups.google.com/g/google-earth-engine-developers](https://groups.google.com/g/google-earth-engine-developers)

3. **Stack Overflow**
   - Tag: `google-earth-engine`

4. **Session Resources**
   - Review Session 2 documentation
   - Check code templates
   - Ask instructor during lab hours

---

## Prevention Tips

✅ **Do:**
- Start with small test areas
- Use `seed` parameter for reproducibility
- Save intermediate results
- Document parameter choices
- Test code incrementally

❌ **Don't:**
- Process entire provinces without testing
- Use `.getInfo()` in loops
- Skip error handling
- Ignore warning messages
- Change multiple things at once when debugging

---

**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 2
