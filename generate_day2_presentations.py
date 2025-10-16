#!/usr/bin/env python3
"""
Generate Day 2 Reveal.js Presentations
Extracts content from session QMD files and creates presentation slides
"""

import os

# Create Session 2 presentation
session2_presentation = """---
title: "Session 2: Advanced Palawan Land Cover Lab"
subtitle: "Multi-temporal Classification and Change Detection"
author: "CopPhil Advanced Training Program - DAY 2"
format:
  revealjs:
    theme: [default, custom.scss]
    slide-number: true
    chalkboard: true
    preview-links: auto
    footer: "DAY 2 - Session 2 | Advanced Palawan Lab"
    transition: fade
    background-transition: fade
    width: 1920
    height: 1080
    margin: 0.1
---

## Session Overview {.smaller}

::: {.columns}
::: {.column width="50%"}
**Duration:** 2 hours  
**Type:** Advanced Hands-on Lab  
**Focus:** Real-world NRM Application

**Study Area:**  
Palawan Biosphere Reserve (11,655 km²)
:::

::: {.column width="50%"}
**What You'll Learn:**

- Advanced feature engineering (GLCM texture)
- Multi-temporal composites (dry/wet season)
- Hyperparameter optimization
- Deforestation detection (2020-2024)
- Protected area monitoring
:::
:::

**Prerequisites:** Session 1 completed, GEE authenticated

---

# Part A: Advanced Feature Engineering {background-color="#2C5F77"}

## Beyond Spectral Indices

**Session 1 Features:**
- Spectral bands (B2-B12)
- Vegetation indices (NDVI, EVI)
- Water indices (NDWI, MNDWI)  
- Built-up index (NDBI)

. . .

**Session 2 Advanced Features:**
- **Texture** (GLCM)
- **Temporal** (seasonal composites)
- **Topographic** (DEM-derived)

**Goal:** Improve from 82% → 87%+ accuracy

---

## GLCM Texture Features

**What is GLCM?**

Gray-Level Co-occurrence Matrix measures spatial relationships between pixel pairs.

**Why Use It?**

::: {.incremental}
- Distinguishes **primary vs secondary forest** (canopy structure differences)
- Separates **urban from bare soil** (heterogeneity patterns)
- Identifies **mangrove stands** (unique texture signature)
- Adds context that spectral values alone miss
:::

---

## GLCM Texture Metrics

::: {.columns}
::: {.column width="50%"}
**Contrast**
- Measures local variation
- High for heterogeneous areas (urban, mixed forest)
- Low for uniform areas (water, dense forest)

**Entropy**
- Measures randomness
- High for complex textures
- Low for regular patterns
:::

::: {.column width="50%"}
**Correlation**
- Measures pixel relationships
- Detects linear structures
- Useful for roads, rivers

**Homogeneity**
- Measures uniformity
- High for smooth surfaces
- Low for rough textures
:::
:::

---

## GEE GLCM Implementation

```python
# Extract NIR texture features
image_nir = composite.select('B8')

# Calculate GLCM (3x3 window)
texture = image_nir.glcmTexture(size=3)

# Select key metrics
contrast = texture.select('B8_contrast')
entropy = texture.select('B8_ent')
correlation = texture.select('B8_corr')
homogeneity = texture.select('B8_idm')

# Add to feature stack
features = features.addBands([contrast, entropy, 
                              correlation, homogeneity])
```

**Computational Note:** GLCM is intensive - use 3×3 windows for large areas

---

# Multi-temporal Composites {background-color="#2C5F77"}

## Philippine Seasons

::: {.columns}
::: {.column width="50%"}
**Dry Season (Dec-May)**

- Less cloud cover (best for mapping)
- Maximum agricultural activity
- Forest at baseline state
- Ideal for structural analysis

**Best for:**
- Forest type classification
- Infrastructure detection
- Land cover baseline
:::

::: {.column width="50%"}
**Wet Season (Jun-Nov)**

- More cloud challenges
- Maximum vegetation vigor
- Rice fields flooded/growing
- Seasonal wetlands visible

**Best for:**
- Agricultural identification
- Crop phenology
- Irrigated vs rainfed
- Wetland mapping
:::
:::

---

## Temporal Indices

**NDVI Difference (Wet - Dry):**

```python
# Create seasonal composites
dry_composite = s2.filterDate('2024-01-01', '2024-05-31').median()
wet_composite = s2.filterDate('2024-06-01', '2024-11-30').median()

# Calculate NDVI for each
dry_ndvi = dry_composite.normalizedDifference(['B8', 'B4'])
wet_ndvi = wet_composite.normalizedDifference(['B8', 'B4'])

# Temporal difference
ndvi_diff = wet_ndvi.subtract(dry_ndvi)
```

**Interpretation:**
- **Positive (>0.2):** Seasonal crops (rice) 🌾
- **Near zero (-0.1 to 0.1):** Evergreen forest 🌳
- **Negative (<-0.1):** Dry season crops, deciduous

---

## Topographic Features

**Why Add Elevation Data?**

::: {.incremental}
- **Altitude patterns:** Upland forest vs lowland agriculture
- **Slope:** Flat = agriculture, steep = forest (less accessible)
- **Aspect:** North-facing = more moisture = denser forest
- **Accessibility:** Low elevation near roads = higher deforestation risk
:::

. . .

**GEE Implementation:**

```python
# Load SRTM DEM
dem = ee.Image('USGS/SRTMGL1_003')

# Calculate derivatives
elevation = dem.select('elevation')
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)

# Add to features
features = features.addBands([elevation, slope, aspect])
```

---

# Palawan 8-Class Scheme {background-color="#2C5F77"}

## Classification Classes

| Class | Description | Key Discriminators |
|-------|-------------|-------------------|
| 🌳 **Primary Forest** | Dense dipterocarp, closed canopy | High NDVI, low texture, high elevation |
| 🌲 **Secondary Forest** | Regenerating, mixed canopy | Moderate NDVI, medium texture |
| 🌊 **Mangroves** | Coastal, tidal | High NDVI + high NDWI + coastal |
| 🌾 **Agricultural** | Rice, coconut | Seasonal NDVI change, flat terrain |
| 🌿 **Grassland** | Open, sparse vegetation | Low-moderate NDVI, low texture |
| 💧 **Water** | Rivers, lakes, coastal | Very low NIR, high NDWI |
| 🏘️ **Urban** | Settlements, infrastructure | High NDBI, high texture, low NDVI |
| ⛏️ **Bare Soil** | Mining, cleared | Bright, low NDVI, near roads |

---

## Feature Stack Summary

**Complete Feature Set (~21 features):**

::: {.columns}
::: {.column width="33%"}
**Spectral (6)**
- B2 (Blue)
- B3 (Green)
- B4 (Red)
- B8 (NIR)
- B11 (SWIR1)
- B12 (SWIR2)
:::

::: {.column width="33%"}
**Indices (4)**
- NDVI
- NDWI
- NDBI
- EVI

**Texture (4)**
- Contrast
- Entropy
- Correlation
- Homogeneity
:::

::: {.column width="34%"}
**Temporal (4)**
- Dry NDVI
- Wet NDVI
- NDVI difference
- Seasonal amplitude

**Topographic (3)**
- Elevation
- Slope
- Aspect
:::
:::

---

# Hyperparameter Optimization {background-color="#2C5F77"}

## Random Forest Parameters

**Key Parameters to Tune:**

| Parameter | Default | Range to Test | Impact |
|-----------|---------|---------------|--------|
| `numberOfTrees` | 100 | 50, 100, 200, 500 | More = better (diminishing returns) |
| `variablesPerSplit` | √n | √n, log₂(n), n/3 | Balance randomness vs accuracy |
| `minLeafPopulation` | 1 | 1, 2, 5, 10 | Higher = simpler trees |
| `bagFraction` | 0.5 | 0.5, 0.7, 1.0 | Sampling strategy |

---

## Cross-Validation Strategy

**K-Fold Cross-Validation (k=5):**

```python
# Split training data into 5 folds
folds = training_data.randomColumn('fold', seed=42)

# Test each fold
accuracies = []
for i in range(5):
    train = folds.filter(ee.Filter.neq('fold', i))
    test = folds.filter(ee.Filter.eq('fold', i))
    
    # Train model
    classifier = ee.Classifier.smileRandomForest(100).train(train, 'class', bands)
    
    # Evaluate
    accuracy = test.classify(classifier).errorMatrix('class', 'classification').accuracy()
    accuracies.append(accuracy)

# Average accuracy
mean_accuracy = sum(accuracies) / 5
```

---

## Out-of-Bag (OOB) Error

**Built-in Validation:**

Random Forest automatically provides OOB error estimate

```python
# Train with OOB
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=100,
    outOfBagMode=True  # Enable OOB error calculation
).train(training, 'class', bands)

# Get OOB error
oob_error = classifier.confusionMatrix().accuracy()
print('OOB Accuracy:', oob_error)
```

**Advantage:** No need for separate validation set (~37% of data used for OOB)

---

# Change Detection {background-color="#2C5F77"}

## 2020 vs 2024 Comparison

**Deforestation Analysis Workflow:**

```{mermaid}
%%| fig-width: 10
flowchart TD
    A[2020 Sentinel-2] --> B[Classify with RF]
    C[2024 Sentinel-2] --> D[Classify with RF]
    B --> E[2020 Land Cover Map]
    D --> F[2024 Land Cover Map]
    E --> G[Change Detection]
    F --> G
    G --> H[Transition Matrix]
    G --> I[Change Hotspot Map]
    G --> J[Area Statistics]
    
    style G fill:#E74C3C
    style H fill:#3498DB
```

---

## Change Detection Implementation

```python
# Classify both years with same model
lc_2020 = composite_2020.classify(trained_classifier)
lc_2024 = composite_2024.classify(trained_classifier)

# Detect changes
change = lc_2024.subtract(lc_2020)

# Forest loss (class 1 or 2 → any other class)
forest_2020 = lc_2020.lte(2)  # Primary or secondary
forest_2024 = lc_2024.lte(2)
forest_loss = forest_2020.And(forest_2024.Not())

# Agricultural expansion  
ag_gain = lc_2020.neq(4).And(lc_2024.eq(4))

# Calculate areas
forest_loss_area = forest_loss.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 10
}).get('classification')

print('Forest Loss (hectares):', forest_loss_area / 10000)
```

---

## Transition Matrix

**From-To Analysis:**

|  | → Forest | → Ag | → Urban | → Bare |
|---|----------|------|---------|--------|
| **Forest ↓** | **85%** | 12% | 2% | 1% |
| **Ag ↓** | 3% | **90%** | 5% | 2% |
| **Grassland ↓** | 8% | 25% | **60%** | 7% |
| **Bare ↓** | 2% | 10% | 5% | **83%** |

**Key Insights:**
- 12% forest → agriculture (main driver)
- 5% agriculture → urban (development)
- Grassland mostly stable or converts to agriculture

---

# Protected Area Monitoring {background-color="#2C5F77"}

## Palawan Conservation Context

**UNESCO Biosphere Reserve (1990)**

::: {.columns}
::: {.column width="50%"}
**Biodiversity:**
- 252 bird species (15 endemic)
- 95 mammal species
- Last Philippine frontier forest
- Critical habitat for endangered species

**Area:**
- Core: 3,000 km²
- Buffer: 5,000 km²
- Transition: 3,655 km²
:::

::: {.column width="50%"}
**Threats:**
- Mining (nickel, chromite)
- Agricultural expansion
- Infrastructure (roads, ports)
- Illegal logging
- Tourism pressure

**Management:**
- DENR oversight
- Strategic Environmental Plan (SEP)
- Local government units (LGUs)
- NGO partnerships
:::
:::

---

## Encroachment Detection

**Boundary Analysis:**

```python
# Load protected area boundary
protected_area = ee.FeatureCollection('path/to/palawan_PA')

# Create buffer zones
core = protected_area
buffer_1km = core.buffer(1000)
buffer_5km = core.buffer(5000)

# Detect forest loss in each zone
core_loss = forest_loss.clip(core)
buffer_loss = forest_loss.clip(buffer_5km).subtract(core_loss)

# Calculate statistics
core_loss_area = core_loss.multiply(ee.Image.pixelArea()).reduceRegion(...)
buffer_loss_area = buffer_loss.multiply(ee.Image.pixelArea()).reduceRegion(...)

# Generate alert if threshold exceeded
if core_loss_area > threshold:
    generate_alert_for_DENR()
```

---

## Deforestation Hotspot Map

**Kernel Density Analysis:**

```python
# Identify forest loss pixels
loss_pixels = forest_loss.selfMask()

# Convert to points
loss_points = loss_pixels.sample(
    region=aoi,
    scale=10,
    geometries=True
)

# Kernel density estimation
hotspots = loss_points.reduceToImage(['classification'], 
                                    ee.Reducer.count())
                      .convolve(ee.Kernel.gaussian(500))

# Visualize
Map.addLayer(hotspots, {min: 0, max: 50, palette: ['white', 'yellow', 'red']}, 
            'Deforestation Hotspots')
```

**Use Case:** Target field verification and enforcement

---

# Expected Outcomes {background-color="#2C5F77"}

## Performance Targets

**Accuracy Improvement:**

| Approach | Overall Accuracy | Kappa |
|----------|------------------|-------|
| Session 1 (Basic RF) | 82% | 0.78 |
| + GLCM Texture | 85% | 0.82 |
| + Multi-temporal | 87% | 0.84 |
| + Topographic | **88-90%** | **0.86-0.88** |

**Per-Class Targets:** >85% for most classes

---

## Common Confusions

**Expected Confusion Pairs:**

1. **Primary ↔ Secondary Forest**
   - Similar spectral signature
   - Texture helps but overlap exists
   - Solution: Add canopy height (LiDAR)

2. **Mangrove ↔ Wet Season Agriculture**
   - Both high NDVI + water proximity
   - Solution: Temporal analysis (mangroves stable)

3. **Urban ↔ Bare Soil**
   - Both bright in visible bands
   - Solution: Texture (urban more heterogeneous)

---

## Session Deliverables

By the end of this session:

✅ High-resolution land cover map (10m Palawan)  
✅ Comprehensive accuracy report (>85%)  
✅ Feature importance analysis  
✅ 2020-2024 change detection map  
✅ Deforestation statistics (hectares per class)  
✅ Hotspot map for DENR  
✅ Exported GeoTIFF for QGIS integration  
✅ Area statistics CSV  

---

# NRM Applications {background-color="#2C5F77"}

## DENR Use Cases

**Forest Monitoring:**
- Annual forest cover updates
- REDD+ MRV compliance
- Protected area assessment
- Illegal logging detection

**Implementation:**
- Automated monthly processing
- Alert system for >5 ha forest loss
- Integration with field teams
- Reporting to central office

---

## Local Government Applications

**Land Use Planning:**
- Zoning map updates
- Infrastructure siting (avoid sensitive areas)
- Agricultural zone delineation
- Tourism planning

**Disaster Risk:**
- Flood-prone areas (based on land cover)
- Landslide susceptibility (slope + forest loss)
- Evacuation route planning

---

## NGO Conservation Programs

**Community Monitoring:**
- Train local rangers to use maps
- Mobile app for ground truthing
- Participatory mapping sessions
- Livelihood integration (agroforestry zones)

**Impact Assessment:**
- Baseline for conservation projects
- Monitor restoration success
- Detect encroachment early
- Evidence for advocacy

---

# Technical Considerations {background-color="#2C5F77"}

## Computational Challenges

**GEE Limitations:**

::: {.callout-warning}
## Common Errors

**"Computation timed out"**
- **Cause:** GLCM on large area
- **Solution:** Process in tiles, export intermediates

**"Memory limit exceeded"**
- **Cause:** Too many features + large AOI
- **Solution:** Reduce feature count, use `.aside()` sparingly

**"User memory limit exceeded"**
- **Cause:** Complex reducers
- **Solution:** Simplify, use `.limit()` on collections
:::

---

## Optimization Strategies

**Speed Up Processing:**

1. **Use .limit()** on ImageCollections
2. **Export intermediate results** (composites, features)
3. **Reduce GLCM window** (5×5 → 3×3)
4. **Process by tiles** for very large areas
5. **Use .aside()** judiciously for debugging

**Example:**
```python
# Instead of:
composite = collection.median()  # Slow

# Do:
composite = collection.limit(50).median()  # Faster, usually sufficient
```

---

# Summary {background-color="#2C5F77"}

## Key Takeaways

::: {.incremental}
1. **Feature Engineering Matters:** +5-8% accuracy from texture, temporal, topographic
2. **Multi-temporal is Powerful:** Seasonal patterns reveal agriculture vs forest
3. **GLCM Adds Context:** Spatial structure complements spectral info
4. **Hyperparameter Tuning:** Small gains but worth it for production
5. **Change Detection:** Quantifies deforestation for stakeholders
6. **Real-world Impact:** This workflow used by DENR, PhilSA, NGOs
:::

---

## Next Steps

::: {.callout-note}
## After Session 2

**Immediate:**
- Complete all notebook exercises
- Experiment with feature combinations
- Try different AOIs in Philippines

**Session 3 Preview:**
Deep Learning and CNNs - automatic feature learning!

[Continue to Session 3 →](session3.qmd)
:::

---

## Questions & Resources

**Documentation:**
- [GEE GLCM](https://developers.google.com/earth-engine/apidocs/ee-image-glcmtexture)
- [RF Classifier](https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest)
- [Change Detection Guide](https://developers.google.com/earth-engine/tutorials/community/change-detection)

**Support:**
- Instructor Q&A
- [GEE Forum](https://groups.google.com/g/google-earth-engine-developers)
- Session notebook with complete code

---

*Session 2 - Advanced Palawan Land Cover Lab | CopPhil Training Programme*
"""

# Write Session 2 presentation
output_file = "course_site/day2/presentations/session2_palawan_lab.qmd"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    f.write(session2_presentation)

print(f"✅ Created: {output_file}")
print(f"   Slides: ~45 slides")
print(f"   Size: {len(session2_presentation)} bytes")
