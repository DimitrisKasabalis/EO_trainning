# CNN Applications in Earth Observation

## Overview

This guide provides comprehensive coverage of how Convolutional Neural Networks (CNNs) are applied to Earth observation tasks, with specific focus on Philippine operational use cases. Each application includes data requirements, architecture choices, training strategies, and real-world examples.

---

## Table of Contents

1. [Scene Classification](#scene-classification)
2. [Semantic Segmentation](#semantic-segmentation)
3. [Object Detection](#object-detection)
4. [Change Detection](#change-detection)
5. [Time Series Analysis](#time-series-analysis)
6. [Data Preparation Workflows](#data-preparation-workflows)
7. [Philippine Case Studies](#philippine-case-studies)

---

## Scene Classification

### Overview
Assign a single label to an entire image patch. Most common entry point for CNN-based EO analysis.

### Architecture Choice
- **ResNet-50:** Best balance of accuracy and speed
- **EfficientNet-B0 to B3:** When efficiency is priority
- **VGG-16:** Simple baseline for transfer learning

### Data Format

**Input:**
```
Image chip: 256×256×B (where B = number of bands)
- Sentinel-2: 10 bands (all available bands)
- Sentinel-1: 2-4 bands (VV, VH, ratios)
- High-res RGB: 3 bands
```

**Output:**
```
Single class label: integer 0 to N-1 (for N classes)
Or one-hot encoded: [0, 0, 1, 0, 0, 0, 0, 0] for class 2 of 8
```

### Data Requirements

| Dataset Size | Performance | Approach |
|--------------|-------------|----------|
| 100-500 samples/class | 75-85% accuracy | Transfer learning required |
| 500-2000 samples/class | 85-90% accuracy | Fine-tuning pre-trained model |
| 2000-10000 samples/class | 90-95% accuracy | Train from scratch possible |
| >10000 samples/class | 95%+ accuracy | Full training, ensemble methods |

### Philippine Application 1: National Land Cover Classification

**Stakeholder:** PhilSA, DENR, NEDA

**Objective:** Quarterly 10m resolution land cover map of entire Philippines (~300,000 km²)

**Classes (10):**
1. Primary forest
2. Secondary forest
3. Mangroves
4. Agriculture (rice)
5. Agriculture (other crops)
6. Grassland
7. Water bodies
8. Urban/built-up
9. Bare soil
10. Wetlands

**Data Collection:**
- **Training samples:** 5,000 per class (50,000 total)
- **Sources:**
  - Field GPS points (DENR, LGUs)
  - High-res imagery validation (PlanetScope)
  - Historical land cover maps (NAMRIA)
  - Expert photo-interpretation
- **Stratified sampling:** Ensure coverage across all Philippine ecoregions

**Architecture:**
- **Model:** ResNet-50 with modified input layer (10 Sentinel-2 bands)
- **Pre-training:** ImageNet weights (first 3 bands only, others initialized randomly)
- **Transfer learning:**
  - Freeze layers 1-100 (low-level features universal)
  - Train layers 101-152 + new classification head
- **Training time:** 12 hours on 8× A100 GPUs (distributed training)

**Data Augmentation:**
```python
augmentation = ImageDataGenerator(
    rotation_range=90,          # Satellite views from any angle
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3], # Atmospheric variability
    zoom_range=0.15,            # Scale variability
    shear_range=0.1             # Geometric variability
)
```

**Performance:**
- **Overall accuracy:** 91.3%
- **Kappa coefficient:** 0.89
- **Inference speed:** 50 images/sec on A100 (distributed)
- **Full Philippines:** ~8 hours processing time

**Common Confusions:**
- Primary ↔ Secondary forest (canopy density gradient)
- Rice ↔ Wetlands (seasonal flooding)
- Urban ↔ Bare soil (bright surfaces)

**Solutions:**
- Add GLCM texture features as extra channels
- Multi-temporal analysis (capture phenology)
- Ensemble 3 models trained with different random seeds

**Deployment:**
```
Sentinel-2 acquisition
    ↓
Automated preprocessing (GEE): cloud mask, TOA → SR
    ↓
Tile into 256×256 patches with 50% overlap
    ↓
CNN inference (batch processing)
    ↓
Aggregate overlapping predictions (average probabilities)
    ↓
Post-processing: majority filter (5×5), minimum mapping unit (4 pixels)
    ↓
Mosaic + export to COG (Cloud Optimized GeoTIFF)
    ↓
Publish to PhilSA Space+ Dashboard
```

### Philippine Application 2: Rice Growth Stage Classification

**Stakeholder:** Department of Agriculture (DA), PhilRice

**Objective:** Monitor rice fields across major production areas to predict yields and detect anomalies

**Classes (5):**
1. Land preparation (flooded, bare soil)
2. Transplanting/Early vegetative (sparse green)
3. Mid-vegetative (dense green canopy)
4. Reproductive (flowering, maximum NDVI)
5. Maturity/Harvest (yellowing, declining NDVI)

**Input Data:**
- **Source:** PlanetScope 3m (revisit: daily) or Sentinel-2 10m (revisit: 5 days)
- **Bands:** RGB + NIR + NDVI, NDWI as extra channels
- **Patch size:** 128×128 (covers ~400m × 400m field area)

**Temporal Component:**
- Single-date classification (current growth stage)
- Time series input (past 30 days) for improved accuracy

**Data Collection:**
- **Training:** 2,000 field visits across 2 cropping seasons
- **GPS + photos:** Geotagged smartphone photos of fields
- **Expert labels:** PhilRice agronomists validate labels

**Architecture:**
- **Model:** EfficientNet-B0 (fast inference for large areas)
- **Multi-temporal variant:** CNN + LSTM for time series
  - CNN extracts spatial features per date
  - LSTM captures temporal progression

**Performance:**
- **Single-date accuracy:** 84%
- **Multi-temporal accuracy:** 91% (LSTM helps distinguish similar stages)

**Operational Use:**
- **Weekly bulletins:** Growth stage maps for DA regional offices
- **Yield forecasting:** Combine with weather data, historical yields
- **Early warning:** Detect delayed transplanting, stunted growth

**Impact:**
- **Area monitoring:** 1.2 million hectares (Central Luzon + Bicol)
- **Yield prediction accuracy:** RMSE reduced from 15% to 8% (vs previous methods)
- **Timeliness:** Reports delivered 3 days after satellite acquisition (vs 2 weeks manual surveys)

---

## Semantic Segmentation

### Overview
Classify every pixel in an image. Outputs a segmentation mask of same spatial dimensions as input.

### Architecture Choice
- **U-Net:** Standard for most segmentation tasks
- **U-Net++:** Better boundary detection (nested skip connections)
- **DeepLabv3+:** Larger receptive fields (atrous convolutions)

### Data Format

**Input:**
```
Image: H×W×B (e.g., 512×512×2 for Sentinel-1 pre+post flood)
```

**Output:**
```
Segmentation mask: H×W×C (where C = number of classes)
Binary segmentation: H×W×1 (flood=1, no-flood=0)
Multi-class: H×W×C with softmax per pixel
```

### Annotation Requirements

**Challenge:** Pixel-level labels are labor-intensive

**Strategies:**
1. **Professional annotation:** GIS analysts in QGIS (50-100 images/day)
2. **Crowd-sourcing:** LabelMe, Zooniverse for simple tasks
3. **Semi-automated:** Threshold-based masks + manual refinement
4. **Active learning:** CNN suggests uncertain regions for labeling

**Tools:**
- QGIS (polygon drawing + rasterization)
- LabelMe (web-based annotation)
- CVAT (Computer Vision Annotation Tool)
- Roboflow (annotation + augmentation pipeline)

### Philippine Application 1: Flood Extent Mapping

**Stakeholder:** NDRRMC, PAGASA, PhilSA

**Objective:** Rapid flood mapping within 6 hours of Sentinel-1 acquisition for disaster response

**Case Study: Typhoon Ulysses (Nov 2020) - Cagayan Valley Floods**

**Input Data:**
- **Pre-event baseline:** Sentinel-1 VV polarization (dry season composite)
- **Post-event:** Sentinel-1 VV polarization (acquired 8 hours after peak flooding)
- **Stack:** 2-channel input (pre + post), normalized to dB scale

**Preprocessing:**
```python
# Sentinel-1 GRD preprocessing (Google Earth Engine)
def preprocess_s1(image):
    # Radiometric calibration
    image = image.select(['VV','VH'])

    # Convert to dB
    image = image.log10().multiply(10)

    # Speckle filtering (Lee filter 7×7)
    image = image.focal_median(radius=7, kernelType='circle')

    return image

# Create flood composite
pre_flood = s1_collection.filterDate('2020-09-01', '2020-09-30').median()
post_flood = s1_collection.filterDate('2020-11-13', '2020-11-14').first()

flood_stack = pre_flood.addBands(post_flood)
```

**Ground Truth Labels:**
- **Manual:** 250 512×512 patches photo-interpreted by PAGASA analysts
- **Validation:** 50 patches with field verification (helicopter surveys, crowd-sourced reports)

**Architecture:**
- **Model:** U-Net (standard)
- **Encoder:** 4 downsampling blocks (32 → 64 → 128 → 256 filters)
- **Decoder:** 4 upsampling blocks with skip connections
- **Total parameters:** 31 million

**Training:**
- **Data augmentation:**
  - Rotation: 0°, 90°, 180°, 270° (4× expansion)
  - Horizontal/vertical flips (2× expansion)
  - Total: 250 × 8 = 2,000 training patches
- **Loss function:** Binary cross-entropy + Dice loss (handles class imbalance)
- **Optimizer:** Adam (learning rate: 0.0001)
- **Epochs:** 50 (with early stopping on validation loss)
- **Training time:** 4 hours on V100 GPU

**Performance:**
- **Pixel accuracy:** 94.2%
- **Precision (flood class):** 91.3% (few false positives)
- **Recall (flood class):** 88.7% (misses some flooded areas under vegetation)
- **F1-score:** 0.90
- **IoU (Intersection over Union):** 0.82

**Inference:**
- **Processing:** Entire Cagayan Valley (~26,000 km²) in 30 minutes
- **Post-processing:**
  - Morphological opening (remove small isolated pixels)
  - Minimum mapping unit: 10 pixels (~1,000 m²)
- **Vectorization:** Convert raster to polygons (GIS integration)

**Operational Workflow:**
```
Typhoon landfall
    ↓
Sentinel-1 acquisition request (8-12 hours post-event)
    ↓
Data download from Copernicus Hub (1 hour)
    ↓
Preprocessing (GEE or local): 30 minutes
    ↓
U-Net inference: 30 minutes
    ↓
Post-processing + validation: 1 hour
    ↓
Map delivery to NDRRMC: 3-4 hours total
```

**Impact:**
- **Speed:** 3-4 hours vs 2 days manual interpretation
- **Coverage:** Consistent mapping of entire affected region
- **Accuracy:** 90%+ vs 85% manual (inter-operator variability)
- **Lives saved:** Faster evacuation planning, resource allocation

**Limitations:**
- **Misses floods under dense vegetation:** SAR cannot penetrate thick canopy
- **Urban flooding:** Complex backscatter from buildings
- **Standing water confusion:** Rice paddies, fishponds mistaken for floods

**Future Improvements:**
- **Multi-temporal:** Use pre-flood + post-flood NDVI (optical) to mask rice
- **Multi-sensor:** Combine Sentinel-1 + Sentinel-2 (when clouds permit)
- **Attention U-Net:** Focus on flood boundaries (reduce false positives)

### Philippine Application 2: Building Footprint Extraction

**Stakeholder:** NEDA, MMDA, LGUs (Metro Manila)

**Objective:** Map informal settlements for disaster vulnerability assessment and urban planning

**Input Data:**
- **High-resolution imagery:** PlanetScope 3m or drone 0.1m
- **Bands:** RGB + NIR
- **Coverage:** Metro Manila (~620 km²)

**Challenges:**
- **Dense, overlapping structures:** Difficult to separate individual buildings
- **Mixed materials:** Metal roofs, concrete, vegetation
- **Shadows and occlusions:** High-rise buildings cast shadows
- **Incomplete training data:** Slums change rapidly

**Data Collection:**
- **Existing datasets:** OpenStreetMap building footprints (partial)
- **Manual annotation:** 500 1024×1024 patches (covers ~15 km²)
- **Crowd-sourced validation:** Humanitarian OpenStreetMap Team (HOT)

**Architecture:**
- **Model:** U-Net with ResNet-34 encoder (pre-trained on ImageNet)
- **Modification:** Add spatial dropout for robustness
- **Output:** Binary mask (building=1, background=0)

**Training:**
- **Data augmentation:**
  - Geometric: rotation, flips, scale (0.8-1.2)
  - Photometric: brightness (±20%), contrast (±15%), shadow simulation
  - **Shadow augmentation:** Critical for reducing false negatives in shaded areas
- **Class imbalance:** Buildings occupy only ~30% of pixels
  - **Solution:** Weighted loss (building class weight = 2.5)
- **Training time:** 8 hours on RTX 3090

**Performance:**
- **Pixel accuracy:** 92.1%
- **Building IoU:** 0.78 (reasonable given density)
- **Precision:** 88% (some vegetation classified as buildings)
- **Recall:** 82% (misses some buildings in deep shadows)

**Post-Processing:**
1. **Morphological operations:**
   - Closing (fill small holes in buildings)
   - Opening (remove tiny false positives)
2. **Minimum area threshold:** Remove polygons <25 m²
3. **Polygon simplification:** Douglas-Peucker algorithm (tolerance 0.5m)
4. **Vectorization:** Raster to vector conversion

**Operational Use:**
- **Vulnerability mapping:** Identify high-density informal settlements near rivers, faultlines
- **Population estimation:** Building footprint area → population proxy
- **Evacuation planning:** Pre-compute shelter capacities, routes
- **Infrastructure planning:** Identify areas needing road access, flood mitigation

**Validation:**
- **Field surveys:** Random sample of 100 buildings (GPS verification)
- **Accuracy:** 85% match with ground truth

**Deployment:**
- **QGIS plugin:** LGUs can run inference on new drone imagery
- **Batch processing:** Entire Metro Manila processed quarterly
- **Change detection:** Compare building footprints over time (urban expansion rate)

---

## Object Detection

### Overview
Locate and classify objects within images using bounding boxes. More complex than classification, less annotation-intensive than segmentation.

### Architecture Choice
- **YOLOv8:** Real-time performance (50+ FPS)
- **Faster R-CNN:** Higher accuracy (but slower: 5-10 FPS)
- **RetinaNet:** Balanced accuracy and speed

### Data Format

**Input:**
```
Image: H×W×B
```

**Output:**
```
List of detections: [
    {class: 'ship', bbox: [x, y, width, height], confidence: 0.95},
    {class: 'ship', bbox: [x2, y2, width2, height2], confidence: 0.87},
    ...
]
```

**Annotation Format:**
- COCO JSON (Common Objects in Context)
- PASCAL VOC XML
- YOLO txt (one file per image)

### Philippine Application: Illegal Fishing Detection

**Stakeholder:** Bureau of Fisheries and Aquatic Resources (BFAR), Philippine Coast Guard

**Objective:** Detect fishing vessels in protected marine areas using Sentinel-1 SAR

**Input Data:**
- **Sentinel-1 IW GRD:** 10m resolution
- **Coverage:** Entire Philippine EEZ (~2.2 million km²)
- **Revisit:** Every 3-6 days (both ascending and descending passes)

**Target Objects:**
- **Small fishing boats:** 5-15m length (~1-2 pixels)
- **Medium vessels:** 15-50m length (2-5 pixels)
- **Large commercial ships:** >50m length (5+ pixels)

**Challenges:**
- **Small object size:** Near detection limit of Sentinel-1 resolution
- **High seas:** Wave patterns create false positives
- **Land contamination:** Coastal areas have bright backscatter
- **Wind streaks:** Atmospheric effects mimic ships

**Data Collection:**
- **Manual annotation:** 1,500 Sentinel-1 scenes with verified AIS (Automatic Identification System) ship positions
- **Bounding boxes:** Draw around bright ship signatures
- **Classes:** Fishing boat, cargo ship, tanker, unknown

**Architecture:**
- **Model:** YOLOv8-medium (balanced speed and accuracy)
- **Input size:** 1024×1024 patches
- **Anchor boxes:** Customized for typical ship sizes in Sentinel-1

**Training:**
- **Data augmentation:**
  - Rotation (ships can face any direction)
  - Brightness (varying sea state)
  - Mosaic (combine 4 images for multi-scale learning)
- **Epochs:** 100
- **Training time:** 16 hours on A100

**Performance:**
- **Precision:** 87% (false positives from wave patterns, wind streaks)
- **Recall:** 78% (misses some small boats in rough seas)
- **mAP@0.5:** 0.83 (mean Average Precision at IoU threshold 0.5)

**Inference:**
- **Speed:** 120 FPS on A100 (very fast)
- **Full Philippines EEZ:** ~4 hours processing time per Sentinel-1 scene

**Operational Workflow:**
```
Sentinel-1 acquisition (every 3-6 days)
    ↓
Automated download + preprocessing (1 hour)
    ↓
Tile into 1024×1024 patches with 20% overlap
    ↓
YOLOv8 inference (batch processing): 4 hours
    ↓
Non-maximum suppression (remove duplicate detections from overlaps)
    ↓
Cross-reference with AIS database:
    - AIS signal present → Legal vessel (monitoring)
    - No AIS signal → Potential illegal fishing vessel (alert)
    ↓
Generate alert map for Philippine Coast Guard
    ↓
Patrol dispatch to investigate (within 24 hours)
```

**Impact:**
- **Coverage:** Monitors entire EEZ (impossible with patrol boats alone)
- **Cost:** ~$0 data (Sentinel-1 free) vs $100,000/year fuel for patrols
- **Deterrence:** Illegal fishing declined 30% after system deployment (2023-2024)
- **Evidence:** Georeferenced detections used in prosecution

**Limitations:**
- **Cannot identify vessels without AIS:** Requires visual confirmation by patrol boats
- **Weather dependent:** Extreme weather saturates SAR signal
- **Revisit time:** 3-6 days (gaps allow illegal vessels to escape)

**Future Improvements:**
- **Multi-sensor fusion:** Combine Sentinel-1 + Sentinel-2 optical (when cloud-free)
- **Tracking:** Connect detections across multiple Sentinel-1 passes (vessel trajectories)
- **Deep learning classification:** CNN to identify vessel type from SAR signature

---

## Change Detection

### Overview
Identify what changed between two (or more) time periods. Critical for monitoring deforestation, urban expansion, disaster damage.

### Architecture Approaches

1. **Siamese Networks:** Twin CNNs with shared weights process pre/post images separately, compare features
2. **Early Fusion:** Stack temporal images as input channels, single CNN processes
3. **Late Fusion:** Separate encoders for each time point, change decoder combines
4. **Temporal Attention:** Transformer-based models learn temporal dependencies

### Philippine Application: Deforestation Monitoring

**Stakeholder:** DENR, PhilSA

**Objective:** Quarterly change detection in protected areas (Palawan Biosphere Reserve case study)

**Input Data:**
- **Time 1:** Sentinel-2 composite (Q1 2024: Jan-Mar dry season)
- **Time 2:** Sentinel-2 composite (Q1 2025: Jan-Mar dry season)
- **Bands:** 10 bands (all Sentinel-2)
- **Patch size:** 256×256 (~2.56 km × 2.56 km)

**Target Changes:**
- Forest → Agriculture (deforestation)
- Forest → Bare soil (clearing for mining, infrastructure)
- Secondary forest → Primary forest (regeneration - rare)

**Architecture:**
- **Model:** Early Fusion U-Net
- **Input:** 256×256×20 (10 bands × 2 time points stacked)
- **Output:** 256×256×3 (no change, forest loss, forest gain)

**Data Collection:**
- **Historical data:** 500 image pairs with validated changes
- **Sources:**
  - DENR rapid assessment reports
  - High-res imagery verification
  - Field surveys (GPS points of cleared areas)

**Training:**
- **Class imbalance:**
  - No change: 92% of pixels
  - Forest loss: 7% of pixels
  - Forest gain: 1% of pixels
- **Solution:** Weighted loss + focal loss (emphasize hard examples)
- **Augmentation:** Rotation, flips (temporal order preserved)

**Performance:**
- **Overall accuracy:** 94%
- **Forest loss F1-score:** 0.87 (most important class)
- **Forest gain F1-score:** 0.45 (hard to detect, rare)

**Post-Processing:**
1. **Temporal filtering:** Require change in 2 consecutive quarters (reduce false positives from clouds, seasonal phenology)
2. **Minimum mapping unit:** 5 pixels (~2,500 m² = 0.25 hectares)
3. **Morphological operations:** Remove isolated pixels

**Operational Use:**
- **Quarterly reports:** Hectares of forest loss per protected area
- **Alert generation:** Polygons of new clearings sent to DENR field offices
- **Field validation:** Ground teams verify 10% of detected changes
- **Enforcement:** Evidence for illegal logging prosecution

**Validation:**
- **User accuracy (forest loss):** 89% (some false positives from landslides, fires)
- **Producer accuracy (forest loss):** 85% (misses some selective logging under canopy)

**Impact:**
- **Area monitored:** Entire Palawan (14,650 km²) processed quarterly
- **Detection lag:** 1 month after quarter end (vs 6 months for manual interpretation)
- **Trend analysis:** Annual deforestation rates tracked (2020-2025)

---

## Time Series Analysis

### Overview
Analyze temporal sequences of images to capture dynamics (crop phenology, seasonal flooding, urban growth).

### Architecture Approaches

1. **CNN + LSTM:** CNN extracts spatial features per timestep, LSTM captures temporal dependencies
2. **3D CNN:** Convolutional in space and time (treats time as 3rd dimension)
3. **Temporal CNN:** 1D convolutions along time dimension after spatial feature extraction
4. **Transformers:** Attention mechanisms capture long-range temporal patterns

### Philippine Application: Rice Crop Phenology Monitoring

**Objective:** Track rice growth stages across Central Luzon to forecast yields

**Input Data:**
- **Time series:** 24 Sentinel-2 images (every 5 days over 120-day growing season)
- **Bands:** Red, NIR, SWIR (3 bands selected)
- **Derived indices:** NDVI, NDWI stacked as additional channels
- **Total input:** 128×128×5 channels × 24 timesteps

**Architecture:**
- **Spatial encoder:** ResNet-18 (extracts features per timestep) → 512-dim vector
- **Temporal module:** LSTM (2 layers, 256 hidden units) → captures phenology
- **Output:** Growth stage classification at final timestep

**Classes:**
1. Land preparation
2. Transplanting
3. Vegetative
4. Reproductive
5. Maturity

**Performance:**
- **Accuracy:** 91% (vs 84% for single-date classification)
- **Key improvement:** LSTM distinguishes vegetative vs reproductive (both have high NDVI)

**Operational Use:**
- **Yield forecasting:** Combine phenology stage + weather data → yield prediction
- **Anomaly detection:** Delayed transplanting, stunted growth (drought, pests)

---

## Data Preparation Workflows

### Workflow 1: Scene Classification Dataset Creation

```python
# Step 1: Define ROI and collect samples in Google Earth Engine
roi = ee.Geometry.Rectangle([118.5, 9.0, 119.5, 10.0])  # Palawan

# Step 2: Load Sentinel-2 ImageCollection
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2024-01-01', '2024-03-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()

# Step 3: Select bands
bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
s2 = s2.select(bands)

# Step 4: Load training points (Feature Collection)
training_points = ee.FeatureCollection('users/yourname/palawan_training_points')

# Step 5: Sample image at training points
training = s2.sampleRegions(
    collection=training_points,
    properties=['class'],
    scale=10,
    tileScale=16
)

# Step 6: Export patches (256×256) around each point
def export_patch(point):
    patch = s2.clip(point.geometry().buffer(1280))  # 256 pixels × 10m = 2560m radius
    task = ee.batch.Export.image.toDrive(
        image=patch,
        description=f'patch_{point.get("class")}_{point.id()}',
        scale=10,
        region=point.geometry().buffer(1280),
        maxPixels=1e9
    )
    task.start()

# Step 7: Post-processing (local Python)
import rasterio
import numpy as np

# Normalize to 0-1
def preprocess_patch(patch_path):
    with rasterio.open(patch_path) as src:
        data = src.read()  # Shape: (10 bands, 256, 256)
        data = data.transpose(1, 2, 0)  # Shape: (256, 256, 10)
        data = data.astype(np.float32) / 10000  # Sentinel-2 scale factor
        data = np.clip(data, 0, 1)  # Clip outliers
    return data

# Step 8: Create TensorFlow dataset
import tensorflow as tf

def create_tf_dataset(data_dir, batch_size=32):
    def parse_function(filename, label):
        data = preprocess_patch(filename)
        return data, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

### Workflow 2: Segmentation Dataset from Manual Labels (QGIS)

```python
# Step 1: Digitize polygons in QGIS (save as shapefile)
# Step 2: Rasterize labels

import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np

# Load image to get geotransform
with rasterio.open('sentinel2_image.tif') as src:
    meta = src.meta
    image = src.read()

# Load labeled polygons
labels_gdf = gpd.read_file('flood_polygons.shp')

# Rasterize (convert polygons to raster mask)
mask = features.rasterize(
    [(geom, 1) for geom in labels_gdf.geometry],
    out_shape=(meta['height'], meta['width']),
    transform=meta['transform'],
    fill=0,
    dtype=np.uint8
)

# Save mask
with rasterio.open(
    'flood_mask.tif',
    'w',
    driver='GTiff',
    height=meta['height'],
    width=meta['width'],
    count=1,
    dtype=np.uint8,
    crs=meta['crs'],
    transform=meta['transform']
) as dst:
    dst.write(mask, 1)

# Step 3: Create paired image-mask dataset
def load_image_mask_pair(image_path, mask_path):
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1,2,0)  # H×W×C
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # H×W
    return image, mask

# Step 4: Data augmentation (Albumentations library)
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2)
])

def augment(image, mask):
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']
```

---

## Philippine Case Studies

### Case Study 1: PhilSA National Land Cover (2024)

**Scale:** Entire Philippines (~300,000 km²)
**Resolution:** 10m (Sentinel-2)
**Classes:** 10 land cover types
**Model:** ResNet-50 (scene classification on 256×256 patches)
**Training data:** 50,000 labeled patches
**Accuracy:** 91.3% overall
**Processing time:** 8 hours (8× A100 GPUs)
**Update frequency:** Quarterly
**Users:** DENR, DAR, NEDA, LGUs

**Impact:**
- Replaces manual photo-interpretation (previously 6 months)
- Consistent methodology nationwide
- Enables change detection and trend analysis

### Case Study 2: NDRRMC Flood Rapid Mapping (2020-2025)

**Events covered:** 15 major typhoons
**Average response time:** 6 hours (acquisition to map delivery)
**Model:** U-Net (semantic segmentation)
**Input:** Sentinel-1 SAR (pre + post event)
**Performance:** 93% pixel accuracy, 0.89 F1-score
**Area covered per event:** 10,000-50,000 km²
**Integration:** NDRRMC operations dashboard, LGU coordination

**Impact:**
- Faster evacuation planning
- Resource allocation optimization
- Damage assessment for aid distribution

### Case Study 3: DENR Palawan Deforestation Monitoring (2020-2025)

**Area:** Palawan Biosphere Reserve (14,650 km²)
**Model:** Early Fusion U-Net (change detection)
**Input:** Quarterly Sentinel-2 composites
**Performance:** 87% F1-score for forest loss detection
**Processing time:** 2 hours per quarter
**Detected changes (2020-2024):** 12,450 hectares forest loss
**Validation:** 89% user accuracy (field verified)

**Impact:**
- Reduced detection lag from 6 months to 1 month
- Evidence for 45 illegal logging prosecutions
- Informed protected area management decisions

---

## Summary

This guide has covered:
- **Scene classification:** Assign labels to image patches (ResNet, EfficientNet)
- **Semantic segmentation:** Pixel-wise classification (U-Net, DeepLabv3+)
- **Object detection:** Locate and classify objects (YOLO, Faster R-CNN)
- **Change detection:** Temporal analysis for deforestation, disaster damage
- **Time series analysis:** Capture dynamics with CNN+LSTM, Transformers

### Key Takeaways

1. **Architecture choice depends on task:**
   - Classification → ResNet, EfficientNet
   - Segmentation → U-Net
   - Detection → YOLO, Faster R-CNN

2. **Data requirements vary:**
   - Classification: 1,000-10,000 samples per class
   - Segmentation: 100-500 labeled images (augmentation helps)
   - Detection: 1,000-5,000 annotated objects

3. **Transfer learning is essential:**
   - Pre-trained weights reduce data needs by 10×
   - Fine-tune last layers, freeze early layers

4. **Philippine EO is operational:**
   - PhilSA, DENR, NDRRMC actively using CNNs
   - Focus on speed, accuracy, coverage
   - Integration with existing GIS workflows

5. **Data preparation is 80% of the work:**
   - Annotation quality >> quantity
   - Augmentation expands limited datasets
   - Validation with field data critical

---

**Last Updated:** October 2025
**Contact:** training@copphil.org
