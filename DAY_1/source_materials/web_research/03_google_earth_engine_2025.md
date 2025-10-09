# Google Earth Engine Python API Resources (2025)

## Official Documentation

### Main Resources
- **Developer Guide:** https://developers.google.com/earth-engine/
- **Python API Intro:** https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api
- **Python Installation:** https://developers.google.com/earth-engine/guides/python_install
- **GitHub Repository:** https://github.com/google/earthengine-api

### Last Updated
- **Python API Tutorial:** October 2025

## Installation

### Methods
1. **Conda:** `conda install -c conda-forge earthengine-api`
2. **Pip:** `pip install earthengine-api`
3. **Google Colab:** Pre-installed

### Authentication & Initialization
```python
import ee
ee.Authenticate()  # One-time authentication
ee.Initialize()    # Initialize for each session
```

## Sentinel Data in Earth Engine

### Sentinel-2 Datasets
- **Main Collection:** `COPERNICUS/S2_SR` (Surface Reflectance)
- **Catalog:** https://developers.google.com/earth-engine/datasets/catalog/sentinel-2
- **Capabilities:**
  - High-resolution multispectral imagery
  - Vegetation monitoring
  - Soil and water cover analysis
  - Land cover change detection
  - Disaster risk assessment

### Sentinel-1 Datasets
- **Main Collection:** `COPERNICUS/S1_GRD`
- **Algorithms Guide:** https://developers.google.com/earth-engine/guides/sentinel1
- **Capabilities:**
  - Dual-polarization C-band SAR
  - All-weather, day-night imaging
  - Flood mapping
  - Surface moisture detection

## Key Features

### Data Scale
- **Petabytes of Earth observation data**
- Global coverage
- Historical archives
- Regular updates

### Data Sources
- Landsat missions
- Sentinel missions
- MODIS
- Climate datasets
- Terrain data

## Machine Learning in Earth Engine

### Built-in Capabilities
- Random Forest
- CART (Classification and Regression Trees)
- Support Vector Machines (SVM)
- Naive Bayes

### External Framework Integration
For deep learning (CNN, neural networks):
- TensorFlow
- PyTorch
- Train models outside Earth Engine
- Apply models to Earth Engine data

## 2025 Learning Resources

### GBIF Data Blog Tutorial (January 2025)
- **URL:** https://data-blog.gbif.org/post/2025-01-29-understanding-google-earth-engine-gee-and-its-python-api-a-primer-for-gbif-users/
- **Focus:** GEE Python API for biodiversity applications
- **Level:** Beginner to intermediate

### End-to-End GEE Course
- **URL:** https://courses.spatialthoughts.com/end-to-end-gee.html
- **Content:**
  - Comprehensive Earth Engine training
  - Python API modules
  - Hands-on exercises
  - Real-world applications

### Sentinel Data Downloader Tool
- **GitHub:** https://github.com/alessandrosebastianelli/SentinelDataDownloaderTool
- **Purpose:** Automatic dataset creation for AI applications
- **Uses:** Google Earth Engine Python API

## Best Practices

### Authentication
- Required before any Earth Engine operations
- One-time setup per environment
- Uses Google account credentials

### Data Processing
- Leverage cloud computing power
- Process large datasets without downloading
- Use filters to reduce data volume
- Export only final results

### Python Integration
- Works seamlessly with NumPy
- Compatible with pandas/geopandas
- Integrates with matplotlib for visualization
- Export to GeoTIFF, Shapefile, CSV

## Common Workflows

### 1. Image Collection Filtering
```python
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(aoi) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)
```

### 2. Cloud Masking
```python
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
           qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)
```

### 3. Creating Composites
```python
composite = collection.map(maskS2clouds).median()
```

### 4. Exporting Data
```python
task = ee.batch.Export.image.toDrive(
    image=composite,
    description='export_description',
    folder='EarthEngine',
    scale=10,
    region=aoi
)
task.start()
```

## Advanced Topics

### Processing at Scale
- Use `getRegion()` for small areas
- Use `Export` for large areas
- Monitor task status
- Consider computational limits

### Integration with AI/ML
- Export training data
- Sample image collections
- Create feature vectors
- Prepare datasets for TensorFlow/PyTorch

## Resources for Further Learning

1. **Official Tutorials:** https://developers.google.com/earth-engine/tutorials
2. **Community Forum:** https://groups.google.com/g/google-earth-engine-developers
3. **Example Scripts:** Available in Code Editor
4. **Academic Papers:** Search "Google Earth Engine" on Google Scholar
