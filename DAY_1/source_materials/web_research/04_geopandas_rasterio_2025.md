# GeoPandas and Rasterio Tutorial Resources (2025)

## GeoPandas

### Official Documentation
- **Main Site:** https://geopandas.org/
- **Rasterio Integration:** https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html

### Key Features
- Extension of pandas for spatial data
- Built on top of Shapely for geometry operations
- Uses Fiona for file I/O
- Matplotlib integration for visualization

### Common Operations
1. Reading vector data (Shapefile, GeoJSON, etc.)
2. Coordinate reference system (CRS) transformations
3. Spatial joins and overlays
4. Attribute and spatial queries
5. Visualization and mapping

## Rasterio

### Official Documentation
- **Main Site:** https://rasterio.readthedocs.io/
- **Version:** 1.4.3 (latest as of 2025)

### Key Features
- Reads and writes multiple raster formats via GDAL
- NumPy array-based raster processing
- GeoJSON integration
- Built-in matplotlib visualization (rasterio.plot)

### Core Capabilities
- Reading GeoTIFF and other raster formats
- Georeferencing and coordinate transformations
- Band operations and calculations
- Masking and clipping
- Resampling and reprojection

## Combined GeoPandas + Rasterio Workflows

### 1. Sampling Raster Values at Points
**Use Case:** Extract raster values at vector point locations

```python
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen

# Read vector points
points = gpd.read_file('points.shp')

# Read raster
with rasterio.open('raster.tif') as src:
    # Sample raster at point locations
    coords = [(x, y) for x, y in zip(points.geometry.x, points.geometry.y)]
    sampled_values = [x for x in src.sample(coords)]
```

### 2. Cropping Raster with Vector Boundary
**Use Case:** Clip raster to polygon extent

```python
from rasterio.mask import mask

# Read vector polygon
boundary = gpd.read_file('boundary.shp')

# Ensure same CRS
with rasterio.open('raster.tif') as src:
    # Get geometry in proper format
    shapes = boundary.geometry.values

    # Crop raster
    out_image, out_transform = mask(src, shapes, crop=True)
```

### 3. CRS Alignment
**Use Case:** Ensure vector and raster have matching coordinate systems

```python
# Check raster CRS
with rasterio.open('raster.tif') as src:
    raster_crs = src.crs

# Check vector CRS
vector = gpd.read_file('vector.shp')
vector_crs = vector.crs

# Reproject vector if needed
if raster_crs != vector_crs:
    vector = vector.to_crs(raster_crs)
```

## 2025 Python Geospatial Ecosystem

### Core Libraries
According to 2025 guides, essential tools include:

#### Data Management
- **GeoPandas:** Vector data handling
- **Shapely:** Geometric operations
- **Rasterio:** Raster data I/O
- **Fiona:** Vector file I/O
- **PyProj:** Coordinate transformations
- **Rtree:** Spatial indexing

#### Analysis & Modeling
- **PySAL:** Spatial analysis
- **OSMnx:** OpenStreetMap data
- **xarray + rioxarray:** Multi-dimensional arrays
- **GeoAlchemy2:** Database integration

#### Visualization
- **Matplotlib:** Basic plotting
- **Folium:** Interactive maps
- **Contextily:** Basemap tiles
- **Plotly:** Interactive visualizations

## Tutorial Resources

### 1. Carpentries Geospatial Python
- **URL:** https://carpentries-incubator.github.io/geospatial-python/
- **Topics:**
  - Loading raster and vector data
  - Cropping rasters with vector boundaries
  - CRS handling
  - Data visualization

### 2. DataCamp GeoPandas Tutorial
- **URL:** https://www.datacamp.com/tutorial/geopandas-tutorial-geospatial-analysis
- **Level:** Beginner to intermediate
- **Focus:** Introduction to geospatial analysis

### 3. Hatari Labs Tutorials
- **URL:** https://hatarilabs.com/ih-en/extract-point-value-from-a-raster-file-with-python-geopandas-and-rasterio-tutorial
- **Focus:** Practical point-to-raster operations

### 4. WUR Geoscripting
- **URL:** https://geoscripting-wur.github.io/PythonRaster/
- **Focus:** Comprehensive raster handling

## Best Practices for EO Applications

### 1. Memory Management
- Use context managers (`with` statements)
- Read only necessary windows from large rasters
- Process data in chunks for large datasets

### 2. CRS Handling
- Always check and align CRS between datasets
- Use appropriate projections for area calculations
- Document CRS transformations

### 3. Data Types
- Understand NumPy dtypes for raster data
- Handle nodata values appropriately
- Scale values correctly (e.g., Sentinel-2 reflectance)

### 4. Vectorization
- Use NumPy operations instead of loops
- Leverage pandas/geopandas vectorized methods
- Apply functions to entire arrays when possible

## Common Earth Observation Workflows

### 1. NDVI Calculation
```python
with rasterio.open('sentinel2.tif') as src:
    red = src.read(4).astype(float)
    nir = src.read(8).astype(float)
    ndvi = (nir - red) / (nir + red)
```

### 2. Zonal Statistics
```python
import rasterstats

# Calculate statistics for each polygon
stats = rasterstats.zonal_stats(
    polygons,
    'raster.tif',
    stats=['mean', 'std', 'min', 'max']
)
```

### 3. Reprojection
```python
from rasterio.warp import calculate_default_transform, reproject

with rasterio.open('input.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        'crs': 'EPSG:4326',
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('output.tif', 'w', **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs='EPSG:4326'
        )
```

## Integration with Google Colab

### Installation
```python
!pip install geopandas rasterio
```

### Google Drive Integration
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Memory Considerations
- Colab has RAM limitations
- Use smaller subsets for demonstration
- Consider cloud-optimized GeoTIFF (COG) format

## Additional Resources

### Geoapify Guide (2025)
- **URL:** https://www.geoapify.com/python-geospatial-data-analysis/
- Comprehensive overview of 12 Python libraries for geospatial analysis
- Modern best practices and tool selection

### NCEAS Arctic Research Course
- **URL:** https://learning.nceas.ucsb.edu/2023-03-arctic/sections/geopandas.html
- Scalable and reproducible approaches
- Focus on research applications
