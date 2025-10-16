"""
Enhanced GLCM Texture Feature Template for Google Earth Engine
Gray-Level Co-occurrence Matrix texture features for land cover classification

IMPROVEMENTS:
- Added error handling and validation
- Performance optimizations
- Additional utility functions
- Better documentation
- Visualization helpers
- Export functionality

Usage: Copy and adapt these functions for your Session 2 notebook
"""

import ee
from typing import List, Optional, Dict, Union


def validate_image(image: ee.Image, required_bands: Optional[List[str]] = None) -> bool:
    """
    Validate that an image contains required bands

    Parameters:
    -----------
    image : ee.Image
        Image to validate
    required_bands : list, optional
        List of band names that must be present

    Returns:
    --------
    bool : True if valid

    Raises:
    -------
    ValueError if validation fails
    """
    if image is None:
        raise ValueError("Image cannot be None")

    if required_bands:
        available_bands = image.bandNames().getInfo()
        missing = [b for b in required_bands if b not in available_bands]
        if missing:
            raise ValueError(f"Missing required bands: {missing}")

    return True


def add_glcm_texture(image: ee.Image,
                     bands: List[str] = ['B8'],
                     radius: int = 1,
                     kernel: Optional[ee.Kernel] = None) -> ee.Image:
    """
    Add GLCM texture features to an image with validation

    Parameters:
    -----------
    image : ee.Image
        Input Sentinel-2 image
    bands : list
        Bands to compute texture from (default: NIR band B8)
    radius : int
        Neighborhood radius in pixels (default: 1 = 3x3 window)
    kernel : ee.Kernel, optional
        Custom kernel for GLCM computation

    Returns:
    --------
    ee.Image with added texture bands

    Texture Features:
    - asm: Angular Second Moment (uniformity)
    - contrast: Local variation
    - corr: Correlation between pixel pairs
    - var: Variance
    - idm: Inverse Difference Moment (homogeneity)
    - savg: Sum Average
    - svar: Sum Variance
    - sent: Sum Entropy
    - ent: Entropy (randomness)
    - dvar: Difference Variance
    - dent: Difference Entropy
    - imcorr1, imcorr2: Information Measures of Correlation
    """

    # Validate inputs
    validate_image(image, bands)

    if radius < 1 or radius > 5:
        raise ValueError("Radius must be between 1 and 5")

    # Select band for texture calculation
    gray = image.select(bands)

    # Compute GLCM with specified radius or kernel
    if kernel:
        glcm = gray.glcmTexture(kernel=kernel)
    else:
        size = radius * 2 + 1
        glcm = gray.glcmTexture(size=size)

    # Return image with GLCM bands added
    return image.addBands(glcm)


def add_selected_glcm(image: ee.Image,
                     bands: List[str] = ['B8'],
                     features: List[str] = ['contrast', 'ent', 'corr'],
                     radius: int = 1) -> ee.Image:
    """
    Add only selected GLCM texture features (faster computation)

    Parameters:
    -----------
    image : ee.Image
        Input Sentinel-2 image
    bands : list
        Bands to compute texture from
    features : list
        Which texture features to keep
        Options: 'contrast', 'corr', 'ent', 'var', 'idm', 'asm',
                'savg', 'svar', 'sent', 'dvar', 'dent'
    radius : int
        Neighborhood radius (default: 1)

    Returns:
    --------
    ee.Image with selected texture bands
    """

    # Validate
    validate_image(image, bands)

    if not features:
        raise ValueError("Must specify at least one feature")

    # Compute all GLCM features
    glcm = add_glcm_texture(image, bands=bands, radius=radius)

    # Build list of band names to select
    texture_bands = []
    for band in bands:
        for feature in features:
            texture_bands.append(f'{band}_{feature}')

    # Select only requested features
    selected = glcm.select(texture_bands)

    return image.addBands(selected)


def glcm_for_classification(image: ee.Image,
                            nir_band: str = 'B8',
                            red_band: str = 'B4',
                            swir_band: Optional[str] = None) -> ee.Image:
    """
    Add optimized GLCM features for land cover classification
    Uses NIR for vegetation texture, Red for overall contrast,
    and optionally SWIR for urban/soil texture

    Parameters:
    -----------
    image : ee.Image
        Input Sentinel-2 image
    nir_band : str
        Near-infrared band name (default: 'B8')
    red_band : str
        Red band name (default: 'B4')
    swir_band : str, optional
        SWIR band for urban/soil texture (e.g., 'B11')

    Returns:
    --------
    ee.Image with classification-optimized texture features
    """

    # Validate
    required = [nir_band, red_band]
    if swir_band:
        required.append(swir_band)
    validate_image(image, required)

    # NIR texture (good for vegetation structure)
    nir_texture = image.select(nir_band).glcmTexture(size=3)
    nir_contrast = nir_texture.select(f'{nir_band}_contrast').rename('nir_texture_contrast')
    nir_entropy = nir_texture.select(f'{nir_band}_ent').rename('nir_texture_entropy')
    nir_corr = nir_texture.select(f'{nir_band}_corr').rename('nir_texture_corr')

    # Red texture (good for overall image structure)
    red_texture = image.select(red_band).glcmTexture(size=3)
    red_contrast = red_texture.select(f'{red_band}_contrast').rename('red_texture_contrast')

    # Collect features
    texture_features = [nir_contrast, nir_entropy, nir_corr, red_contrast]

    # Optional SWIR texture (good for urban/bare soil)
    if swir_band:
        swir_texture = image.select(swir_band).glcmTexture(size=3)
        swir_contrast = swir_texture.select(f'{swir_band}_contrast').rename('swir_texture_contrast')
        swir_entropy = swir_texture.select(f'{swir_band}_ent').rename('swir_texture_entropy')
        texture_features.extend([swir_contrast, swir_entropy])

    # Combine selected features
    texture_bands = ee.Image.cat(texture_features)

    return image.addBands(texture_bands)


def multiscale_glcm(image: ee.Image,
                   band: str = 'B8',
                   radii: List[int] = [1, 2],
                   features: List[str] = ['contrast', 'ent']) -> ee.Image:
    """
    Compute GLCM at multiple scales (ENHANCED VERSION)
    Useful for capturing texture at different spatial frequencies

    Parameters:
    -----------
    image : ee.Image
        Input image
    band : str
        Band to compute texture from
    radii : list
        List of radii (window sizes) to use
    features : list
        Which features to compute (reduces computation time)

    Returns:
    --------
    ee.Image with multi-scale texture features

    WARNING: Computationally expensive! Use sparingly.
    """

    validate_image(image, [band])

    texture_features = []

    for radius in radii:
        glcm = image.select(band).glcmTexture(size=radius * 2 + 1)

        # Select only requested features
        for feature in features:
            band_name = f'{band}_{feature}'
            if band_name in glcm.bandNames().getInfo():
                texture_band = glcm.select(band_name).rename(f'texture_{feature}_r{radius}')
                texture_features.append(texture_band)

    return image.addBands(texture_features)


def add_texture_indices(image: ee.Image,
                        contrast_band: str = 'nir_texture_contrast',
                        entropy_band: str = 'nir_texture_entropy') -> ee.Image:
    """
    Add derived texture indices for enhanced discrimination

    Parameters:
    -----------
    image : ee.Image
        Image with existing texture bands
    contrast_band : str
        Name of contrast texture band
    entropy_band : str
        Name of entropy texture band

    Returns:
    --------
    ee.Image with texture index bands
    """

    # Texture Heterogeneity Index (contrast + entropy)
    thi = image.select(contrast_band).add(image.select(entropy_band)) \
               .rename('texture_heterogeneity_index')

    # Texture Ratio (contrast / entropy) - avoid division by zero
    tr = image.select(contrast_band).divide(
        image.select(entropy_band).add(0.0001)
    ).rename('texture_ratio')

    # Normalized contrast
    contrast = image.select(contrast_band)
    norm_contrast = contrast.divide(contrast.add(1)).rename('normalized_contrast')

    return image.addBands([thi, tr, norm_contrast])


def get_optimal_glcm_bands() -> List[str]:
    """
    Get list of most useful GLCM bands for classification
    Based on research and practical testing

    Returns:
    --------
    List of recommended texture band names
    """
    return [
        'nir_texture_contrast',
        'nir_texture_entropy',
        'nir_texture_corr',
        'red_texture_contrast'
    ]


def batch_glcm_processing(image_collection: ee.ImageCollection,
                         bands: List[str] = ['B8'],
                         features: List[str] = ['contrast', 'ent', 'corr']) -> ee.ImageCollection:
    """
    Apply GLCM texture to an entire image collection

    Parameters:
    -----------
    image_collection : ee.ImageCollection
        Collection of images to process
    bands : list
        Bands to compute texture from
    features : list
        Texture features to compute

    Returns:
    --------
    ee.ImageCollection with texture features added
    """

    def add_texture(img):
        return add_selected_glcm(img, bands=bands, features=features)

    return image_collection.map(add_texture)


def visualize_texture(image: ee.Image,
                     texture_band: str = 'nir_texture_contrast',
                     geometry: ee.Geometry = None,
                     min_val: float = None,
                     max_val: float = None) -> Dict:
    """
    Create visualization parameters for texture bands

    Parameters:
    -----------
    image : ee.Image
        Image with texture bands
    texture_band : str
        Name of texture band to visualize
    geometry : ee.Geometry, optional
        Region for computing statistics
    min_val, max_val : float, optional
        Manual min/max values for visualization

    Returns:
    --------
    Dict with visualization parameters for Map.addLayer()
    """

    # Auto-compute min/max if not provided
    if min_val is None or max_val is None:
        if geometry is None:
            raise ValueError("Must provide geometry for auto min/max computation")

        stats = image.select(texture_band).reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=geometry,
            scale=30,
            maxPixels=1e6
        ).getInfo()

        min_val = min_val or stats.get(f'{texture_band}_p2', 0)
        max_val = max_val or stats.get(f'{texture_band}_p98', 1)

    vis_params = {
        'bands': [texture_band],
        'min': min_val,
        'max': max_val,
        'palette': ['blue', 'white', 'red']
    }

    return vis_params


def export_texture_features(image: ee.Image,
                           geometry: ee.Geometry,
                           description: str,
                           scale: int = 10,
                           crs: str = 'EPSG:4326',
                           max_pixels: int = 1e9) -> ee.batch.Task:
    """
    Export texture features to Google Drive

    Parameters:
    -----------
    image : ee.Image
        Image with texture features
    geometry : ee.Geometry
        Region to export
    description : str
        Export task description
    scale : int
        Export resolution in meters
    crs : str
        Coordinate reference system
    max_pixels : int
        Maximum number of pixels

    Returns:
    --------
    ee.batch.Task : Export task (call .start() to begin)
    """

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        scale=scale,
        region=geometry,
        crs=crs,
        maxPixels=max_pixels,
        fileFormat='GeoTIFF'
    )

    return task


# Example workflows
def example_basic_workflow():
    """
    Example 1: Basic texture feature extraction
    """

    # Define area of interest (Palawan)
    aoi = ee.Geometry.Rectangle([118.5, 9.5, 119.5, 10.5])

    # Load Sentinel-2 composite
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate('2024-01-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Add texture features
    image_with_texture = glcm_for_classification(s2, nir_band='B8', red_band='B4')

    # Add derived indices
    image_enhanced = add_texture_indices(image_with_texture)

    print('Enhanced bands:', image_enhanced.bandNames().getInfo())

    return image_enhanced, aoi


def example_classification_ready():
    """
    Example 2: Prepare classification-ready image with optimal features
    """

    aoi = ee.Geometry.Rectangle([118.5, 9.5, 119.0, 10.0])

    # Load and preprocess
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate('2024-01-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Add spectral bands (RGB + NIR + SWIR)
    spectral = s2.select(['B4', 'B3', 'B2', 'B8', 'B11'])

    # Add NDVI
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Add texture features
    texture = glcm_for_classification(s2, nir_band='B8', red_band='B4', swir_band='B11')

    # Combine all features
    classification_image = spectral.addBands([ndvi, texture])

    # Select only classification features
    feature_bands = ['B4', 'B3', 'B2', 'B8', 'B11', 'NDVI'] + get_optimal_glcm_bands()

    if 'B11' in s2.bandNames().getInfo():
        feature_bands.extend(['swir_texture_contrast', 'swir_texture_entropy'])

    final_image = classification_image.select(feature_bands)

    print(f'Classification features ({len(feature_bands)}): {feature_bands}')

    return final_image, aoi


def example_performance_optimized():
    """
    Example 3: Performance-optimized texture computation
    For large areas or real-time applications
    """

    aoi = ee.Geometry.Rectangle([118.5, 9.5, 119.5, 10.5])

    # Load composite
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate('2024-01-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Use only most discriminative features
    image_optimized = add_selected_glcm(
        s2,
        bands=['B8'],  # Only NIR
        features=['contrast', 'ent'],  # Only 2 features
        radius=1  # Smallest window
    )

    print('Optimized for speed - minimal texture features')
    print('Bands:', image_optimized.bandNames().getInfo())

    return image_optimized, aoi


# Performance tips
PERFORMANCE_TIPS = """
ENHANCED GLCM PERFORMANCE OPTIMIZATION GUIDE:

1. FEATURE SELECTION (Most Important):
   - Use add_selected_glcm() instead of add_glcm_texture()
   - Select only 2-3 features: ['contrast', 'ent', 'corr']
   - Avoid computing all 13 GLCM features unless necessary

2. BAND SELECTION:
   - Single band (B8): Fastest
   - Two bands (B8, B4): Good balance
   - Three+ bands: Use only if critical for accuracy

3. WINDOW SIZE:
   - radius=1 (3x3): Fastest, good for 10m data
   - radius=2 (5x5): Slower but captures more context
   - radius=3+: Very slow, rarely needed

4. PROCESSING STRATEGY:
   - Compute on median/mean composite, NOT individual images
   - Use smaller AOI for testing (< 100 km²)
   - Process in tiles for large areas
   - Export results for reuse

5. MEMORY MANAGEMENT:
   - Avoid calling .getInfo() on large areas
   - Use .reduceRegion() with maxPixels limit
   - Export to Drive instead of client-side processing

6. BATCH PROCESSING:
   - Use batch_glcm_processing() for collections
   - Limit collection size (< 10 images)
   - Monitor task progress in Code Editor

7. VISUALIZATION:
   - Use visualize_texture() for proper scaling
   - Compute percentiles only on small regions
   - Cache visualization parameters

8. DEBUGGING:
   - Test on tiny area first (0.01° x 0.01°)
   - Use .aside(print) to monitor progress
   - Check band names with .bandNames().getInfo()
"""


if __name__ == "__main__":
    print("Enhanced GLCM Template Module")
    print("=" * 60)
    print("\nAvailable functions:")
    print("- add_glcm_texture()")
    print("- add_selected_glcm()")
    print("- glcm_for_classification()")
    print("- multiscale_glcm()")
    print("- add_texture_indices()")
    print("- batch_glcm_processing()")
    print("- visualize_texture()")
    print("- export_texture_features()")
    print("\nExample workflows:")
    print("- example_basic_workflow()")
    print("- example_classification_ready()")
    print("- example_performance_optimized()")
