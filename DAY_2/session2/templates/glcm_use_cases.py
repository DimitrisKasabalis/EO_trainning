"""
Practical GLCM Use Cases for Philippine Earth Observation

Real-world examples demonstrating texture analysis for:
1. Forest type classification (Primary vs Secondary forest)
2. Urban vs Rural discrimination
3. Mangrove mapping
4. Agricultural land use
5. Disaster assessment (flood/landslide impact)

Each example includes:
- Problem description
- Data preparation
- Texture feature selection
- Implementation code
- Expected results
"""

import ee

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    print("Please authenticate: earthengine authenticate")

from glcm_template_enhanced import (
    glcm_for_classification,
    add_selected_glcm,
    add_texture_indices,
    visualize_texture
)


# =============================================================================
# USE CASE 1: Primary vs Secondary Forest Classification (Palawan)
# =============================================================================

def use_case_1_forest_classification():
    """
    Problem: Distinguish primary forest from secondary forest in Palawan

    Challenge: Spectral signatures are similar, but canopy structure differs
    Solution: Use texture features to capture structural complexity

    Key Insight: Primary forest has more heterogeneous canopy (higher contrast/entropy)
    """

    print("\n" + "="*60)
    print("USE CASE 1: Forest Type Classification")
    print("="*60)

    # Area: Palawan forest region
    palawan_forest = ee.Geometry.Rectangle([118.6, 9.7, 118.9, 10.0])

    # Load Sentinel-2 data (dry season for better visibility)
    s2_forest = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(palawan_forest) \
        .filterDate('2024-01-01', '2024-03-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .median()

    # Strategy: NIR band captures vegetation structure best
    # Use contrast and entropy to measure canopy heterogeneity

    forest_with_texture = add_selected_glcm(
        s2_forest,
        bands=['B8'],  # NIR for vegetation
        features=['contrast', 'ent', 'corr', 'var'],  # Structural features
        radius=2  # 5x5 window to capture canopy patterns
    )

    # Add NDVI for vegetation vigor
    ndvi = s2_forest.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Combine features
    classification_features = forest_with_texture.select([
        'B8', 'B4', 'B11',  # Spectral
        'NDVI',  # Index
        'B8_contrast', 'B8_ent', 'B8_corr', 'B8_var'  # Texture
    ]).addBands(ndvi)

    print("\nForest Classification Features:")
    print("- Spectral: NIR, Red, SWIR")
    print("- Index: NDVI (vegetation health)")
    print("- Texture: Contrast (canopy variation)")
    print("           Entropy (canopy randomness)")
    print("           Correlation (pattern regularity)")
    print("           Variance (intensity variation)")

    print("\nExpected Results:")
    print("✓ Primary Forest: High contrast + High entropy")
    print("✓ Secondary Forest: Medium contrast + Medium entropy")
    print("✓ Degraded Forest: Low contrast + Low entropy")

    # Visualization params
    vis_texture = {
        'bands': ['B8_contrast'],
        'min': 0,
        'max': 500,
        'palette': ['darkgreen', 'yellow', 'red']
    }

    return {
        'image': classification_features,
        'geometry': palawan_forest,
        'vis_params': vis_texture,
        'description': 'Forest type classification using canopy texture'
    }


# =============================================================================
# USE CASE 2: Urban vs Rural Area Discrimination (Metro Manila)
# =============================================================================

def use_case_2_urban_rural():
    """
    Problem: Map urban extent and intensity in Metro Manila

    Challenge: Mixed land use, varying urban density
    Solution: Urban areas have high textural heterogeneity

    Key Insight: Buildings create high contrast and entropy in all bands
    """

    print("\n" + "="*60)
    print("USE CASE 2: Urban vs Rural Mapping")
    print("="*60)

    # Metro Manila and surroundings
    metro_manila = ee.Geometry.Rectangle([120.9, 14.4, 121.2, 14.8])

    # Load Sentinel-2
    s2_urban = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(metro_manila) \
        .filterDate('2024-01-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Strategy: Use Red and SWIR bands - good for urban features
    # Urban areas show high texture in all wavelengths

    urban_with_texture = add_selected_glcm(
        s2_urban,
        bands=['B4', 'B11'],  # Red and SWIR
        features=['contrast', 'ent', 'asm'],  # Heterogeneity measures
        radius=1  # 3x3 for fine urban features
    )

    # Urban indices
    ndbi = s2_urban.normalizedDifference(['B11', 'B8']).rename('NDBI')  # Built-up
    ndvi = s2_urban.normalizedDifference(['B8', 'B4']).rename('NDVI')  # Vegetation

    # Add texture indices
    urban_enhanced = add_texture_indices(
        urban_with_texture,
        contrast_band='B11_contrast',
        entropy_band='B11_ent'
    )

    # Combine features
    urban_features = urban_enhanced.select([
        'B4', 'B11', 'B8',  # Spectral
        'NDBI', 'NDVI',  # Indices
        'B4_contrast', 'B4_ent',  # Red texture
        'B11_contrast', 'B11_ent',  # SWIR texture
        'texture_heterogeneity_index'  # Derived
    ]).addBands([ndbi, ndvi])

    print("\nUrban Classification Features:")
    print("- Spectral: Red, SWIR, NIR")
    print("- Indices: NDBI (built-up), NDVI (vegetation)")
    print("- Texture: Contrast & Entropy from Red and SWIR")
    print("- Derived: Texture Heterogeneity Index")

    print("\nExpected Results:")
    print("✓ Dense Urban: High NDBI + High texture contrast")
    print("✓ Suburban: Medium NDBI + Medium texture")
    print("✓ Rural: Low NDBI + Low texture contrast")

    vis_urban = {
        'bands': ['texture_heterogeneity_index'],
        'min': 0,
        'max': 100,
        'palette': ['blue', 'yellow', 'red']
    }

    return {
        'image': urban_features,
        'geometry': metro_manila,
        'vis_params': vis_urban,
        'description': 'Urban intensity mapping using texture'
    }


# =============================================================================
# USE CASE 3: Mangrove Forest Mapping (Coastal Regions)
# =============================================================================

def use_case_3_mangrove_mapping():
    """
    Problem: Map mangrove forests along Philippine coasts

    Challenge: Mangroves have unique spectral AND structural signature
    Solution: Combine spectral water indices with vegetation texture

    Key Insight: Mangroves = high NDVI + high texture + near water
    """

    print("\n" + "="*60)
    print("USE CASE 3: Mangrove Forest Mapping")
    print("="*60)

    # Coastal Palawan (known mangrove areas)
    coastal_area = ee.Geometry.Rectangle([118.4, 9.5, 118.7, 9.8])

    # Load Sentinel-2 (dry season)
    s2_coastal = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(coastal_area) \
        .filterDate('2024-02-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)) \
        .median()

    # Strategy: Mangroves have distinctive texture due to root structure
    # Use NIR texture + water/vegetation indices

    mangrove_with_texture = add_selected_glcm(
        s2_coastal,
        bands=['B8', 'B3'],  # NIR and Green
        features=['contrast', 'ent', 'corr'],
        radius=1
    )

    # Mangrove-specific indices
    ndvi = s2_coastal.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = s2_coastal.normalizedDifference(['B3', 'B8']).rename('NDWI')
    mndwi = s2_coastal.normalizedDifference(['B3', 'B11']).rename('MNDWI')

    # Mangrove Vegetation Index (simple approach)
    mvi = s2_coastal.select('B8').subtract(s2_coastal.select('B3')).rename('MVI')

    # Combine features
    mangrove_features = mangrove_with_texture.select([
        'B8', 'B4', 'B3', 'B11',  # Spectral
        'NDVI', 'NDWI', 'MNDWI', 'MVI',  # Indices
        'B8_contrast', 'B8_ent', 'B8_corr',  # NIR texture
        'B3_contrast'  # Green texture (water boundary)
    ]).addBands([ndvi, ndwi, mndwi, mvi])

    print("\nMangrove Classification Features:")
    print("- Spectral: NIR, Red, Green, SWIR")
    print("- Indices: NDVI (vegetation)")
    print("           NDWI/MNDWI (water content)")
    print("           MVI (mangrove-specific)")
    print("- Texture: NIR contrast/entropy (canopy structure)")
    print("           Green contrast (water boundary)")

    print("\nExpected Results:")
    print("✓ Mangrove: High NDVI + Positive MNDWI + High NIR texture")
    print("✓ Terrestrial Forest: High NDVI + Negative MNDWI")
    print("✓ Water: Low NDVI + High MNDWI + Low texture")

    # Create mangrove probability mask (simple rule-based)
    mangrove_mask = ndvi.gt(0.3).And(mndwi.gt(-0.3)).And(
        mangrove_features.select('B8_contrast').gt(50)
    ).rename('mangrove_probability')

    vis_mangrove = {
        'bands': ['B8_contrast'],
        'min': 0,
        'max': 200,
        'palette': ['blue', 'green', 'darkgreen']
    }

    return {
        'image': mangrove_features.addBands(mangrove_mask),
        'geometry': coastal_area,
        'vis_params': vis_mangrove,
        'description': 'Mangrove mapping with texture and spectral indices'
    }


# =============================================================================
# USE CASE 4: Agricultural Land Use Classification
# =============================================================================

def use_case_4_agriculture():
    """
    Problem: Classify agricultural land use (rice, corn, vegetables, etc.)

    Challenge: Crops have similar spectral signatures at certain growth stages
    Solution: Texture captures field patterns and crop structure

    Key Insight: Different crops have different spatial patterns
    """

    print("\n" + "="*60)
    print("USE CASE 4: Agricultural Land Use")
    print("="*60)

    # Central Luzon agricultural area
    central_luzon = ee.Geometry.Rectangle([120.5, 15.3, 120.8, 15.6])

    # Load Sentinel-2 (mid-growing season)
    s2_agri = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(central_luzon) \
        .filterDate('2024-02-01', '2024-03-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Strategy: Rice paddies have low texture (uniform)
    # Mixed crops have higher texture (varied)
    # Field boundaries create patterns

    agri_with_texture = add_selected_glcm(
        s2_agri,
        bands=['B8', 'B4'],  # NIR and Red
        features=['contrast', 'ent', 'idm', 'asm'],  # Uniformity measures
        radius=2  # 5x5 to capture field patterns
    )

    # Agricultural indices
    ndvi = s2_agri.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = s2_agri.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': s2_agri.select('B8'),
            'RED': s2_agri.select('B4'),
            'BLUE': s2_agri.select('B2')
        }).rename('EVI')

    # Combine features
    agri_features = agri_with_texture.select([
        'B8', 'B4', 'B3', 'B11',  # Spectral
        'NDVI', 'EVI',  # Vegetation indices
        'B8_contrast', 'B8_ent',  # Variation
        'B8_idm', 'B8_asm',  # Uniformity
        'B4_contrast'  # Red texture
    ]).addBands([ndvi, evi])

    print("\nAgricultural Classification Features:")
    print("- Spectral: NIR, Red, Green, SWIR")
    print("- Indices: NDVI, EVI (crop health)")
    print("- Texture: Contrast/Entropy (field variation)")
    print("           IDM/ASM (field uniformity)")

    print("\nExpected Results:")
    print("✓ Rice Paddies: High NDVI + Low contrast + High IDM (uniform)")
    print("✓ Mixed Crops: High NDVI + High contrast + Low IDM (varied)")
    print("✓ Bare Fields: Low NDVI + Medium contrast")

    vis_agri = {
        'bands': ['B8_idm'],  # Inverse Difference Moment (uniformity)
        'min': 0.3,
        'max': 0.9,
        'palette': ['red', 'yellow', 'green']
    }

    return {
        'image': agri_features,
        'geometry': central_luzon,
        'vis_params': vis_agri,
        'description': 'Agricultural land use with texture uniformity'
    }


# =============================================================================
# USE CASE 5: Flood Impact Assessment (Textural Change)
# =============================================================================

def use_case_5_flood_assessment():
    """
    Problem: Assess flood impact on agricultural and urban areas

    Challenge: Water mapping alone doesn't show impact on structures
    Solution: Compare texture before/after flood event

    Key Insight: Flooding smooths texture (reduces contrast/entropy)
    """

    print("\n" + "="*60)
    print("USE CASE 5: Flood Impact Assessment")
    print("="*60)

    # Central Luzon flood-prone area
    flood_area = ee.Geometry.Rectangle([120.8, 15.0, 121.2, 15.4])

    # Before flood (dry season)
    s2_before = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(flood_area) \
        .filterDate('2024-01-01', '2024-02-28') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # After flood (wet season - simulated)
    s2_after = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(flood_area) \
        .filterDate('2024-07-01', '2024-08-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Compute texture for both periods
    before_texture = add_selected_glcm(
        s2_before,
        bands=['B8', 'B11'],
        features=['contrast', 'ent'],
        radius=1
    )

    after_texture = add_selected_glcm(
        s2_after,
        bands=['B8', 'B11'],
        features=['contrast', 'ent'],
        radius=1
    )

    # Compute texture change
    contrast_change = after_texture.select('B8_contrast').subtract(
        before_texture.select('B8_contrast')
    ).rename('contrast_change')

    entropy_change = after_texture.select('B8_ent').subtract(
        before_texture.select('B8_ent')
    ).rename('entropy_change')

    # Water detection (MNDWI)
    mndwi_after = s2_after.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    water_mask = mndwi_after.gt(0.3)

    # Flood impact index: Large negative texture change + water presence
    flood_impact = contrast_change.multiply(-1).multiply(
        water_mask
    ).rename('flood_impact_score')

    # Combine
    flood_assessment = ee.Image.cat([
        contrast_change,
        entropy_change,
        mndwi_after,
        water_mask.rename('water'),
        flood_impact
    ])

    print("\nFlood Assessment Features:")
    print("- Texture Change: Contrast and Entropy differences")
    print("- Water Detection: MNDWI")
    print("- Impact Score: Texture loss × Water presence")

    print("\nExpected Results:")
    print("✓ Flooded Vegetation: Large negative contrast change")
    print("✓ Flooded Urban: Moderate negative change + debris patterns")
    print("✓ Standing Water: Highly negative change + high MNDWI")
    print("✓ Unaffected: Near-zero change")

    vis_impact = {
        'bands': ['flood_impact_score'],
        'min': 0,
        'max': 100,
        'palette': ['white', 'yellow', 'orange', 'red']
    }

    return {
        'image': flood_assessment,
        'geometry': flood_area,
        'vis_params': vis_impact,
        'description': 'Flood impact using texture change analysis'
    }


# =============================================================================
# USE CASE 6: Quarry and Mining Site Detection
# =============================================================================

def use_case_6_mining_detection():
    """
    Problem: Detect illegal or unregulated mining/quarry sites

    Challenge: Bare soil signatures are similar
    Solution: Mining sites have characteristic texture patterns (terraces, pits)

    Key Insight: Mining creates high contrast geometric patterns
    """

    print("\n" + "="*60)
    print("USE CASE 6: Mining Site Detection")
    print("="*60)

    # Mindanao mining area
    mining_area = ee.Geometry.Rectangle([126.0, 7.5, 126.5, 8.0])

    # Load Sentinel-2
    s2_mining = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(mining_area) \
        .filterDate('2024-01-01', '2024-04-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()

    # Strategy: Mining creates linear/geometric patterns
    # Use SWIR for exposed soil + texture for patterns

    mining_with_texture = add_selected_glcm(
        s2_mining,
        bands=['B11', 'B4'],  # SWIR and Red
        features=['contrast', 'ent', 'corr'],
        radius=2  # Larger window for geometric patterns
    )

    # Mining-relevant indices
    ndvi = s2_mining.normalizedDifference(['B8', 'B4']).rename('NDVI')
    bsi = s2_mining.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))', {
            'SWIR': s2_mining.select('B11'),
            'RED': s2_mining.select('B4'),
            'NIR': s2_mining.select('B8'),
            'BLUE': s2_mining.select('B2')
        }).rename('BSI')  # Bare Soil Index

    # Combine
    mining_features = mining_with_texture.select([
        'B11', 'B4', 'B8',
        'NDVI', 'BSI',
        'B11_contrast', 'B11_ent', 'B11_corr',
        'B4_contrast'
    ]).addBands([ndvi, bsi])

    # Simple mining indicator
    mining_indicator = bsi.gt(0).And(
        mining_features.select('B11_contrast').gt(100)
    ).And(ndvi.lt(0.3)).rename('potential_mining')

    print("\nMining Detection Features:")
    print("- Spectral: SWIR, Red, NIR")
    print("- Indices: NDVI (low vegetation)")
    print("           BSI (exposed soil)")
    print("- Texture: High SWIR contrast (geometric patterns)")
    print("           Correlation (regularity)")

    print("\nExpected Results:")
    print("✓ Active Mining: High BSI + High contrast + Low NDVI")
    print("✓ Natural Bare Soil: High BSI + Low contrast")
    print("✓ Vegetated: Low BSI + Variable contrast")

    vis_mining = {
        'bands': ['B11_contrast'],
        'min': 0,
        'max': 300,
        'palette': ['yellow', 'orange', 'red']
    }

    return {
        'image': mining_features.addBands(mining_indicator),
        'geometry': mining_area,
        'vis_params': vis_mining,
        'description': 'Mining site detection with geometric patterns'
    }


# =============================================================================
# Main Example Runner
# =============================================================================

def run_all_use_cases():
    """
    Run all use cases and print summaries
    """

    use_cases = [
        ("Forest Classification", use_case_1_forest_classification),
        ("Urban Mapping", use_case_2_urban_rural),
        ("Mangrove Mapping", use_case_3_mangrove_mapping),
        ("Agricultural Classification", use_case_4_agriculture),
        ("Flood Assessment", use_case_5_flood_assessment),
        ("Mining Detection", use_case_6_mining_detection)
    ]

    results = {}

    print("\n" + "="*60)
    print("GLCM USE CASES FOR PHILIPPINE EO")
    print("="*60)

    for name, func in use_cases:
        try:
            result = func()
            results[name] = result
            print(f"\n✓ {name}: Ready")
        except Exception as e:
            print(f"\n✗ {name}: Error - {e}")

    print("\n" + "="*60)
    print(f"Completed {len(results)}/{len(use_cases)} use cases")
    print("="*60)

    return results


# Quick reference guide
USE_CASE_SUMMARY = """
GLCM USE CASE QUICK REFERENCE:

1. FOREST CLASSIFICATION:
   - Features: NIR contrast + entropy
   - Window: 5x5 (radius=2)
   - Key: Canopy heterogeneity

2. URBAN MAPPING:
   - Features: Red + SWIR texture
   - Window: 3x3 (radius=1)
   - Key: Structural complexity

3. MANGROVE MAPPING:
   - Features: NIR + Green texture + MNDWI
   - Window: 3x3 (radius=1)
   - Key: Wetland vegetation structure

4. AGRICULTURE:
   - Features: NIR uniformity (IDM, ASM)
   - Window: 5x5 (radius=2)
   - Key: Field homogeneity

5. FLOOD ASSESSMENT:
   - Features: Texture change (before/after)
   - Window: 3x3 (radius=1)
   - Key: Smoothing effect of water

6. MINING DETECTION:
   - Features: SWIR contrast + correlation
   - Window: 5x5 (radius=2)
   - Key: Geometric patterns

GENERAL TIPS:
- Always combine texture with spectral indices
- Start with simple features (contrast, entropy)
- Use appropriate window size for target features
- Validate with ground truth data
- Consider seasonal effects
"""


if __name__ == "__main__":
    print(USE_CASE_SUMMARY)
    print("\nTo run examples, call individual functions:")
    print("  result = use_case_1_forest_classification()")
    print("  result = use_case_2_urban_rural()")
    print("  etc.")
    print("\nOr run all:")
    print("  results = run_all_use_cases()")
