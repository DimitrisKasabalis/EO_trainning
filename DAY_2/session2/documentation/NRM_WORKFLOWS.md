# Natural Resource Management (NRM) Workflows
**Session 2: Applied EO Classification for Philippine Conservation**

---

## Overview

This guide provides practical workflows for applying Random Forest land cover classification to real Natural Resource Management challenges in the Philippines, with focus on Palawan conservation.

---

## Table of Contents

1. [Forest Monitoring & REDD+](#forest-monitoring)
2. [Deforestation Detection & Alerts](#deforestation-detection)
3. [Agricultural Expansion Tracking](#agricultural-expansion)
4. [Protected Area Monitoring](#protected-area-monitoring)
5. [Coastal & Mangrove Assessment](#coastal-mangrove)
6. [Mining Impact Assessment](#mining-impact)
7. [Stakeholder Reporting](#stakeholder-reporting)

---

## <a name="forest-monitoring"></a>1. Forest Monitoring & REDD+

### Objective
Monitor forest cover changes for DENR reporting and REDD+ MRV (Monitoring, Reporting, Verification).

### Requirements
- **Temporal:** Annual updates (dry season consistent)
- **Accuracy:** >85% overall, >90% for forest classes
- **Classes:** Primary forest, secondary forest, non-forest
- **Outputs:** Area statistics, change maps, carbon estimates

### Workflow

#### Step 1: Annual Classification
```python
def classify_forest_annual(aoi, year, training_polygons, classifier):
    """
    Create annual forest classification
    """
    # Dry season composite (consistent season each year)
    composite = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate(f'{year}-01-01', f'{year}-05-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_s2_clouds) \
        .median() \
        .clip(aoi)
    
    # Add features
    features = add_indices(composite)
    features = add_texture(features)  # Important for forest types
    
    # Classify
    classified = features.classify(classifier).rename(f'lulc_{year}')
    
    return classified

# Annual series
years = [2020, 2021, 2022, 2023, 2024]
annual_classifications = {}

for year in years:
    classified = classify_forest_annual(aoi, year, training_polygons, classifier)
    annual_classifications[year] = classified
    print(f"✓ {year} classification complete")
```

#### Step 2: Calculate Forest Area
```python
def calculate_forest_area(classified, aoi, forest_classes=[1, 2]):
    """
    Calculate total forest area
    """
    # Create forest mask
    forest = classified.eq(forest_classes[0])
    for class_id in forest_classes[1:]:
        forest = forest.Or(classified.eq(class_id))
    
    # Calculate area
    area = forest.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    
    area_ha = ee.Number(area.get(classified.bandNames().get(0))).divide(10000)
    return area_ha.getInfo()

# Annual forest cover
forest_timeseries = {}
for year, classified in annual_classifications.items():
    area_ha = calculate_forest_area(classified, aoi)
    forest_timeseries[year] = area_ha
    print(f"{year}: {area_ha:,.2f} ha")
```

#### Step 3: REDD+ Reporting
```python
import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame for reporting
df = pd.DataFrame({
    'Year': list(forest_timeseries.keys()),
    'Forest_Area_ha': list(forest_timeseries.values())
})

# Calculate annual change
df['Annual_Change_ha'] = df['Forest_Area_ha'].diff()
df['Annual_Change_pct'] = (df['Annual_Change_ha'] / df['Forest_Area_ha'].shift(1)) * 100

# Estimate carbon (simplified)
# Average biomass: 150 tC/ha for Philippine forests
CARBON_PER_HA = 150  # tonnes C per hectare
CO2_PER_C = 3.67     # CO2 equivalent

df['Forest_Carbon_tC'] = df['Forest_Area_ha'] * CARBON_PER_HA
df['Carbon_Change_tC'] = df['Forest_Carbon_tC'].diff()
df['CO2_Change_tCO2e'] = df['Carbon_Change_tC'] * CO2_PER_C

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Forest area trend
ax1.plot(df['Year'], df['Forest_Area_ha'], 'o-', linewidth=2, markersize=8, color='darkgreen')
ax1.fill_between(df['Year'], df['Forest_Area_ha'], alpha=0.3, color='green')
ax1.set_ylabel('Forest Area (hectares)', fontweight='bold')
ax1.set_title('Palawan Forest Cover Trend (2020-2024)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Annual change
ax2.bar(df['Year'][1:], df['Annual_Change_ha'][1:], color=['red' if x < 0 else 'green' for x in df['Annual_Change_ha'][1:]])
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Annual Change (hectares)', fontweight='bold')
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_title('Annual Forest Change', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forest_monitoring_report.png', dpi=300)
plt.show()

# Export to CSV for REDD+ reporting
df.to_csv('palawan_forest_timeseries_2020_2024.csv', index=False)
print("✓ REDD+ report generated")
```

---

## <a name="deforestation-detection"></a>2. Deforestation Detection & Alerts

### Objective
Detect and alert on forest loss events for rapid response.

### Workflow

#### Early Warning System
```python
def detect_recent_deforestation(aoi, baseline_year=2023, current_year=2024):
    """
    Detect forest loss for early warning
    """
    # Baseline forest map
    baseline = classify_forest_annual(aoi, baseline_year, training, classifier)
    baseline_forest = baseline.eq(1).Or(baseline.eq(2))
    
    # Current forest map
    current = classify_forest_annual(aoi, current_year, training, classifier)
    current_forest = current.eq(1).Or(current.eq(2))
    
    # Detect loss
    forest_loss = baseline_forest.And(current_forest.Not())
    
    # Calculate loss area
    loss_area_ha = forest_loss.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    ).values().get(0)
    
    loss_area_ha = ee.Number(loss_area_ha).divide(10000).getInfo()
    
    return {
        'loss_map': forest_loss,
        'loss_area_ha': loss_area_ha,
        'baseline_year': baseline_year,
        'current_year': current_year
    }

# Run detection
alert = detect_recent_deforestation(aoi, 2023, 2024)

if alert['loss_area_ha'] > 100:  # Alert threshold: 100 hectares
    print(f"🚨 ALERT: {alert['loss_area_ha']:,.2f} ha forest loss detected!")
    print(f"   Period: {alert['baseline_year']}-{alert['current_year']}")
else:
    print(f"✓ Forest loss within normal range: {alert['loss_area_ha']:,.2f} ha")
```

#### Hotspot Identification
```python
def identify_hotspots(loss_map, aoi, radius_m=1000, threshold_pct=5):
    """
    Identify deforestation hotspots
    """
    # Focal statistics (proportion of loss in neighborhood)
    kernel = ee.Kernel.circle(radius=radius_m, units='meters')
    hotspot_intensity = loss_map.focalMean(kernel=kernel).multiply(100)
    
    # Mask to hotspots above threshold
    hotspots = hotspot_intensity.updateMask(hotspot_intensity.gt(threshold_pct))
    
    # Calculate hotspot area
    hotspot_area = hotspots.gt(0).multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13
    )
    
    return {
        'hotspot_map': hotspots,
        'max_intensity': hotspot_intensity.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=aoi,
            scale=100
        ).values().get(0).getInfo(),
        'n_hotspot_pixels': hotspot_area.values().get(0).getInfo()
    }

hotspots = identify_hotspots(alert['loss_map'], aoi)
print(f"Hotspot max intensity: {hotspots['max_intensity']:.1f}% loss")
```

#### Generate Alert Report
```python
def generate_alert_report(alert, hotspots, output_file='deforestation_alert.txt'):
    """
    Generate text report for stakeholders
    """
    report = f"""
==============================================================
DEFORESTATION ALERT REPORT
Palawan Biosphere Reserve
==============================================================

DETECTION PERIOD: {alert['baseline_year']} - {alert['current_year']}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- SUMMARY ---

Total Forest Loss: {alert['loss_area_ha']:,.2f} hectares

Hotspot Analysis:
  - Radius: 1 km
  - Maximum intensity: {hotspots['max_intensity']:.1f}% local forest loss
  - Priority intervention areas identified

--- RECOMMENDED ACTIONS ---

1. IMMEDIATE:
   - Field verification of hotspot locations
   - Check for illegal logging permits
   - Deploy rangers to priority areas

2. SHORT-TERM (1 week):
   - Interview local communities
   - Document driver of loss (mining, agriculture, roads)
   - Coordinate with local government units (LGUs)

3. MEDIUM-TERM (1 month):
   - Update protected area management plan
   - Enhanced patrolling in hotspot zones
   - Community engagement for forest protection

--- NEXT MONITORING ---

Recommended: Monthly updates during high-risk season
Next analysis: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}

==============================================================
Contact: DENR-Palawan | Email: denr.palawan@denr.gov.ph
==============================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Alert report saved: {output_file}")
    return report

report = generate_alert_report(alert, hotspots)
print(report)
```

---

## <a name="agricultural-expansion"></a>3. Agricultural Expansion Tracking

### Objective
Monitor agricultural expansion and conversion from natural habitats.

### Workflow

```python
def track_agricultural_expansion(aoi, year1, year2, forest_classes=[1,2], ag_class=4):
    """
    Track forest to agriculture conversion
    """
    # Classifications for both years
    class_year1 = classify_forest_annual(aoi, year1, training, classifier)
    class_year2 = classify_forest_annual(aoi, year2, training, classifier)
    
    # Create masks
    forest_year1 = class_year1.eq(forest_classes[0])
    for fc in forest_classes[1:]:
        forest_year1 = forest_year1.Or(class_year1.eq(fc))
    
    ag_year2 = class_year2.eq(ag_class)
    
    # Forest to agriculture
    forest_to_ag = forest_year1.And(ag_year2)
    
    # Calculate area
    conversion_area = forest_to_ag.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    
    area_ha = ee.Number(conversion_area.values().get(0)).divide(10000).getInfo()
    
    # Analyze proximity to existing agriculture
    ag_year1 = class_year1.eq(ag_class)
    distance_to_ag = ag_year1.Not().distance(ee.Kernel.euclidean(5000, 'meters'))
    
    # New agriculture proximity
    proximity_stats = distance_to_ag.updateMask(forest_to_ag).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=100
    )
    
    avg_distance_m = ee.Number(proximity_stats.values().get(0)).getInfo()
    
    return {
        'conversion_map': forest_to_ag,
        'area_ha': area_ha,
        'avg_distance_to_existing_ag_m': avg_distance_m,
        'year1': year1,
        'year2': year2
    }

# Track expansion
ag_expansion = track_agricultural_expansion(aoi, 2020, 2024)

print(f"Forest → Agriculture Conversion ({ag_expansion['year1']}-{ag_expansion['year2']})")
print(f"  Area: {ag_expansion['area_ha']:,.2f} ha")
print(f"  Avg distance to existing agriculture: {ag_expansion['avg_distance_to_existing_ag_m']:.0f} m")

if ag_expansion['avg_distance_to_existing_ag_m'] < 500:
    print("  ⚠️ Pattern: Expansion of existing agricultural areas")
elif ag_expansion['avg_distance_to_existing_ag_m'] > 2000:
    print("  🚨 Pattern: New agricultural frontiers (high concern)")
```

---

## <a name="protected-area-monitoring"></a>4. Protected Area Monitoring

### Objective
Monitor land cover changes within and around protected areas.

### Workflow

```python
def monitor_protected_area(pa_boundary, buffer_km=5):
    """
    Assess protected area and buffer zone
    """
    # Create buffer
    buffer = pa_boundary.buffer(buffer_km * 1000)  # Convert km to meters
    buffer_only = buffer.difference(pa_boundary)
    
    # Classify both zones
    year = 2024
    classified = classify_forest_annual(buffer, year, training, classifier)
    
    # Statistics for PA core
    pa_stats = {}
    for class_id in range(1, 9):
        mask = classified.eq(class_id)
        area = mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=pa_boundary,
            scale=10,
            maxPixels=1e13
        )
        pa_stats[class_id] = ee.Number(area.values().get(0)).divide(10000).getInfo()
    
    # Statistics for buffer
    buffer_stats = {}
    for class_id in range(1, 9):
        mask = classified.eq(class_id)
        area = mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer_only,
            scale=10,
            maxPixels=1e13
        )
        buffer_stats[class_id] = ee.Number(area.values().get(0)).divide(10000).getInfo()
    
    return {
        'pa_core': pa_stats,
        'buffer_zone': buffer_stats,
        'classified': classified
    }

# Example: Palawan protected area
pa_boundary = ee.Geometry.Rectangle([118.6, 9.6, 119.4, 10.4])  # Example boundary

pa_monitoring = monitor_protected_area(pa_boundary, buffer_km=5)

# Report
print("PROTECTED AREA MONITORING REPORT")
print("=" * 60)

class_names = {1: 'Primary Forest', 2: 'Secondary Forest', 3: 'Mangroves', 
               4: 'Agricultural', 5: 'Grassland', 6: 'Water', 
               7: 'Urban', 8: 'Bare Soil'}

print("\\nCore Protected Area:")
for class_id, area in pa_monitoring['pa_core'].items():
    pct = (area / sum(pa_monitoring['pa_core'].values())) * 100
    print(f"  {class_names[class_id]:20s}: {area:8,.2f} ha ({pct:5.1f}%)")

print("\\nBuffer Zone (5 km):")
for class_id, area in pa_monitoring['buffer_zone'].items():
    pct = (area / sum(pa_monitoring['buffer_zone'].values())) * 100
    print(f"  {class_names[class_id]:20s}: {area:8,.2f} ha ({pct:5.1f}%)")

# Assess pressure
forest_core = pa_monitoring['pa_core'][1] + pa_monitoring['pa_core'][2]
forest_buffer = pa_monitoring['buffer_zone'][1] + pa_monitoring['buffer_zone'][2]
urban_buffer = pa_monitoring['buffer_zone'][7]
ag_buffer = pa_monitoring['buffer_zone'][4]

forest_core_pct = (forest_core / sum(pa_monitoring['pa_core'].values())) * 100
forest_buffer_pct = (forest_buffer / sum(pa_monitoring['buffer_zone'].values())) * 100

print("\\nASSESSMENT:")
print(f"  Core forest cover: {forest_core_pct:.1f}%")
print(f"  Buffer forest cover: {forest_buffer_pct:.1f}%")

if forest_core_pct < 70:
    print("  🚨 Core integrity compromised - priority intervention needed")
elif forest_buffer_pct < 50:
    print("  ⚠️ Buffer zone under pressure - enhanced management needed")
else:
    print("  ✓ Protected area in good condition")
```

---

## <a name="coastal-mangrove"></a>5. Coastal & Mangrove Assessment

### Objective
Monitor mangrove extent and coastal changes.

### Workflow

```python
def assess_mangroves(coastal_aoi, year, mangrove_class=3):
    """
    Detailed mangrove assessment
    """
    # Classify
    classified = classify_forest_annual(coastal_aoi, year, training, classifier)
    
    # Extract mangroves
    mangroves = classified.eq(mangrove_class)
    
    # Calculate area
    area = mangroves.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=coastal_aoi,
        scale=10,
        maxPixels=1e13
    )
    
    area_ha = ee.Number(area.values().get(0)).divide(10000).getInfo()
    
    # Fragmentation analysis
    # Connected components
    connected = mangroves.connectedPixelCount(100)
    
    # Patch statistics
    large_patches = connected.gte(100)  # Patches >=100 pixels (1 ha)
    large_patch_area = large_patches.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=coastal_aoi,
        scale=10,
        maxPixels=1e13
    )
    
    large_patch_ha = ee.Number(large_patch_area.values().get(0)).divide(10000).getInfo()
    fragmentation_index = large_patch_ha / area_ha if area_ha > 0 else 0
    
    return {
        'total_area_ha': area_ha,
        'large_patch_area_ha': large_patch_ha,
        'fragmentation_index': fragmentation_index,  # 1 = no fragmentation
        'mangrove_map': mangroves
    }

# Multi-year mangrove tracking
coastal_zone = ee.Geometry.Rectangle([118.4, 9.0, 119.0, 10.0])

mangrove_trend = {}
for year in [2020, 2022, 2024]:
    result = assess_mangroves(coastal_zone, year)
    mangrove_trend[year] = result
    print(f"{year}: {result['total_area_ha']:,.2f} ha (Fragmentation: {result['fragmentation_index']:.2f})")
```

---

## <a name="mining-impact"></a>6. Mining Impact Assessment

### Objective
Detect and quantify mining impacts on surrounding ecosystems.

### Workflow

```python
def assess_mining_impact(aoi, year, mining_class=8, buffer_km=2):
    """
    Assess mining footprint and buffer zone impacts
    """
    # Classify
    classified = classify_forest_annual(aoi, year, training, classifier)
    
    # Extract mining areas
    mining = classified.eq(mining_class)
    
    # Mining area
    mining_area = mining.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    mining_ha = ee.Number(mining_area.values().get(0)).divide(10000).getInfo()
    
    # Create buffer around mining
    mining_vector = mining.reduceToVectors(
        geometry=aoi,
        scale=30,
        geometryType='polygon'
    )
    
    mining_buffer = mining_vector.geometry().buffer(buffer_km * 1000)
    
    # Forest loss in buffer
    forest = classified.eq(1).Or(classified.eq(2))
    forest_in_buffer = forest.clip(mining_buffer)
    
    buffer_area = mining_buffer.area().divide(10000).getInfo()
    forest_buffer_area = forest_in_buffer.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=mining_buffer,
        scale=10,
        maxPixels=1e13
    )
    forest_buffer_ha = ee.Number(forest_buffer_area.values().get(0)).divide(10000).getInfo()
    
    forest_buffer_pct = (forest_buffer_ha / buffer_area) * 100 if buffer_area > 0 else 0
    
    return {
        'mining_area_ha': mining_ha,
        'buffer_area_ha': buffer_area,
        'forest_in_buffer_ha': forest_buffer_ha,
        'forest_buffer_pct': forest_buffer_pct
    }

# Assess mining
mining_impact = assess_mining_impact(aoi, 2024)

print("MINING IMPACT ASSESSMENT")
print(f"  Direct mining footprint: {mining_impact['mining_area_ha']:,.2f} ha")
print(f"  Impact buffer (2 km): {mining_impact['buffer_area_ha']:,.2f} ha")
print(f"  Forest in buffer: {mining_impact['forest_in_buffer_ha']:,.2f} ha ({mining_impact['forest_buffer_pct']:.1f}%)")

if mining_impact['forest_buffer_pct'] < 30:
    print("  🚨 Severe degradation in mining buffer zone")
```

---

## <a name="stakeholder-reporting"></a>7. Stakeholder Reporting

### Executive Summary Template
```python
def generate_executive_summary(aoi, year, classifications):
    """
    Generate executive summary for decision-makers
    """
    summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║   PALAWAN LAND COVER MONITORING - EXECUTIVE SUMMARY {year}      ║
╚══════════════════════════════════════════════════════════════════╝

STUDY AREA: Palawan Biosphere Reserve
REPORTING PERIOD: {year}
TOTAL AREA: {aoi.area().divide(1e6).getInfo():,.2f} km²

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FINDINGS:

1. FOREST COVER
   • Primary Forest: {classifications[1]:,.2f} ha
   • Secondary Forest: {classifications[2]:,.2f} ha
   • Total Forest: {classifications[1] + classifications[2]:,.2f} ha
   • Forest Cover: {((classifications[1] + classifications[2]) / sum(classifications.values())) * 100:.1f}%

2. CONSERVATION PRIORITIES
   • Mangroves: {classifications[3]:,.2f} ha
   • Protected status: [Requires field verification]

3. HUMAN PRESSURES
   • Agricultural Land: {classifications[4]:,.2f} ha
   • Urban/Built-up: {classifications[7]:,.2f} ha
   • Mining/Bare Soil: {classifications[8]:,.2f} ha

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDATIONS:

✓ Continue monthly monitoring during high-risk season
✓ Validate hotspots through field surveys
✓ Update protected area management plans
✓ Engage local communities in forest protection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREPARED BY: CopPhil EO Training Program
DATE: {datetime.now().strftime('%B %d, %Y')}
CONTACT: [Your Contact Information]

╚══════════════════════════════════════════════════════════════════╝
"""
    return summary
```

---

## Best Practices

### For All Workflows:

1. **Temporal Consistency**
   - Use same season each year (dry season recommended for Philippines)
   - Maintain consistent cloud thresholds
   - Document methodology changes

2. **Validation**
   - Ground-truth critical areas
   - Engage local knowledge
   - Cross-check with high-resolution imagery

3. **Communication**
   - Tailor reports to audience (technical vs executive)
   - Include clear visualizations
   - Provide actionable recommendations

4. **Documentation**
   - Record all parameters
   - Save code and workflows
   - Version control for long-term monitoring

---

**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 2
