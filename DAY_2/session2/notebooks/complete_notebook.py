"""
Complete Session 2 notebook with Parts C and D
Adds optimization and change detection sections
"""

import nbformat as nbf

# Load existing notebook
with open('session2_extended_lab_STUDENT.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Add remaining cells for Parts C and D

additional_cells = []

# =============================================================================
# PART C CONTINUED
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""### Visualize Tuning Results"""))

additional_cells.append(nbf.v4.new_code_cell("""# Plot tuning results
fig, ax = plt.subplots(figsize=(10, 6))

trees_list = list(results.keys())
acc_list = [v*100 for v in results.values()]

ax.plot(trees_list, acc_list, 'o-', linewidth=2, markersize=10, color='#2196F3')
ax.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Random Forest Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([min(acc_list)-2, max(acc_list)+2])

# Highlight best
best_idx = acc_list.index(max(acc_list))
ax.scatter([trees_list[best_idx]], [acc_list[best_idx]], 
           s=200, c='red', marker='*', zorder=5, label='Best')
ax.legend()

plt.tight_layout()
plt.show()"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## C.2: Post-Processing

Reduce "salt-and-pepper" noise using majority filtering.
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Apply majority filter
print("Applying post-processing...")

# Focal mode filter (3x3 window)
classified_filtered = classified.focal_mode(radius=1, kernelType='square')

print("✓ Majority filter applied (3x3 window)")

# Compare before/after
Map4 = geemap.Map(center=[10.0, 119.0], zoom=11)
Map4.addLayer(classified, {'min': 1, 'max': 8, 'palette': class_colors}, 
              'Original Classification')
Map4.addLayer(classified_filtered, {'min': 1, 'max': 8, 'palette': class_colors}, 
              'Filtered Classification')
Map4.addLayerControl()
Map4"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### TODO Exercise 5: Compare Filtering Results

**Task:** Zoom in on the map above and compare the original vs filtered classification.

**Questions:**
1. What differences do you notice?
2. Does filtering improve visual quality?
3. Are there any disadvantages to filtering?

*YOUR ANSWERS HERE*

---
"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### 🎉 Part C Complete!

You've successfully:
- ✅ Performed hyperparameter tuning
- ✅ Identified optimal tree count
- ✅ Applied post-processing filters

---
"""))

# =============================================================================
# PART D: CHANGE DETECTION
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# Part D: NRM Applications & Change Detection (15 minutes)

Apply classification to conservation challenges: detect forest loss from 2020 to 2024.

---

## D.1: Create 2020 Classification

We'll create a comparable classification for 2020 to detect changes.
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Create 2020 dry season composite
print("Creating 2020 composite for comparison...")

dry_2020 = ee.ImageCollection('COPERNICUS/S2_SR') \\
    .filterBounds(aoi) \\
    .filterDate('2020-01-01', '2020-05-31') \\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \\
    .map(mask_s2_clouds) \\
    .median() \\
    .clip(aoi)

# Add indices (simplified for speed)
def add_basic_indices(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return img.addBands([ndvi, ndwi])

dry_2020 = add_basic_indices(dry_2020)

# Use simplified feature set for 2020
features_2020 = dry_2020.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI'])

# Classify 2020
classified_2020 = features_2020.classify(classifier).rename('classification_2020')

print("✓ 2020 classification complete")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## D.2: Detect Forest Loss
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Detect forest loss (class 1 or 2 in 2020, NOT in 2024)
print("Detecting forest loss...")

# Create forest masks
forest_2020 = classified_2020.eq(1).Or(classified_2020.eq(2))
forest_2024 = classified_filtered.eq(1).Or(classified_filtered.eq(2))

# Forest loss: was forest in 2020, not forest in 2024
forest_loss = forest_2020.And(forest_2024.Not()).rename('forest_loss')

print("✓ Forest loss detected")

# Calculate forest loss area
loss_area = forest_loss.multiply(ee.Image.pixelArea()).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=aoi,
    scale=10,
    maxPixels=1e13
)

loss_ha = ee.Number(loss_area.get('forest_loss')).divide(10000).getInfo()
print(f"\\n🚨 Forest Loss (2020-2024): {loss_ha:,.2f} hectares")"""))

additional_cells.append(nbf.v4.new_code_cell("""# Visualize forest loss
Map5 = geemap.Map(center=[10.0, 119.0], zoom=10)

# Background: 2024 classification
Map5.addLayer(classified_filtered, {'min': 1, 'max': 8, 'palette': class_colors}, 
              '2024 Land Cover', False)

# Highlight forest loss in red
Map5.addLayer(forest_loss.updateMask(forest_loss), {'palette': ['red']}, 
              'Forest Loss (2020-2024)')

Map5.add_legend(
    title='Change Detection',
    labels=['Forest Loss', 'No Change'],
    colors=['red', 'lightgray']
)

Map5"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## D.3: Identify Deforestation Hotspots
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Detect hotspots using focal statistics
print("Identifying deforestation hotspots...")

# Create circular kernel (1km radius)
kernel = ee.Kernel.circle(radius=1000, units='meters')

# Calculate proportion of loss pixels in neighborhood
hotspots = forest_loss.focalMean(kernel=kernel).multiply(100).rename('hotspot_intensity')

print("✓ Hotspot analysis complete")

# Visualize hotspots
Map6 = geemap.Map(center=[10.0, 119.0], zoom=9)

hotspot_vis = {
    'min': 0,
    'max': 10,
    'palette': ['white', 'yellow', 'orange', 'red', 'darkred']
}

Map6.addLayer(hotspots.updateMask(hotspots.gt(0.5)), hotspot_vis, 'Deforestation Hotspots')
Map6.add_colorbar(hotspot_vis, label='Forest Loss Intensity (%)')
Map6"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## D.4: Change Matrix & Statistics
"""))

additional_cells.append(nbf.v4.new_code_cell("""### TODO Exercise 6: Analyze Land Use Transitions

**Task:** Calculate how much forest was converted to different land uses.

**Questions to investigate:**
1. How much forest → agriculture conversion occurred?
2. How much forest → bare soil (mining)?
3. Which class replaced forests the most?

**Code template:**
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Create change matrix (from_class * 10 + to_class)
change_matrix = classified_2020.multiply(10).add(classified_filtered).rename('change_code')

# Example: Forest (1) to Agriculture (4) = change code 14
# Forest (1) to Bare Soil (8) = change code 18

# TODO: Calculate specific transitions
# forest_to_ag = change_matrix.eq(14).Or(change_matrix.eq(24))  # Primary or Secondary → Ag

# YOUR CODE HERE to calculate areas for different transitions
"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## D.5: Generate Stakeholder Report
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Create summary report
print("=" * 70)
print("PALAWAN BIOSPHERE RESERVE - CHANGE DETECTION REPORT (2020-2024)")
print("=" * 70)

print(f"\\nStudy Area: {aoi.area().divide(1e6).getInfo():.2f} km²")
print(f"Analysis Period: 2020-2024 (4 years)")
print(f"\\n--- KEY FINDINGS ---\\n")

# Forest loss
total_area_km2 = aoi.area().divide(1e6).getInfo()
loss_percent = (loss_ha / (total_area_km2 * 100)) * 100

print(f"🚨 Total Forest Loss: {loss_ha:,.2f} hectares ({loss_percent:.2f}% of study area)")
print(f"📉 Annual Loss Rate: {loss_ha/4:,.2f} hectares/year")

# 2024 Land cover summary
print(f"\\n--- 2024 LAND COVER DISTRIBUTION ---\\n")
for class_name, area in area_stats.items():
    percent = (area / (total_area_km2 * 100)) * 100
    print(f"  {class_name:20s}: {area:10,.2f} ha ({percent:5.2f}%)")

print(f"\\n--- CONSERVATION IMPLICATIONS ---\\n")
print("• Continued monitoring recommended")
print("• Priority intervention zones identified via hotspot analysis")
print("• Update DENR forest cover database")
print("• Inform REDD+ MRV reporting")

print("\\n" + "=" * 70)
print("Report generated:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("=" * 70)"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## D.6: Export Results
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Export classification to Google Drive
print("Preparing exports...")

# Export 2024 classification
export_task = ee.batch.Export.image.toDrive(
    image=classified_filtered.toUint8(),
    description='Palawan_LULC_2024',
    folder='EO_Training_Exports',
    fileNamePrefix='palawan_lulc_2024',
    region=aoi,
    scale=10,
    maxPixels=1e13,
    crs='EPSG:4326'
)

# Don't start automatically in notebook
print("✓ Export tasks configured")
print("\\nTo start export, run:")
print("  export_task.start()")
print("\\nThen check status at: https://code.earthengine.google.com/tasks")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Export Options

You can export:
- **Classification maps** (GeoTIFF)
- **Forest loss** (binary mask)
- **Hotspots** (intensity raster)
- **Statistics** (CSV via pandas)

For large exports, use Earth Engine Tasks instead of direct download.

---
"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### 🎉 Part D Complete!

You've successfully:
- ✅ Created 2020 baseline classification
- ✅ Detected forest loss (2020-2024)
- ✅ Identified deforestation hotspots
- ✅ Generated stakeholder report
- ✅ Prepared export tasks

---
"""))

# =============================================================================
# CONCLUSION
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# 🎓 Lab Complete!

## Summary of Achievements

In this 2-hour lab, you:

### Part A: Feature Engineering
- ✅ Created seasonal Sentinel-2 composites (dry/wet)
- ✅ Calculated spectral indices (NDVI, NDWI, NDBI, EVI)
- ✅ Extracted GLCM texture features
- ✅ Derived temporal phenology features
- ✅ Integrated topographic data (DEM)
- ✅ Built comprehensive 23-feature stack

### Part B: Classification
- ✅ Loaded training/validation data (80+40 polygons)
- ✅ Trained Random Forest classifier
- ✅ Applied 8-class land cover classification
- ✅ Achieved >85% accuracy (hopefully!)
- ✅ Analyzed feature importance
- ✅ Calculated area statistics

### Part C: Optimization
- ✅ Performed hyperparameter tuning
- ✅ Tested multiple tree counts
- ✅ Applied post-processing filters
- ✅ Improved classification quality

### Part D: NRM Applications
- ✅ Created 2020 baseline classification
- ✅ Detected forest loss (2020-2024)
- ✅ Identified deforestation hotspots
- ✅ Generated stakeholder-ready reports
- ✅ Prepared GIS exports

---

## Key Takeaways

1. **Multi-temporal analysis** significantly improves classification accuracy
2. **Texture features** help distinguish land covers with similar spectra
3. **Topography** provides valuable context for land use patterns
4. **Hyperparameter tuning** can boost accuracy by 2-5%
5. **Change detection** enables monitoring of conservation priorities

---

## Next Steps

### Session 3: Introduction to Deep Learning & CNNs

You're now ready to explore deep learning approaches that can:
- Learn features automatically (vs manual engineering)
- Handle complex spatial patterns
- Achieve higher accuracy on challenging classes
- Scale to very large areas

[Continue to Session 3 →](../../../course_site/day2/sessions/session3.qmd)

### Extended Exercises

Want more practice? Try:

1. **Expand to full Palawan** (entire province)
2. **Add more classes** (coconut vs rice, forest subtypes)
3. **Annual time series** (2017-2024 trends)
4. **Integration with field data**
5. **Automated monitoring** (monthly updates)

---

## Resources

### Code Templates
- [`glcm_template.py`](../templates/glcm_template.py)
- [`temporal_composite_template.py`](../templates/temporal_composite_template.py)
- [`change_detection_template.py`](../templates/change_detection_template.py)

### Documentation
- [GEE Classification Guide](https://developers.google.com/earth-engine/guides/classification)
- [Session 2 Overview](../../../course_site/day2/sessions/session2.qmd)
- [Troubleshooting Guide](../documentation/TROUBLESHOOTING.md)

### Training Data
- [Palawan Training Polygons](../../session1/data/palawan_training_polygons.geojson)
- [Class Definitions](../../session1/data/class_definitions.md)

---

## Questions or Issues?

- 📖 Review session documentation
- 💬 Ask instructor during office hours
- 🌐 Check GEE community forums
- 📧 Submit via course platform

---

## Congratulations! 🎉

You've mastered advanced EO classification techniques and are ready for deep learning!

**Session completed:** """ + """

---

*Developed for CopPhil Advanced Training Program*  
*EU-Philippines Copernicus Programme*
"""))

# Add all new cells to notebook
nb['cells'].extend(additional_cells)

# Write updated notebook
with open('session2_extended_lab_STUDENT.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✓ Session 2 Student Notebook Completed!")
print(f"  Total cells: {len(nb['cells'])}")
print(f"  Added sections: Parts C & D")
print(f"  File: session2_extended_lab_STUDENT.ipynb")
