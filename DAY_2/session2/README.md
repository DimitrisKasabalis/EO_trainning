# Session 2: Land Cover Classification Lab (Palawan Case Study)

## Overview

**Duration:** 1.5-2 hours  
**Type:** Hands-on Lab  
**Prerequisites:** Session 1 completed

This session builds on Session 1 by implementing advanced Random Forest classification techniques for a real-world Philippine Natural Resource Management (NRM) scenario.

---

## Learning Objectives

By the end of this session, participants will be able to:

1. **Engineer advanced features** for Earth observation classification:
   - Texture features (GLCM)
   - Temporal features (multi-season composites)
   - Topographic features (DEM-derived)

2. **Implement optimized classification** workflows:
   - Hyperparameter tuning
   - Cross-validation
   - Class balancing

3. **Apply classification** to real NRM problems:
   - Deforestation detection
   - Change detection analysis
   - Protected area monitoring

4. **Generate stakeholder outputs**:
   - Area statistics
   - Classification maps
   - Change reports

---

## Session Structure

### Part A: Advanced Feature Engineering (30 minutes)
- GLCM texture features
- Multi-temporal Sentinel-2 composites
- SRTM topographic variables
- Feature stacking and selection

### Part B: Palawan Biosphere Reserve Case Study (45 minutes)
- 8-class land cover classification
- High-resolution mapping
- Accuracy assessment
- Area statistics

### Part C: Model Optimization (30 minutes)
- Hyperparameter tuning (grid search)
- K-fold cross-validation
- Class weight balancing
- Post-processing techniques

### Part D: NRM Applications (15 minutes)
- 2020 vs 2024 deforestation analysis
- Agricultural expansion mapping
- Protected area encroachment
- Report generation

---

## Study Area

**Location:** Palawan Biosphere Reserve, Philippines  
**Coordinates:** 9.5°N to 10.5°N, 118.5°E to 119.5°E  
**Area:** ~11,655 km²  
**UNESCO Status:** Biosphere Reserve (1990)

### Conservation Significance
- Last frontier of Philippine biodiversity
- Home to endemic species
- Critical carbon sink
- Under pressure from development

---

## Files

### Notebooks
- `session2_extended_lab_STUDENT.ipynb` - Student version with exercises
- `session2_extended_lab_INSTRUCTOR.ipynb` - Complete with solutions

### Data
- `palawan_biosphere_boundary.geojson` - Study area boundary
- Pre-prepared training data from Session 1
- Sentinel-2 composites (dry/wet season) - downloaded via GEE
- SRTM DEM - accessed via GEE

### Templates
- `glcm_template.py` - GLCM texture calculation
- `temporal_composite_template.py` - Multi-temporal processing
- `change_detection_template.py` - Change analysis workflow
- `export_template.py` - Export utilities

### Documentation
- `HYPERPARAMETER_TUNING.md` - Tuning results and recommendations
- `TROUBLESHOOTING.md` - Common issues and solutions
- `NRM_WORKFLOWS.md` - Application-specific workflows

---

## Technical Requirements

### Software
- Python 3.8+
- Google Earth Engine account
- geemap, ee, scikit-learn
- rasterio, geopandas (optional)

### Computational
- Recommended: 8GB RAM
- GEE quota: Standard tier sufficient
- Runtime: 90-120 minutes

### Data
- Sentinel-2 L2A (2020, 2024)
- SRTM DEM 30m
- Training polygons (Session 1)

---

## Installation

```bash
# Install required packages
pip install earthengine-api geemap scikit-learn pandas numpy matplotlib seaborn

# Authenticate Google Earth Engine
earthengine authenticate
```

---

## Quick Start

1. **Open notebook** in Jupyter Lab or Google Colab
2. **Run setup cells** to import libraries and authenticate GEE
3. **Load study area** and training data
4. **Follow guided exercises** through each section
5. **Complete TODO items** marked in student version

---

## Key Concepts

### GLCM Texture Features
- **Contrast:** Local variation in grayscale
- **Correlation:** Pixel pair correlation
- **Entropy:** Randomness/complexity
- **Homogeneity:** Closeness of distribution

### Multi-temporal Composites
- **Dry Season** (Jan-Apr): Best for forest mapping
- **Wet Season** (Jul-Sep): Agricultural phenology
- **Change Metrics:** Inter-season differences

### Topographic Variables
- **Elevation:** Height above sea level
- **Slope:** Rate of elevation change
- **Aspect:** Slope direction
- **Hillshade:** Illumination simulation

---

## Expected Outcomes

### Classification Results
- **Overall Accuracy:** >85%
- **Kappa Coefficient:** >0.80
- **Per-class Accuracy:** >80% for most classes

### Deliverables
- High-resolution land cover map (10m)
- Accuracy assessment report
- Change detection maps (2020-2024)
- Area statistics by class
- Exported GeoTIFF for GIS

---

## Philippine Context

### Palawan Conservation Issues
- **Deforestation:** Mining, agriculture expansion
- **Mangrove Loss:** Coastal development
- **Urban Sprawl:** Puerto Princesa growth
- **Climate Change:** Sea level rise impacts

### Application to Other Regions
- Methodology transferable to:
  - Mindanao forest monitoring
  - Visayas coastal mapping
  - Luzon agricultural assessment

---

## Common Challenges

### Technical
- GEE memory limits → Use smaller tiles
- Timeout errors → Reduce temporal range
- Missing data → Handle cloud masking carefully

### Classification
- Class confusion → Add texture/temporal features
- Low accuracy → Increase training samples
- Edge effects → Apply spatial filters

---

## Additional Resources

### Documentation
- [GEE Classification Guide](https://developers.google.com/earth-engine/guides/classification)
- [GLCM in GEE](https://developers.google.com/earth-engine/apidocs/ee-image-glcmtexture)
- [Random Forest Tuning](https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters)

### Datasets
- [ESRI 2020 Land Cover](https://livingatlas.arcgis.com/landcover/)
- [Copernicus Global Land Cover](https://land.copernicus.eu/global/)
- [Philippine GIS Data Portal](https://data.gov.ph)

### Papers
- Karra et al. (2021). Global LULC with Sentinel-2 and deep learning
- Phiri et al. (2020). Sentinel-2 LULC classification review
- Talukdar et al. (2020). Land-use change detection

---

## Assessment

### Formative
- Exercise completion (TODO markers)
- Classification accuracy achieved
- Feature engineering attempts
- Troubleshooting approaches

### Summative
- Final classification map quality
- Accuracy metrics (>80% target)
- Change detection analysis
- Written interpretation

---

## Instructor Notes

### Timing Tips
- Part A: Allow extra time for GLCM (computationally intensive)
- Part B: Most students finish in 40-50 minutes
- Part C: Can be shortened if time constrained
- Part D: Good for early finishers

### Common Questions
- "Why is GLCM slow?" → Explain computational complexity
- "Can I use my own area?" → Yes, provide coordinates
- "What if accuracy is low?" → Review training data quality
- "How to export large areas?" → Demonstrate tiling

### Teaching Strategies
- **Demo first:** Show complete workflow before exercises
- **Pair programming:** Partner students for troubleshooting
- **Live coding:** Walk through one feature type together
- **Gallery walk:** Share different parameter combinations

---

## Next Steps

After completing Session 2:

1. **Session 3:** Introduction to Deep Learning and CNNs
   - Transition from traditional ML to deep learning
   - CNN architecture fundamentals
   - Transfer learning concepts

2. **Extended Projects:**
   - Apply to different Philippine regions
   - Add more land cover classes
   - Integrate with field data
   - Time series analysis

3. **Real-world Application:**
   - Partner with local agencies
   - Support conservation planning
   - Contribute to national monitoring

---

## Credits

**Developed by:** CopPhil Training Team  
**Contributors:** EU-Philippines Copernicus Programme  
**Version:** 1.0 (October 2025)

**License:** CC-BY-4.0

---

## Support

For questions or issues:
- Review troubleshooting guide
- Check GEE community forums
- Contact instructor during office hours
- Submit issues via course platform

---

**Ready to become an expert in Earth observation for conservation! 🌍🌳**
