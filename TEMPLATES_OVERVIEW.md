# Session 2 Templates - Overview

**Location:** `DAY_2/session2/templates/`  
**Purpose:** Reusable Python code for advanced Earth Observation analysis  
**Total:** 7 files (~88 KB)

---

## 📁 Template Files

### 1. **glcm_template.py** (7 KB)

**Purpose:** GLCM (Gray-Level Co-occurrence Matrix) texture analysis for land cover classification

**What it does:**
- Calculates texture features from satellite imagery
- Helps distinguish forest types, urban areas, and agricultural patterns
- Extracts features like contrast, entropy, correlation, variance

**Key Functions:**
```python
add_glcm_texture(image, bands=['B8'], radius=1)
# Adds all 13 GLCM texture features to an image

add_selected_glcm(image, bands=['B8'], features=['contrast', 'ent', 'corr'])
# Adds only specific texture features (faster)

compute_glcm_for_roi(image, geometry, bands=['B8'])
# Computes texture for specific area
```

**Use Case in Session 2:**
- Part A: Advanced Feature Engineering
- Distinguishing primary vs secondary forest by canopy texture
- Separating urban from bare soil by heterogeneity
- Improving classification accuracy by 5-10%

**Example Usage:**
```python
# Add texture to Sentinel-2 image
s2_with_texture = add_glcm_texture(s2_image, bands=['B8'], radius=1)

# Use selected features only (faster)
s2_texture = add_selected_glcm(s2_image, 
                                bands=['B8'], 
                                features=['contrast', 'ent'])
```

---

### 2. **glcm_template_enhanced.py** (17 KB)

**Purpose:** Enhanced GLCM with error handling, optimization, and best practices

**Additional Features:**
- Input validation and error checking
- Multiple optimization levels (fast/balanced/quality)
- Multi-band texture computation
- Automatic band availability checks
- Performance optimization tips

**Key Functions:**
```python
add_glcm_texture(image, bands=['B8'], radius=1, kernel=None)
# Enhanced with validation

optimize_glcm_computation(image, bands, window_size='balanced')
# Three optimization levels: 'fast', 'balanced', 'quality'

validate_sentinel2_bands(image)
# Checks if required bands are present
```

**Use Case:**
- Production-ready GLCM implementation
- Handling edge cases and errors
- Performance optimization for large areas
- Teaching best practices

---

### 3. **glcm_use_cases.py** (21 KB)

**Purpose:** Practical examples and use cases for GLCM in EO

**Contains:**
- Forest type classification examples
- Urban mapping use cases
- Agricultural pattern recognition
- Mangrove detection
- Complete workflows with explanations

**Example Use Cases:**
```python
classify_forest_types(s2_image, training_data)
# Uses texture to separate primary/secondary forest

detect_urban_patterns(s2_image, aoi)
# Uses texture heterogeneity for urban mapping

identify_agricultural_fields(s2_image, field_polygons)
# Texture-based field boundary detection
```

---

### 4. **temporal_composite_template.py** (9 KB)

**Purpose:** Multi-temporal compositing for seasonal analysis

**What it does:**
- Creates dry season composites (Jan-May)
- Creates wet season composites (Jun-Nov)
- Calculates phenological features
- Detects seasonal vegetation patterns

**Key Functions:**
```python
create_seasonal_composite(aoi, year, season='dry', cloud_threshold=20)
# Creates median composite for specified season

create_phenology_composites(aoi, year)
# Returns both dry and wet season composites

add_seasonal_features(dry_composite, wet_composite)
# Calculates NDVI difference, phenology metrics

calculate_ndvi_difference(dry_composite, wet_composite)
# Wet NDVI - Dry NDVI (identifies crops vs forest)
```

**Use Case in Session 2:**
- Part A: Temporal Features section
- Distinguishing rice (seasonal) from forest (evergreen)
- Identifying irrigated vs rainfed agriculture
- Capturing phenological cycles

**Example Usage:**
```python
# Create seasonal composites
composites = create_phenology_composites(palawan_aoi, 2024)
dry = composites['dry']
wet = composites['wet']

# Calculate features
ndvi_diff = calculate_ndvi_difference(dry, wet)
# Positive = seasonal crops, Near zero = evergreen forest
```

---

### 5. **change_detection_template.py** (12 KB)

**Purpose:** Detect and quantify land cover changes between time periods

**What it does:**
- Detects forest loss
- Creates change matrices
- Quantifies class transitions
- Identifies change hotspots
- Generates stakeholder reports

**Key Functions:**
```python
detect_forest_loss(classification_t1, classification_t2, forest_classes=[1, 2])
# Identifies pixels that were forest and are now deforested

create_change_matrix(classification_t1, classification_t2)
# Creates from-to transition matrix (e.g., forest→agriculture = code 14)

calculate_class_transitions(change_matrix, from_class, to_classes, aoi)
# Calculates area (hectares) for each transition type

identify_change_hotspots(change_image, aoi, threshold=100)
# Finds clusters of change >100 hectares

generate_change_report(change_matrix, aoi, class_names)
# Creates DataFrame with all transition statistics
```

**Use Case in Session 2:**
- Part D: NRM Applications
- 2020 vs 2024 comparison
- Deforestation hotspot detection
- Agricultural expansion tracking
- Protected area monitoring

**Example Usage:**
```python
# Detect forest loss
forest_loss_2020_2024 = detect_forest_loss(lc_2020, lc_2024, 
                                            forest_classes=[1, 2])

# Create change matrix
change_matrix = create_change_matrix(lc_2020, lc_2024)

# Get forest→agriculture transitions
transitions = calculate_class_transitions(change_matrix, 
                                          from_class=1,  # Forest
                                          to_classes=[4],  # Agriculture
                                          aoi=palawan)
# Returns: {'1→4': 1250.5} hectares
```

---

### 6. **test_glcm_template.py** (10 KB)

**Purpose:** Testing and validation script for GLCM functions

**Contains:**
- Unit tests for GLCM functions
- Validation of texture calculations
- Performance benchmarking
- Example test cases

**Use Case:**
- Quality assurance for GLCM implementations
- Teaching software testing practices
- Debugging GLCM calculations
- Performance optimization

---

### 7. **GLCM_QUICK_REFERENCE.md** (9 KB)

**Purpose:** Quick reference guide for GLCM texture analysis

**Contains:**
- **What is GLCM?** Explanation of texture analysis
- **When to use it:** Decision tree for when texture helps
- **How to compute:** Step-by-step GEE implementation
- **Common issues:** Troubleshooting guide
- **Performance tips:** Speed optimization strategies
- **Interpretation guide:** Understanding texture values
- **Use cases:** Specific EO applications

**Sections:**
1. GLCM Theory (simplified)
2. GEE Implementation Guide
3. Parameter Selection
   - Window size (3×3, 5×5, 7×7)
   - Which bands to use (NIR best for vegetation)
   - Which features to select
4. Common Errors and Solutions
5. Performance Optimization
6. Real-world Examples

---

## 🎯 Value for Session 2

### For Students:

**Reusable Code:**
- Copy-paste ready functions
- No need to code from scratch
- Tested and validated
- Clear documentation

**Learning Resource:**
- Understand how features work
- See best practices
- Learn error handling
- Professional code structure

**Time Saving:**
- Focus on concepts, not syntax
- Quick experimentation
- Modify for their own projects

### For Instructors:

**Teaching Tools:**
- Demonstrate advanced techniques
- Show production-ready code
- Explain optimization strategies
- Reference during live coding

**Consistency:**
- All students use same base code
- Easier to debug issues
- Standardized approach

---

## 📊 How Templates Align with Session 2

### Session 2 Structure:

**Part A: Advanced Feature Engineering (30 min)**
→ Uses: `glcm_template.py`, `temporal_composite_template.py`

**Part B: Palawan Classification (45 min)**
→ Uses: All features from templates

**Part C: Model Optimization (30 min)**
→ Uses: Enhanced versions for speed

**Part D: NRM Applications (15 min)**
→ Uses: `change_detection_template.py`

---

## 💡 Recommendation

### Should You Copy Templates to Production?

**YES - RECOMMENDED** ✅

**Reasons:**
1. **Referenced in Session 2 QMD** - Students expect them
2. **Saves significant time** - No need to code from scratch
3. **Best practices included** - Professional quality code
4. **Copy-paste ready** - Immediate use in notebooks
5. **Troubleshooting reference** - Quick solutions to common issues

**Where to Put Them:**
```bash
course_site/day2/templates/
├── glcm_template.py
├── temporal_composite_template.py
├── change_detection_template.py
└── GLCM_QUICK_REFERENCE.md
```

**Minimal Set (if space is limited):**
- `glcm_template.py` (essential)
- `temporal_composite_template.py` (important)
- `change_detection_template.py` (for Part D)
- `GLCM_QUICK_REFERENCE.md` (helpful guide)

**Impact if NOT included:**
- Students must code everything from scratch
- More debugging time during session
- Slower learning pace
- Less time for concepts

---

## 🚀 Quick Copy Command

If you decide to add templates:

```bash
cd /Users/dimitriskasampalis/Projects/Neuralio/ESAPhil

# Create templates folder
mkdir -p course_site/day2/templates

# Copy essential templates
cp DAY_2/session2/templates/glcm_template.py course_site/day2/templates/
cp DAY_2/session2/templates/temporal_composite_template.py course_site/day2/templates/
cp DAY_2/session2/templates/change_detection_template.py course_site/day2/templates/
cp DAY_2/session2/templates/GLCM_QUICK_REFERENCE.md course_site/day2/templates/

# Optional: Copy enhanced versions too
cp DAY_2/session2/templates/glcm_template_enhanced.py course_site/day2/templates/
cp DAY_2/session2/templates/glcm_use_cases.py course_site/day2/templates/
```

---

## 📋 Summary

**What Templates Contain:**
- ✅ Production-ready Python code for GEE
- ✅ GLCM texture analysis functions
- ✅ Multi-temporal compositing workflows
- ✅ Change detection algorithms
- ✅ Best practices and error handling
- ✅ Documentation and quick reference

**Total Value:** ~88 KB of reusable, tested code

**Recommendation:** **Copy to production folder** (medium priority)

**Impact:**
- **With templates:** Students focus on concepts, faster learning
- **Without templates:** More coding, slower pace, more debugging

---

**Status:** Optional but recommended for best learning experience  
**Time to copy:** 1 minute  
**Value:** High for student experience
