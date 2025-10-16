# Session 4: Model Comparison Guide
**Random Forest vs CNN vs Transfer Learning vs U-Net**

---

## Overview

This guide helps you choose the right AI/ML approach for your Earth Observation task. We compare all methods covered in Day 2.

---

## Quick Decision Tree

```
START: What's your goal?
│
├─ Classify entire image/patch?
│  ├─ Limited data (<1000 images)? → Random Forest
│  ├─ Moderate data (1000-10K)? → CNN from scratch
│  └─ Want best accuracy? → Transfer Learning (ResNet50)
│
└─ Delineate boundaries/pixels?
   └─ Need precise shapes? → U-Net Segmentation
```

---

## Detailed Comparison

### 1. Random Forest (Session 1-2)

**What it does:** Image-level classification using handcrafted features

**Strengths:**
- ✅ Fast training (minutes)
- ✅ Works with small datasets (100-1000 samples)
- ✅ Interpretable (feature importance)
- ✅ No GPU needed
- ✅ Easy to tune

**Weaknesses:**
- ❌ Manual feature engineering required
- ❌ Limited spatial context (3×3 or 5×5 windows)
- ❌ Lower accuracy than CNNs (typically 85-90%)
- ❌ Doesn't learn hierarchical features

**Best for:**
- Quick prototyping
- Limited computational resources
- Small datasets
- When interpretability is critical

**Philippine Use Cases:**
- Initial land cover mapping
- Quick forest/non-forest classification
- Agricultural vs urban discrimination

---

### 2. CNN from Scratch (Session 4A)

**What it does:** Learns spatial features automatically, classifies images

**Strengths:**
- ✅ Automatic feature learning
- ✅ Captures spatial patterns
- ✅ Better accuracy (90-93%)
- ✅ Handles complex patterns
- ✅ No manual feature engineering

**Weaknesses:**
- ❌ Requires more data (1000-10K images)
- ❌ Slower training (15-30 min on GPU)
- ❌ Needs GPU for practical training
- ❌ More hyperparameters to tune
- ❌ Less interpretable

**Best for:**
- Moderate-sized datasets
- When accuracy improvement justifies effort
- Tasks with complex spatial patterns
- Learning about deep learning

**Philippine Use Cases:**
- Detailed land use classification
- Rice growth stage classification
- Urban density mapping

---

### 3. Transfer Learning (Session 4B)

**What it does:** Fine-tunes pre-trained model (ResNet50) on your EO data

**Strengths:**
- ✅ **Best accuracy** (93-97%)
- ✅ Works with less data than CNN from scratch
- ✅ Faster convergence (10-20 min)
- ✅ Leverages ImageNet knowledge
- ✅ State-of-art performance

**Weaknesses:**
- ❌ Still needs moderate dataset (500-5000 images)
- ❌ Requires GPU
- ❌ Larger models (more storage)
- ❌ Less control over architecture

**Best for:**
- When accuracy is critical
- Moderate datasets
- Production systems
- Benchmark comparisons

**Philippine Use Cases:**
- High-accuracy mangrove mapping
- Precision crop type classification
- Critical infrastructure monitoring
- Disaster damage assessment

---

### 4. U-Net Segmentation (Session 4C)

**What it does:** Pixel-level classification, delineates boundaries

**Strengths:**
- ✅ **Precise boundaries** (pixel-level)
- ✅ Excellent for delineation tasks
- ✅ Preserves spatial details (skip connections)
- ✅ Works with 100-1000 labeled patches
- ✅ Outputs shapefile-ready masks

**Weaknesses:**
- ❌ Requires pixel-level annotations (laborious)
- ❌ Slower training (30-60 min)
- ❌ More complex to implement
- ❌ Higher computational cost

**Best for:**
- Boundary delineation
- Field/parcel mapping
- Building footprints
- Infrastructure extraction

**Philippine Use Cases:**
- Mangrove boundary mapping
- Rice field delineation
- Building footprint extraction
- Road network mapping
- Flood extent precise mapping

---

## Performance Comparison (EuroSAT Benchmark)

| Method | Accuracy | Training Time | Data Needed | GPU | Complexity |
|--------|----------|---------------|-------------|-----|------------|
| **Random Forest** | 85-90% | 2-5 min | 100-1K | No | Low |
| **CNN (Scratch)** | 90-93% | 15-30 min | 1K-10K | Yes | Medium |
| **Transfer Learning** | 93-97% | 10-20 min | 500-5K | Yes | Medium |
| **U-Net** | N/A (IoU: 0.75-0.90) | 30-60 min | 100-1K masks | Yes | High |

---

## When to Use Each Method

### Random Forest
✅ **Use when:**
- Dataset < 1000 samples
- Need quick results
- No GPU available
- Interpretability important
- Exploring feasibility

❌ **Avoid when:**
- Need >90% accuracy
- Complex spatial patterns
- Large dataset available

### CNN from Scratch
✅ **Use when:**
- 1000-10K labeled images
- Learning deep learning
- Custom architecture needed
- Moderate accuracy sufficient

❌ **Avoid when:**
- <1000 images (use RF or Transfer Learning)
- Need state-of-art accuracy (use Transfer Learning)
- Limited GPU access

### Transfer Learning
✅ **Use when:**
- Production deployment
- Need best accuracy
- 500-5000 labeled images
- Critical applications

❌ **Avoid when:**
- Very limited data (<500)
- Very different domain (non-RGB)
- Explainability critical

### U-Net Segmentation
✅ **Use when:**
- Need precise boundaries
- Boundary mapping critical
- Can afford pixel annotation
- Segmentation task

❌ **Avoid when:**
- Only need image labels (use classification)
- Can't create pixel masks
- Bounding boxes sufficient (use object detection)

---

## Philippine-Specific Recommendations

### Mangrove Monitoring
**Recommended:** Transfer Learning + U-Net
- Transfer Learning: Identify mangrove areas
- U-Net: Precise boundary delineation
- Rationale: High accuracy + precise boundaries critical

### Rice Agriculture
**Recommended:** Random Forest or CNN
- Random Forest: Quick seasonal classification
- CNN: Growth stage monitoring
- Rationale: Field-level accuracy sufficient

### Informal Settlements
**Recommended:** Transfer Learning
- ResNet50 fine-tuned on urban patterns
- Rationale: Complex patterns, accuracy critical

### Flood Mapping
**Recommended:** U-Net (SAR data)
- Pixel-level inundation extent
- Rationale: Precise boundaries for response

### Forest Change Detection
**Recommended:** Random Forest → CNN (if needed)
- Start with RF for quick assessment
- Upgrade to CNN if accuracy insufficient
- Rationale: Iterative improvement

---

## Cost-Benefit Analysis

### Computational Cost

| Method | Training (GPU hours) | Inference (ms) | Storage (MB) |
|--------|---------------------|----------------|--------------|
| **Random Forest** | 0 (CPU only) | 1-5 | 10-100 |
| **CNN** | 0.25-0.5 | 10-50 | 50-200 |
| **Transfer Learning** | 0.17-0.33 | 20-100 | 100-500 |
| **U-Net** | 0.5-1.0 | 50-200 | 200-500 |

### Data Annotation Cost

| Method | Labels Needed | Time per Sample | Total Time (500 samples) |
|--------|---------------|----------------|--------------------------|
| **RF/CNN/TL** | Image label | 10-30 sec | 1-3 hours |
| **U-Net** | Pixel mask | 5-15 min | 40-125 hours |

**Key Insight:** U-Net requires 40-100× more annotation time!

---

## Hybrid Approaches

### Approach 1: RF → CNN Pipeline
1. Use RF for quick initial classification
2. Identify uncertain areas
3. Train CNN on uncertain regions only
4. Combine predictions

**Benefits:** Best of both worlds, efficient

### Approach 2: Classification → Segmentation
1. CNN identifies areas of interest
2. U-Net segments within those areas
3. Reduces U-Net input size

**Benefits:** Focused computation, faster

### Approach 3: Ensemble
1. Train RF, CNN, Transfer Learning
2. Average predictions (soft voting)
3. Or use majority vote (hard voting)

**Benefits:** Robust, reduces errors

---

## Practical Guidelines

### Starting a New Project

**Week 1: Baseline**
- Random Forest with basic features
- Establishes performance floor (85-90%)
- Fast iteration

**Week 2-3: Deep Learning (if needed)**
- Try CNN from scratch
- Try Transfer Learning
- Compare improvements

**Week 4+: Optimization**
- Tune best performer
- Consider U-Net if boundaries needed
- Deploy

### Model Selection Checklist

□ **Data size:** How many labeled samples?  
□ **Annotation type:** Image labels or pixel masks?  
□ **Accuracy requirement:** Is 85% sufficient or need 95%?  
□ **Computational resources:** GPU available?  
□ **Timeline:** Days or months?  
□ **Interpretability:** Need to explain predictions?  
□ **Deployment:** Real-time or batch?  

---

## Case Study: Palawan Forest Monitoring

**Scenario:** Map forest cover in Palawan for conservation

**Options Evaluated:**

1. **Random Forest**
   - Accuracy: 87%
   - Time: 1 week
   - Cost: Low
   - Decision: Good baseline

2. **CNN from Scratch**
   - Accuracy: 92%
   - Time: 2 weeks
   - Cost: Medium
   - Decision: Significant improvement

3. **Transfer Learning (ResNet50)**
   - Accuracy: 95%
   - Time: 1.5 weeks
   - Cost: Medium
   - Decision: **SELECTED** (best accuracy)

4. **U-Net Segmentation**
   - IoU: 0.85
   - Time: 4 weeks (annotation)
   - Cost: High
   - Decision: Future phase (boundary refinement)

**Final Strategy:**
- Use Transfer Learning for extent mapping
- Use U-Net for high-priority areas needing precise boundaries
- Annual updates with new imagery

---

## Summary Table

| Criterion | Random Forest | CNN | Transfer Learning | U-Net |
|-----------|--------------|-----|-------------------|-------|
| **Accuracy** | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ (IoU) |
| **Speed** | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| **Data Efficiency** | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| **Ease of Use** | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| **Interpretability** | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| **Scalability** | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |

---

## Recommendations by Experience Level

### Beginner
- Start with **Random Forest**
- Learn feature engineering
- Understand ML basics
- Move to CNN when comfortable

### Intermediate
- Try **CNN from scratch**
- Experiment with architectures
- Compare with **Transfer Learning**
- Understand tradeoffs

### Advanced
- Use **Transfer Learning** for production
- Implement **U-Net** for segmentation
- Design hybrid approaches
- Optimize for deployment

---

**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 4
