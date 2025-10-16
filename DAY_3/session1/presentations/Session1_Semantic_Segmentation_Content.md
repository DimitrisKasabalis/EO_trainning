# DAY 3 - Session 1: Semantic Segmentation with U-Net for Earth Observation
## Presentation Content (1.5 hours)

---

## SLIDE 1: Title Slide
**Title:** Semantic Segmentation with U-Net for Earth Observation
**Subtitle:** Advanced Deep Learning for Pixel-Level Analysis
**Day 3, Session 1**

---

## SLIDE 2: Learning Objectives
By the end of this session, you will be able to:
- ✓ Define semantic segmentation and distinguish it from classification and detection
- ✓ Understand the U-Net architecture (encoder-decoder with skip connections)
- ✓ Explain loss functions for segmentation (Cross-Entropy, Dice, IoU)
- ✓ Identify Earth Observation applications for semantic segmentation

---

## MODULE 1: CONCEPT OF SEMANTIC SEGMENTATION (20 mins)

---

## SLIDE 3: What is Semantic Segmentation?
**Definition:**
> Semantic segmentation is the task of classifying **each pixel** in an image into a class label, producing a **pixel-wise labeled map**.

**Key Characteristics:**
- Every pixel gets a label
- Creates precise boundaries
- Dense prediction (not sparse)
- Pixel-level understanding

**Visual:** Show satellite image with overlay mask coloring different land covers

---

## SLIDE 4: Computer Vision Task Hierarchy
**Three Main Tasks:**

| Task | Question | Output | Granularity |
|------|----------|--------|-------------|
| **Classification** | "What's in this image?" | Single label | Image-level |
| **Object Detection** | "Where are the objects?" | Bounding boxes | Object-level |
| **Segmentation** | "Which pixels belong to what?" | Pixel masks | Pixel-level |

**Visual:** Three-panel comparison with same satellite image

---

## SLIDE 5: Image Classification Example
**Task:** Is this image "urban" or "agricultural"?

**Input:** Sentinel-2 image (256x256 pixels)
**Output:** One label → "Urban" (confidence: 0.92)

**Characteristics:**
- Global decision
- No spatial information
- Fast to compute
- Coarse understanding

**Visual:** Satellite image with single label badge

---

## SLIDE 6: Object Detection Example
**Task:** Find all buildings in this image

**Input:** Sentinel-2 image
**Output:** List of bounding boxes:
- Box 1: [x:120, y:50, w:40, h:30] → "Building" (0.95)
- Box 2: [x:200, y:100, w:35, h:28] → "Building" (0.88)
- ...

**Characteristics:**
- Locates individual instances
- Rectangular approximation
- Counts objects
- Moderate detail

**Visual:** Satellite image with colored bounding boxes

---

## SLIDE 7: Semantic Segmentation Example
**Task:** Label every pixel with land cover type

**Input:** Sentinel-2 image (256x256 pixels)
**Output:** Mask (256x256) where each pixel has class:
- Pixel [0,0] → "Water" (class 1)
- Pixel [0,1] → "Water" (class 1)
- Pixel [100,50] → "Forest" (class 2)
- Pixel [200,150] → "Urban" (class 3)

**Characteristics:**
- Precise boundaries
- Pixel-perfect accuracy
- Rich spatial detail
- Computationally intensive

**Visual:** Satellite image with colored segmentation mask overlay

---

## SLIDE 8: Why Semantic Segmentation for EO?
**Advantages:**
1. **Precise Delineation** - Exact boundaries of features (flood extent, forest edges)
2. **Quantitative Analysis** - Accurate area calculations
3. **Change Detection** - Pixel-level comparison over time
4. **Thematic Mapping** - Detailed land cover/land use maps
5. **Decision Support** - Fine-grained information for planning

**Example Use Cases:**
- Mapping exact flood inundation extent
- Delineating agricultural field boundaries
- Extracting building footprints
- Identifying burn scars from wildfires

---

## SLIDE 9: Classification vs Segmentation: Side-by-Side
**Scenario:** Typhoon flood assessment

**Classification Approach:**
- Input: 100 image tiles of affected region
- Output: 73 tiles labeled "flooded", 27 tiles "not flooded"
- **Limitation:** We know flooding occurred but not WHERE exactly

**Segmentation Approach:**
- Input: Same 100 image tiles
- Output: Pixel-level flood masks for each tile
- **Advantage:** Exact flood boundaries → precise area calculation → targeted relief

**Visual:** Split screen comparison

---

## SLIDE 10: Think-Pair-Share Exercise
**Question:** For each scenario, which task is most appropriate?

1. Estimating total urban area in Metro Manila
2. Counting individual vehicles in a parking lot
3. Mapping the exact boundary of Laguna de Bay
4. Determining if an image contains informal settlements

**Take 2 minutes to discuss with your neighbor**

**Answers:**
1. Segmentation (need pixel-level urban extent)
2. Detection (need to count individual instances)
3. Segmentation (need precise water boundary)
4. Classification (yes/no presence)

---

## MODULE 2: U-NET ARCHITECTURE (30 mins)

---

## SLIDE 11: Introduction to U-Net
**Origins:**
- Developed by Ronneberger et al. (2015) for biomedical image segmentation
- Won multiple competitions
- Now widely adopted in Earth Observation

**Why "U-Net"?**
- Architecture shape resembles the letter "U"
- Symmetric encoder-decoder structure

**Key Innovation:**
- Skip connections preserve spatial information

**Visual:** U-Net original paper diagram

---

## SLIDE 12: U-Net Overview Diagram
**Architecture Components:**

```
Input Image (H x W x C)
        ↓
    [ENCODER]
   (Contracting Path)
   - Convolutions
   - Pooling
   - Spatial dimension ↓
   - Feature depth ↑
        ↓
    [BOTTLENECK]
   (Most compressed)
        ↓
    [DECODER]
   (Expansive Path)
   - Upsampling
   - Convolutions
   - Spatial dimension ↑
   - Feature depth ↓
        ↓
Output Mask (H x W x Classes)
```

**Visual:** Full U-Net architecture diagram with annotations

---

## SLIDE 13: Encoder (Contracting Path)
**Purpose:** Extract hierarchical features at multiple scales

**Operations:**
1. **Convolution blocks:**
   - 2x Conv layers (3x3 kernels)
   - ReLU activation
   - (Optional) Batch normalization

2. **Downsampling:**
   - Max pooling (2x2)
   - Spatial dimensions halve
   - Feature channels double

**Example Progression:**
```
Input:     256x256x3  (RGB image)
Block 1:   256x256x64  (after convs)
Pool 1:    128x128x64  (after pooling)
Block 2:   128x128x128 (after convs)
Pool 2:    64x64x128   (after pooling)
Block 3:   64x64x256   (after convs)
Pool 3:    32x32x256   (after pooling)
...
```

**Visual:** Encoder pathway with dimension annotations

---

## SLIDE 14: Recap: Day 2 CNN Concepts in Encoder
**Convolution:**
- Learned filters detect patterns (edges, textures)
- Same padding preserves spatial dimensions
- Creates feature maps

**Pooling:**
- Reduces spatial resolution
- Provides translation invariance
- Concentrates information

**Activation (ReLU):**
- Introduces non-linearity
- Enables learning complex patterns

**Why multiple scales?**
- Early layers: fine details (edges, textures)
- Deep layers: semantic meaning (buildings, water bodies)

---

## SLIDE 15: Bottleneck Layer
**Location:** Bottom of the "U" (most compressed)

**Characteristics:**
- Smallest spatial dimensions (e.g., 16x16)
- Largest number of feature channels (e.g., 1024)
- Maximum abstraction / minimum spatial detail

**Role:**
- Compressed representation of entire image
- Captures global context
- "What" is in the image (content)

**Example:**
```
Input:      256x256x3
After encoder: 16x16x1024
Information: Knows there's water, buildings, vegetation
Lost detail: Exact pixel locations
```

**Visual:** Highlight bottleneck in U-Net diagram

---

## SLIDE 16: Decoder (Expansive Path)
**Purpose:** Reconstruct spatial resolution for pixel-wise predictions

**Operations:**
1. **Upsampling:**
   - Transpose convolution (learnable) OR
   - Bilinear upsampling + convolution
   - Doubles spatial dimensions
   - Halves feature channels

2. **Convolution blocks:**
   - 2x Conv layers (3x3 kernels)
   - ReLU activation
   - Refine predictions

**Example Progression:**
```
Bottleneck:  16x16x1024
Upsample 1:  32x32x512
Block 1:     32x32x512
Upsample 2:  64x64x256
Block 2:     64x64x256
...
Final:       256x256x3 (3 classes)
```

**Visual:** Decoder pathway with dimension annotations

---

## SLIDE 17: Upsampling Methods
**Method 1: Transpose Convolution (Deconvolution)**
- Learned upsampling
- Trainable parameters
- Can introduce checkerboard artifacts

**Method 2: Upsampling + Convolution**
- Fixed upsampling (bilinear/nearest)
- Followed by regular convolution
- Smoother results

**Visualization:**
- Low-res feature map (32x32)
- → Transpose Conv →
- High-res feature map (64x64)

**Common in U-Net:** Transpose convolution with 2x2 kernels, stride 2

---

## SLIDE 18: Skip Connections - The Key Innovation
**Problem without skip connections:**
- Information lost during pooling
- Decoder must reconstruct from coarse features
- Blurry boundaries

**Solution: Skip Connections**
- Copy feature maps from encoder to decoder
- Concatenate at matching spatial resolutions
- Provide fine-grained spatial details

**Visual:** U-Net diagram highlighting horizontal skip connection arrows

---

## SLIDE 19: How Skip Connections Work
**Step-by-step:**

1. **Encoder produces feature map** at resolution 128x128x64
2. **Feature map copied** and saved
3. **Encoder continues** downsampling
4. **Decoder upsamples** back to 128x128x32
5. **Concatenation:** Combine decoder features (128x128x32) with encoder features (128x128x64)
6. **Result:** 128x128x96 feature map with both:
   - Context from decoder (what)
   - Detail from encoder (where)

**Benefit:** Best of both worlds - semantic understanding + precise localization

---

## SLIDE 20: Why Skip Connections Matter for EO
**Earth Observation Challenges:**
- Need precise boundaries (coastlines, field edges, flood extent)
- Small objects (individual buildings, narrow rivers)
- High spatial accuracy required

**Skip Connections Provide:**
- **Spatial precision:** Exact pixel locations preserved
- **Boundary sharpness:** Crisp edges in output masks
- **Small object detection:** Fine details not lost
- **Gradient flow:** Helps training deep networks

**Example:**
- Without skips: Flood boundary ±10 pixels
- With skips: Flood boundary ±1-2 pixels

---

## SLIDE 21: U-Net Architecture Summary
**Complete Flow:**

```
Input (256x256x3)
    ↓ [Conv+ReLU] → Skip 1
    ↓ [Pool ↓]
    ↓ [Conv+ReLU] → Skip 2
    ↓ [Pool ↓]
    ↓ [Conv+ReLU] → Skip 3
    ↓ [Pool ↓]
    ↓ [Conv+ReLU] → Skip 4
    ↓ [Pool ↓]
  Bottleneck [Conv+ReLU]
    ↓ [Upsample ↑]
    ↓ [Concat Skip 4]
    ↓ [Conv+ReLU]
    ↓ [Upsample ↑]
    ↓ [Concat Skip 3]
    ↓ [Conv+ReLU]
    ... (continue upsampling)
Output (256x256x num_classes)
```

**Visual:** Complete annotated U-Net diagram

---

## SLIDE 22: Padding in U-Net
**Recall from Day 2:** Padding preserves spatial dimensions

**In U-Net:**
- **"Same" padding** commonly used
- Ensures encoder and decoder feature maps align for concatenation
- Without proper padding: dimension mismatch errors

**Example:**
- Encoder: 128x128 feature map
- Decoder after upsample: 128x128
- Concatenation: ✓ Dimensions match

**Alternative (original U-Net):**
- "Valid" padding (no padding)
- Feature maps shrink
- Must crop encoder features to match decoder
- More complex implementation

**Modern Practice:** Use same padding for simplicity

---

## MODULE 3: APPLICATIONS IN EARTH OBSERVATION (15 mins)

---

## SLIDE 23: U-Net in Earth Observation - Overview
**Why U-Net is Popular in EO:**
- ✓ Accurate with limited training data
- ✓ Preserves fine spatial details
- ✓ Fast inference on large images
- ✓ Flexible architecture (easy to modify)
- ✓ Proven track record

**Common Applications:**
1. Flood mapping
2. Land cover classification
3. Road extraction
4. Building segmentation
5. Vegetation monitoring

---

## SLIDE 24: Application 1 - Flood Mapping
**Use Case:** Disaster response and damage assessment

**Data Sources:**
- Sentinel-1 SAR (cloud-penetrating)
- Sentinel-2 optical (high resolution)

**Task:**
- Binary segmentation: Flooded vs Non-flooded
- Input: SAR backscatter or optical RGB
- Output: Flood extent mask

**Benefits:**
- Rapid mapping (hours after event)
- Precise extent calculation
- Time-series monitoring (flood evolution)
- Integration with GIS for planning

**Real Example:** Typhoon Ulysses (2020) - Central Luzon floods mapped using U-Net on Sentinel-1

**Visual:** Before/after flood images with U-Net segmentation overlay

---

## SLIDE 25: Application 2 - Land Cover Mapping
**Use Case:** Environmental monitoring, urban planning

**Data Sources:**
- Sentinel-2 multispectral (10m)
- Landsat 8/9 (30m)
- High-res commercial imagery

**Task:**
- Multi-class segmentation: Water, Forest, Urban, Agriculture, Barren
- Input: Multi-spectral bands (RGB + NIR + SWIR)
- Output: Land cover classification map

**Benefits:**
- Detailed thematic maps
- Change detection over time
- Biodiversity assessments
- Carbon stock estimation

**Visual:** Satellite image with color-coded land cover segmentation

---

## SLIDE 26: Application 3 - Road and Infrastructure Extraction
**Use Case:** Map updating, transportation planning

**Data Sources:**
- High-resolution aerial imagery
- Sentinel-2 (10m) for major roads
- SAR for all-weather mapping

**Task:**
- Binary segmentation: Road vs Background
- Input: RGB or SAR intensity
- Output: Road network mask

**Challenges:**
- Thin linear features
- Occlusion (trees, shadows)
- Complex backgrounds

**U-Net Advantages:**
- Skip connections preserve road continuity
- Learns to follow linear patterns

**Visual:** Aerial image with extracted road network overlay

---

## SLIDE 27: Application 4 - Building Footprint Delineation
**Use Case:** Urban mapping, population estimation, disaster risk

**Data Sources:**
- Very high-resolution imagery (<1m)
- Sentinel-2 for large structures

**Task:**
- Binary/multi-class: Building vs Background (or building types)
- Input: RGB/NIR imagery
- Output: Building footprint polygons

**Benefits:**
- Automated mapping at scale
- Informal settlement detection
- Damage assessment (pre/post disaster)
- 3D city model generation (with height data)

**Variants:**
- Residual U-Net
- Attention U-Net
- For complex dense urban scenes

**Visual:** Urban scene with building footprints outlined

---

## SLIDE 28: Application 5 - Vegetation and Crop Monitoring
**Use Case:** Agriculture, forestry, ecosystem health

**Data Sources:**
- Sentinel-2 multispectral
- PlanetScope (3m daily)
- UAV imagery for field-scale

**Task:**
- Multi-class: Crop types (rice, corn, sugarcane, etc.)
- Or binary: Vegetation vs Non-vegetation
- Input: Multi-temporal + multi-spectral
- Output: Crop mask or crop type map

**Benefits:**
- Yield prediction
- Irrigation monitoring
- Disease detection
- Deforestation tracking

**Visual:** Agricultural landscape with crop type segmentation

---

## SLIDE 29: U-Net Success Factors in EO
**Why does U-Net work so well for satellite imagery?**

1. **Data Efficiency:**
   - Performs well with 100s-1000s of training samples
   - Data augmentation helps (rotations, flips relevant for nadir view)

2. **Spatial Precision:**
   - Skip connections critical for boundary accuracy
   - Important for legal boundaries, property lines, hazard zones

3. **Multi-Scale Learning:**
   - Encoder captures both local textures and global context
   - Essential for varied EO features (small boats to large lakes)

4. **Transfer Learning:**
   - Pre-trained encoders (ImageNet) boost performance
   - Domain adaptation from natural images to satellite

5. **Computational Feasibility:**
   - Efficient architecture
   - Can process 256x256 or 512x512 patches on standard GPUs

---

## MODULE 4: LOSS FUNCTIONS FOR SEGMENTATION (25 mins)

---

## SLIDE 30: Why Loss Functions Matter
**Training Goal:** Adjust network weights to minimize prediction error

**Loss Function:** Mathematical measure of difference between:
- Predicted mask (model output)
- Ground truth mask (annotated)

**For Segmentation:**
- Not just one number (like classification)
- Compare pixel-by-pixel across entire image
- Different losses emphasize different aspects

**Choice of Loss Function:**
- Affects what the model learns to optimize
- Critical for imbalanced datasets (common in EO)
- Can drastically change results

---

## SLIDE 31: Challenge: Class Imbalance in EO
**Common Scenario:**
- Flood mapping: 95% non-flooded, 5% flooded pixels
- Ship detection: 99.5% water/land, 0.5% ships
- Building segmentation: 80% background, 20% buildings

**Problem with Simple Accuracy:**
```python
Model predicts: ALL pixels = "non-flooded"
Accuracy: 95% ✓ (looks great!)
But: Completely useless - missed all floods!
```

**Need Loss Functions That:**
- Handle class imbalance
- Focus on minority class
- Reward overlap, not just pixel-wise accuracy

---

## SLIDE 32: Loss Function 1 - Pixel-wise Cross-Entropy
**How it Works:**
- Treat each pixel as independent classification
- Compare predicted class probability to true class
- Average loss across all pixels

**Formula (simplified):**
```
CE = -Σ(y_true * log(y_pred))
```

**For Segmentation:**
```
Total Loss = Average CE across all pixels
```

**Advantages:**
- ✓ Standard, well-understood
- ✓ Strong gradients for learning
- ✓ Works with multi-class (softmax)

**Disadvantages:**
- ✗ Dominated by majority class
- ✗ Doesn't directly optimize overlap
- ✗ Can ignore minority classes

---

## SLIDE 33: Weighted Cross-Entropy
**Solution to Imbalance:** Assign higher weight to minority class

**Formula:**
```
Weighted CE = -Σ(w_class * y_true * log(y_pred))
```

**Example:**
- 95% background → weight = 1.0
- 5% flood → weight = 19.0 (inverse frequency)

**Effect:**
- Model pays 19x more attention to flood pixels
- Penalized heavily for missing flood pixels

**Implementation:**
```python
# TensorFlow/Keras
loss = tf.keras.losses.CategoricalCrossentropy(
    class_weight={0: 1.0, 1: 19.0}
)
```

**Advantage:** Simple fix for imbalance
**Limitation:** Still pixel-wise, not region-based

---

## SLIDE 34: Loss Function 2 - Dice Loss (F1 Score)
**Concept:** Measure overlap between prediction and ground truth

**Formula:**
```
Dice = 2 * |Prediction ∩ Truth| / (|Prediction| + |Truth|)
Dice Loss = 1 - Dice
```

**Interpretation:**
- Dice = 1.0: Perfect overlap
- Dice = 0.5: 50% overlap
- Dice = 0.0: No overlap

**Visual:** Venn diagram showing intersection and union

**Why for Imbalanced Data?**
- Focuses on the object (minority class)
- Treats foreground and background asymmetrically
- One correctly segmented small flood patch = significant Dice increase

---

## SLIDE 35: Dice Loss - Properties
**Advantages:**
- ✓ Inherently handles class imbalance
- ✓ Directly optimizes overlap metric
- ✓ Good for small objects
- ✓ No need to manually set class weights

**Disadvantages:**
- ✗ Less stable gradients (can be noisy early in training)
- ✗ May converge slower than cross-entropy
- ✗ Requires careful implementation (avoid division by zero)

**When to Use:**
- Minority class is critical (floods, buildings, ships)
- Small objects or regions
- Imbalanced datasets (>10:1 ratio)

**Common in EO:** Flood mapping, building extraction, road segmentation

---

## SLIDE 36: Loss Function 3 - IoU Loss (Jaccard)
**Concept:** Similar to Dice, measures overlap ratio

**Formula:**
```
IoU = |Prediction ∩ Truth| / |Prediction ∪ Truth|
IoU Loss = 1 - IoU
```

**Difference from Dice:**
- IoU: Intersection / Union
- Dice: 2 * Intersection / (Sum of areas)
- Numerically different but conceptually similar

**Visual:** Venn diagram highlighting union vs sum

**Properties:**
- Emphasizes boundary accuracy
- Penalizes both false positives and false negatives equally
- Standard metric in segmentation challenges

---

## SLIDE 37: Dice vs IoU - When to Choose?
**Dice Loss:**
- More forgiving (numerator x2)
- Smoother gradients
- Commonly preferred for training

**IoU Loss:**
- Stricter metric
- Better aligns with evaluation
- Good for boundary-critical tasks

**In Practice:**
- Both work well for imbalanced EO data
- Dice slightly more popular in medical and EO imaging
- Try both and compare results

**Relationship:**
```
Dice = 2*IoU / (1 + IoU)
```

---

## SLIDE 38: Loss Function 4 - Combined Losses
**Best of Both Worlds:** Combine complementary losses

**Common Combination:**
```
Total Loss = α * CE_loss + β * Dice_loss
```
Where α and β are weighting factors (e.g., α=0.5, β=0.5)

**Why Combine?**
- Cross-Entropy: Strong gradients, pixel-level accuracy
- Dice: Region overlap, handles imbalance

**Benefits:**
- Stable training (CE provides strong signal)
- Balanced optimization (Dice ensures overlap)
- Often achieves best results

**Example:**
```python
def combined_loss(y_true, y_pred):
    ce = categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * ce + 0.5 * dice
```

---

## SLIDE 39: Other Losses - Brief Mention
**Focal Loss:**
- Modified cross-entropy
- Down-weights easy examples
- Focuses on hard-to-classify pixels
- Good for extreme imbalance (used in object detection)

**Tversky Loss:**
- Generalization of Dice
- Adjustable weights for false positives vs false negatives
- Use when one type of error is more costly

**Boundary Loss:**
- Emphasizes accuracy at region boundaries
- Useful when precise edges are critical (coastlines, property lines)

**In This Course:**
- Focus on Cross-Entropy, Dice, and IoU
- These cover 90% of segmentation use cases

---

## SLIDE 40: Loss Function Selection Guide
**Decision Tree:**

```
Is your data balanced (classes ~50/50)?
├─ Yes → Cross-Entropy (simple, effective)
└─ No (imbalanced)
    ├─ Minority class critical? → Dice or IoU Loss
    ├─ Need stable training? → Weighted Cross-Entropy
    └─ Want best performance? → Combined Loss (CE + Dice)
```

**EO Common Practice:**
- Flood mapping: Dice or Combined
- Land cover (balanced): Cross-Entropy
- Building extraction: Dice or IoU
- Road extraction: Combined Loss

---

## SLIDE 41: Practical Example - Flood Mapping Loss Choice
**Scenario:**
- Dataset: 1000 Sentinel-1 images, Central Luzon floods
- Class distribution: 92% non-flooded, 8% flooded

**Option 1: Cross-Entropy**
- Result: Model predicts mostly non-flooded, IoU = 0.12
- Issue: Misses many flooded areas

**Option 2: Weighted Cross-Entropy**
- Weight flooded class 11.5x
- Result: Better, IoU = 0.54
- Issue: Some false positives

**Option 3: Dice Loss**
- Result: IoU = 0.68, good recall
- Issue: Slightly noisy predictions

**Option 4: Combined (CE + Dice)**
- Result: IoU = 0.73, balanced precision/recall ✓
- **Best choice for this case**

---

## SLIDE 42: Implementing Loss Functions
**In Practice:**

**TensorFlow/Keras:**
```python
# Built-in
loss = tf.keras.losses.CategoricalCrossentropy()

# Custom Dice Loss
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
```

**PyTorch:**
```python
# Built-in
criterion = nn.CrossEntropyLoss()

# Custom Dice Loss
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1.
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
```

---

## SLIDE 43: Hands-On Preview - Session 2
**Next Session:** Flood Mapping with U-Net

**What You'll Do:**
1. Load Sentinel-1 SAR data (Typhoon Ulysses floods)
2. Build U-Net model in TensorFlow/PyTorch
3. Train with Dice Loss
4. Evaluate performance (IoU, F1-score)
5. Visualize flood predictions

**Dataset:**
- ~500 pre-processed SAR patches (256x256)
- Binary masks (flooded / non-flooded)
- Real Central Luzon flood event

**Expected Results:**
- IoU > 0.70 with fine-tuned model
- Visual flood extent maps

---

## SLIDE 44: Key Takeaways - Session 1
**Semantic Segmentation:**
- ✓ Pixel-wise classification for precise boundaries
- ✓ Different from classification (image-level) and detection (bounding boxes)
- ✓ Essential for EO applications requiring spatial accuracy

**U-Net Architecture:**
- ✓ Encoder-decoder structure with skip connections
- ✓ Skip connections preserve spatial detail
- ✓ Proven architecture for limited training data

**Loss Functions:**
- ✓ Cross-Entropy: standard but sensitive to imbalance
- ✓ Dice/IoU: handle imbalance, optimize overlap
- ✓ Combined losses often best for EO

**Applications:**
- ✓ Flood mapping, land cover, buildings, roads, vegetation
- ✓ Wide adoption in Earth Observation community

---

## SLIDE 45: Quick Knowledge Check
**Question 1:** What is the main advantage of skip connections in U-Net?
- A) Faster training
- B) Preserves spatial details
- C) Reduces parameters
- D) Increases depth

**Question 2:** For flood mapping with 5% flooded pixels, which loss is most appropriate?
- A) Standard Cross-Entropy
- B) Mean Squared Error
- C) Dice Loss
- D) Hinge Loss

**Question 3:** What does the bottleneck layer in U-Net contain?
- A) Most spatial detail
- B) Most compressed representation
- C) Skip connections
- D) Output predictions

**Answers:** 1-B, 2-C, 3-B

---

## SLIDE 46: Resources & Further Reading
**Papers:**
- Ronneberger et al. (2015) - U-Net: Convolutional Networks for Biomedical Image Segmentation
- Flood mapping with U-Net: Recent papers on Sentinel-1 SAR

**Tutorials:**
- TensorFlow U-Net tutorial
- PyTorch semantic segmentation examples

**Datasets:**
- Sen1Floods11 (global flood dataset)
- DeepGlobe Land Cover Challenge
- SpaceNet Building Detection

**Philippine Context:**
- PhilSA flood monitoring initiatives
- DOST-ASTI disaster response projects

---

## SLIDE 47: Q&A and Discussion
**Discussion Prompts:**
1. What EO applications in your work could benefit from semantic segmentation?
2. What challenges do you anticipate with limited training data?
3. How might you validate segmentation results in the field?

**Open Floor for Questions**

---

## SLIDE 48: Break (10 minutes)
**Coming Up Next:**
- Session 2: Hands-on Flood Mapping with U-Net
- Get ready to code!

**During Break:**
- Ensure Google Colab access
- Check GPU availability
- Review Python/TensorFlow basics if needed

---

## SLIDE 49: Preview - Session 2 Workflow
**Hands-On Steps:**
1. **Setup** (10 min): Environment, data loading
2. **Data Exploration** (15 min): Visualize SAR images and masks
3. **Model Building** (20 min): Implement U-Net
4. **Training** (30 min): Train on flood dataset
5. **Evaluation** (30 min): Metrics and visualization
6. **Discussion** (15 min): Results interpretation

**Be Prepared:**
- Basic Python knowledge
- Familiarity with NumPy
- Patience (model training takes time!)

---

## END OF SESSION 1 PRESENTATION

**Total Slides:** 49
**Duration:** ~75 minutes (with discussions)
**Format:** Instructor-led with interactive elements

---

## INSTRUCTOR NOTES:
- **Slide 10:** Allow 3-4 minutes for pair discussion
- **Slide 29:** Emphasize transfer learning (revisit in Session 2)
- **Slide 41:** Use as case study, reference in hands-on
- **Slides 42:** Code examples for reference, will be in notebooks
- **Slide 45:** Quick quiz to check understanding
- **Timing:** Aim for 20-20-15-20 split (concept-architecture-applications-losses)
