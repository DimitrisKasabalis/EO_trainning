# CNN Architectures for Earth Observation

## Overview

This document provides detailed explanations of classic and modern Convolutional Neural Network (CNN) architectures used in Earth observation applications. Understanding these architectures helps you choose the right model for your specific EO task and data constraints.

---

## Table of Contents

1. [LeNet-5 (1998)](#lenet-5-1998)
2. [AlexNet (2012)](#alexnet-2012)
3. [VGG-16 (2014)](#vgg-16-2014)
4. [ResNet (2015)](#resnet-2015)
5. [U-Net (2015)](#u-net-2015)
6. [EfficientNet (2019)](#efficientnet-2019)
7. [Vision Transformers (2020)](#vision-transformers-2020)
8. [Architecture Comparison Table](#architecture-comparison-table)
9. [Choosing the Right Architecture](#choosing-the-right-architecture)

---

## LeNet-5 (1998)

### Overview
The pioneering CNN architecture that proved deep learning could solve real-world image classification problems. Originally designed for handwritten digit recognition (MNIST dataset).

### Architecture

```
Input (32×32×1)
    ↓
Conv1: 6 filters (5×5) → 28×28×6
    ↓
AvgPool1: (2×2) → 14×14×6
    ↓
Conv2: 16 filters (5×5) → 10×10×16
    ↓
AvgPool2: (2×2) → 5×5×16
    ↓
Flatten → 400
    ↓
FC1: 120 neurons
    ↓
FC2: 84 neurons
    ↓
Output: 10 classes (softmax)
```

### Key Characteristics
- **Total Parameters:** ~60,000
- **Depth:** 7 layers (2 conv + 2 pool + 3 FC)
- **Activation:** Tanh (originally), ReLU in modern implementations
- **Pooling:** Average pooling (subsampling)

### EO Applications
- **Historical significance:** Inspired all modern CNNs
- **Current use:** Educational examples, very simple classification tasks
- **Limitations:** Too shallow for complex EO patterns (mixed pixels, atmospheric effects)
- **Philippine context:** Not recommended for operational use; Session 3 demonstrations only

### Strengths
- Simple to understand and implement
- Fast training and inference
- Low memory requirements

### Weaknesses
- Limited capacity for complex patterns
- Outdated design choices (average pooling, tanh)
- Poor performance on modern datasets

---

## AlexNet (2012)

### Overview
The architecture that sparked the deep learning revolution by winning ImageNet 2012 with a massive margin. Proved that deep CNNs trained on GPUs could outperform traditional computer vision methods.

### Architecture

```
Input (227×227×3)
    ↓
Conv1: 96 filters (11×11, stride 4) → 55×55×96
    ↓ ReLU → MaxPool (3×3, stride 2)
Conv2: 256 filters (5×5) → 27×27×256
    ↓ ReLU → MaxPool (3×3, stride 2)
Conv3: 384 filters (3×3) → 13×13×384
    ↓ ReLU
Conv4: 384 filters (3×3) → 13×13×384
    ↓ ReLU
Conv5: 256 filters (3×3) → 13×13×256
    ↓ ReLU → MaxPool (3×3, stride 2) → 6×6×256
    ↓
Flatten → FC1: 4096 → Dropout (0.5)
    ↓
FC2: 4096 → Dropout (0.5)
    ↓
Output: 1000 classes (softmax)
```

### Key Characteristics
- **Total Parameters:** ~60 million
- **Depth:** 8 learned layers (5 conv + 3 FC)
- **Innovations:**
  - First to use ReLU activation (faster training)
  - Dropout for regularization
  - Data augmentation at scale
  - GPU training (2 GTX 580s)

### EO Applications
- **Transfer learning baseline:** Pre-trained weights available
- **Scene classification:** Adapted for Sentinel-2 RGB composite classification
- **Limitations:** Designed for RGB images, needs adaptation for multi-band EO data

### Strengths
- Powerful feature extractor
- Pre-trained on ImageNet (useful for transfer learning)
- ReLU and dropout prevent overfitting

### Weaknesses
- Very large FC layers (most parameters)
- Fixed input size (227×227)
- Not optimized for modern hardware

---

## VGG-16 (2014)

### Overview
Very Deep Convolutional Networks (VGG) demonstrated that network depth (number of layers) is crucial for performance. Used simple 3×3 convolutions stacked repeatedly.

### Architecture

```
Input (224×224×3)
    ↓
Block 1:
  Conv (64, 3×3) → Conv (64, 3×3) → MaxPool (2×2)
    ↓
Block 2:
  Conv (128, 3×3) → Conv (128, 3×3) → MaxPool (2×2)
    ↓
Block 3:
  Conv (256, 3×3) → Conv (256, 3×3) → Conv (256, 3×3) → MaxPool (2×2)
    ↓
Block 4:
  Conv (512, 3×3) → Conv (512, 3×3) → Conv (512, 3×3) → MaxPool (2×2)
    ↓
Block 5:
  Conv (512, 3×3) → Conv (512, 3×3) → Conv (512, 3×3) → MaxPool (2×2)
    ↓
Flatten → FC (4096) → FC (4096) → Output (1000)
```

### Key Characteristics
- **Total Parameters:** 138 million (mostly in FC layers)
- **Depth:** 16 weight layers (13 conv + 3 FC)
- **Key insight:** Two 3×3 convs have same receptive field as one 5×5, but with fewer parameters and more non-linearity
- **Variants:** VGG-11, VGG-13, VGG-16, VGG-19

### EO Applications

#### Scene Classification
- **EuroSAT benchmark:** VGG-16 achieves ~95% accuracy on 10-class Sentinel-2 classification
- **PhilSA land cover:** Fine-tuned VGG-16 for Philippine-specific classes (forest types, mangroves, rice)
- **Cloud detection:** Binary classification (cloud vs clear)

#### Transfer Learning
- **Standard baseline:** Pre-trained VGG-16 weights widely used
- **Feature extraction:** Remove top FC layers, use conv features for traditional ML
- **Fine-tuning:** Freeze early layers, train last conv blocks + new classifier head

### Strengths
- Simple, uniform architecture (easy to understand and implement)
- Excellent feature extractor for transfer learning
- Pre-trained weights readily available
- Robust performance across many tasks

### Weaknesses
- Very slow training and inference (138M parameters)
- High memory consumption
- Large model size (528 MB)
- FC layers contain 90% of parameters (inefficient)

### Philippine EO Context
**Recommended for:**
- Initial experiments with transfer learning
- Small-scale land cover classification (<10 classes)
- When interpretability and simplicity are priorities

**Not recommended for:**
- Large-scale production systems (use ResNet or EfficientNet)
- Resource-constrained deployment (edge devices, mobile)
- When training from scratch (too many parameters)

---

## ResNet (2015)

### Overview
Residual Networks solved the vanishing gradient problem through skip connections (residual blocks), enabling training of very deep networks (50, 101, 152+ layers). Won ImageNet 2015.

### Residual Block (Key Innovation)

```
Input x
    ↓
    ├─────────────────────┐  (identity shortcut)
    ↓                     ↓
Conv (3×3)
    ↓ ReLU                ↓
Conv (3×3)               ↓
    ↓                     ↓
    └──────(+)───────────┘  (element-wise addition)
         ↓
       ReLU
      Output
```

**Mathematical Formulation:**
- Standard block learns: `H(x) = F(x)`
- Residual block learns: `H(x) = F(x) + x`
- Easier to learn `F(x) = H(x) - x` (residual) than `H(x)` directly

### ResNet-50 Architecture

```
Input (224×224×3)
    ↓
Conv1: 64 filters (7×7, stride 2) → 112×112×64
    ↓ MaxPool (3×3, stride 2) → 56×56×64
    ↓
Conv2_x: 3 residual blocks → 56×56×256
    ↓
Conv3_x: 4 residual blocks → 28×28×512
    ↓
Conv4_x: 6 residual blocks → 14×14×1024
    ↓
Conv5_x: 3 residual blocks → 7×7×2048
    ↓
Global Average Pooling → 2048
    ↓
FC → Output (1000 classes)
```

### Key Characteristics
- **Total Parameters:** 25.6 million (ResNet-50)
- **Depth:** 50 layers (ResNet-50), up to 152+ for ResNet-152
- **Innovations:**
  - Skip connections prevent vanishing gradients
  - Batch normalization after each conv
  - Global average pooling (no large FC layers)
  - Bottleneck blocks (1×1 → 3×3 → 1×1) reduce computation

### EO Applications

#### Land Cover Classification
**Palawan Case Study (Session 2 → Session 4 progression):**
- **Input:** 256×256×10 Sentinel-2 patch (all 10 bands)
- **Adaptation:** Modify first conv layer for 10 channels instead of 3 RGB
- **Classes:** Primary forest, secondary forest, mangroves, agriculture, grassland, water, urban, bare soil
- **Performance:** 88-92% accuracy (vs 83-86% for Random Forest)
- **Training:** 1,000 samples per class, 50 epochs, ~3 hours on Colab T4 GPU

#### Multi-temporal Change Detection
- **Input:** Stack temporal images as channels (e.g., 12 months × 10 bands = 120 channels)
- **Output:** Binary change mask
- **Application:** Deforestation detection in protected areas

#### Scene-Level Disaster Assessment
- **Pre-disaster:** Sentinel-2 optical
- **Post-disaster:** Sentinel-1 SAR + Sentinel-2 (cloud permitting)
- **Classes:** No damage, minor damage, major damage, destroyed
- **Integration:** NDRRMC rapid assessment pipeline

### Transfer Learning Strategy

```python
# Load ImageNet pre-trained ResNet-50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# For Sentinel-2 (10 bands), modify input layer
# Option 1: Use RGB bands only (B4, B3, B2)
# Option 2: Create new input layer + copy weights (PCA of bands → 3 channels)
# Option 3: Train from scratch (more data needed)

# Freeze early layers (low-level features are universal)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Add custom head for Philippine land cover
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(8, activation='softmax')(x)  # 8 classes

model = Model(inputs=base_model.input, outputs=output)
```

### Strengths
- State-of-the-art accuracy on many benchmarks
- Trains efficiently despite depth (skip connections)
- No large FC layers (memory efficient)
- Pre-trained weights widely available
- Scales well to very deep networks

### Weaknesses
- More complex than VGG (harder to understand residual blocks)
- Still computationally intensive for very deep variants
- Requires careful learning rate scheduling
- Fixed input size (can be adapted with global pooling)

### Philippine EO Recommendations

**Use ResNet-50 when:**
- Need state-of-the-art classification accuracy
- Have ≥1,000 training samples per class
- Can fine-tune pre-trained weights
- Target is scene-level classification

**Use ResNet-101/152 when:**
- Have very large datasets (>10,000 samples per class)
- Need maximum accuracy for critical applications (disaster response)
- Have GPU resources for longer training

**Avoid ResNet when:**
- Need pixel-level segmentation (use U-Net instead)
- Have very limited data (<500 samples) and no pre-trained weights
- Require real-time edge deployment (use MobileNet/EfficientNet)

---

## U-Net (2015)

### Overview
Designed for biomedical image segmentation (cell boundaries in microscopy), U-Net revolutionized semantic segmentation by using skip connections to preserve spatial detail. Now the standard architecture for pixel-wise EO tasks.

### Architecture

```
Encoder (Contracting Path):

Input (256×256×10)  # Sentinel-2
    ↓
Conv-Conv (64) → 256×256×64  ──────────┐ (skip connection)
    ↓ MaxPool (2×2)                    │
Conv-Conv (128) → 128×128×128 ─────────┼──┐
    ↓ MaxPool (2×2)                    │  │
Conv-Conv (256) → 64×64×256 ───────────┼──┼──┐
    ↓ MaxPool (2×2)                    │  │  │
Conv-Conv (512) → 32×32×512 ───────────┼──┼──┼──┐
    ↓ MaxPool (2×2)                    │  │  │  │
                                       │  │  │  │
Bottleneck:                            │  │  │  │
Conv-Conv (1024) → 16×16×1024          │  │  │  │
                                       │  │  │  │
Decoder (Expansive Path):              │  │  │  │
    ↓ UpConv (2×2) → 32×32×512 ────────┘  │  │  │
    ↓ Concatenate + Conv-Conv (512)       │  │  │
    ↓ UpConv (2×2) → 64×64×256 ───────────┘  │  │
    ↓ Concatenate + Conv-Conv (256)          │  │
    ↓ UpConv (2×2) → 128×128×128 ────────────┘  │
    ↓ Concatenate + Conv-Conv (128)             │
    ↓ UpConv (2×2) → 256×256×64 ────────────────┘
    ↓ Concatenate + Conv-Conv (64)
    ↓
Output: 256×256×2  (flood/non-flood)
```

### Key Characteristics
- **Total Parameters:** ~31 million (depends on depth and channels)
- **Architecture:**
  - Symmetric encoder-decoder structure (U-shape)
  - Skip connections concatenate encoder features to decoder
  - No FC layers (fully convolutional)
- **Innovations:**
  - Skip connections preserve spatial detail lost during downsampling
  - Works with small datasets (100-500 annotated images)
  - Outputs same spatial size as input

### EO Applications

#### Flood Mapping (Session 4 Focus)

**Pampanga River Basin Case Study:**

**Input:**
- **Pre-event:** Sentinel-1 VV polarization (baseline)
- **Post-event:** Sentinel-1 VV polarization (flood)
- **Stack:** 2-channel input (pre + post)
- **Size:** 512×512 patches (5.12 km × 5.12 km at 10m resolution)

**Output:**
- **Binary mask:** Flooded (1) vs Non-flooded (0)
- **Post-processing:** Morphological operations (remove small isolated pixels)
- **Vectorization:** Convert to polygons for GIS integration

**Training:**
- **Dataset:** 300 labeled images (from historical floods)
- **Augmentation:** Rotation, flips (8× expansion → 2,400 samples)
- **Validation:** 20% holdout + field validation
- **Performance:** 93% pixel accuracy, 0.89 F1-score

**Operational Workflow:**
```
Typhoon Event
    ↓
Sentinel-1 Acquisition (within 12 hours)
    ↓
Automated Preprocessing (GEE)
    ↓
U-Net Inference (30 min for entire region)
    ↓
Validation + Post-processing (1 hour)
    ↓
Flood Map Delivered to NDRRMC (Total: 6 hours)
```

**Impact:**
- **Speed:** 6 hours vs 2 days manual interpretation
- **Coverage:** Consistent mapping across large areas
- **Repeatability:** Same quality regardless of analyst
- **Cost:** ~$0 data (Sentinel-1 free) + GPU compute (~$1)

#### Building Footprint Extraction

**Metro Manila Informal Settlements:**

**Input:**
- High-resolution imagery (PlanetScope 3m or drone 0.1m)
- RGB + NIR (4 channels)

**Output:**
- Binary mask: Building (1) vs Background (0)
- Post-processing: Vectorization + polygon simplification

**Applications:**
- Disaster vulnerability assessment (NEDA)
- Urban planning (MMDA)
- Population estimation (PSA)

**Challenges:**
- Dense, overlapping structures
- Mixed materials (metal, concrete, vegetation)
- Shadows and occlusions
- Incomplete training data

**Solutions:**
- Data augmentation (brightness, contrast for shadow variations)
- Post-processing (minimum area threshold, morphological closing)
- Ensemble models (average predictions from 3 U-Nets)

#### Forest Degradation Detection

**Input:**
- Multi-temporal Sentinel-2 (dry season mosaic, 10 bands)
- NDVI, NDMI as additional channels
- 256×256 patches

**Output:**
- Multi-class segmentation:
  - Intact forest
  - Degraded forest (selective logging)
  - Recent clearing
  - Background

**DENR Application:**
- Protected area monitoring
- REDD+ MRV compliance
- Illegal logging hotspots

#### Agricultural Field Boundaries

**Input:**
- Peak growing season Sentinel-2 (10 bands)
- NDVI, NDWI for crop identification

**Output:**
- Binary mask: Field boundary (1) vs Interior (0)
- Post-process: Skeletonization + vectorization

**Applications:**
- Field-level yield estimation
- Irrigation planning
- Subsidy verification (DA)

### U-Net Variants for EO

#### U-Net++
- Nested skip connections (denser connections)
- Better gradient flow
- Improved boundary detection
- **Use case:** High-precision building extraction

#### Attention U-Net
- Attention gates highlight relevant features
- Suppresses background noise
- **Use case:** Small object detection (isolated buildings)

#### 3D U-Net
- Processes volumetric data (time as 3rd dimension)
- **Use case:** Multi-temporal change detection

### Strengths
- Excellent pixel-wise accuracy with limited training data
- Preserves spatial detail through skip connections
- No FC layers (memory efficient, works with any input size)
- State-of-the-art for segmentation tasks
- Fast inference (0.1-0.5 sec per 512×512 image on GPU)

### Weaknesses
- Requires pixel-level annotations (labor-intensive)
- Checkerboard artifacts from upsampling (use resize-convolution)
- Memory intensive during training (large feature maps)
- Poor at capturing very long-range dependencies

### Philippine EO Recommendations

**Use U-Net when:**
- Task requires pixel-wise predictions (segmentation)
- Need to preserve fine spatial detail (boundaries)
- Have 100-500 labeled images (feasible with augmentation)
- Examples: Flood maps, building extraction, field boundaries

**Avoid U-Net when:**
- Task is scene-level classification (use ResNet)
- Need bounding boxes (use object detection architecture)
- Have very limited training data (<50 images)

---

## EfficientNet (2019)

### Overview
Systematically scales network depth, width, and resolution using compound scaling. Achieves state-of-the-art accuracy with fewer parameters and FLOPs than ResNet.

### Key Innovation: Compound Scaling

Traditional scaling (inefficient):
- Scale depth only: ResNet-50 → ResNet-101 → ResNet-152
- Scale width only: ResNet-50 (64, 128, 256, 512 filters) → (128, 256, 512, 1024)
- Scale resolution only: 224×224 → 384×384

EfficientNet compound scaling:
- Simultaneously scale depth × width × resolution
- Optimal ratios determined through neural architecture search (NAS)

**Scaling Formula:**
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

subject to: α × β² × γ² ≈ 2
            α ≥ 1, β ≥ 1, γ ≥ 1
```

### EfficientNet Family

| Model | Parameters | Top-1 Accuracy (ImageNet) | FLOPs | Relative Speed |
|-------|-----------|---------------------------|-------|----------------|
| EfficientNet-B0 | 5.3M | 77.1% | 0.39B | 1.0× |
| EfficientNet-B1 | 7.8M | 79.1% | 0.70B | 0.88× |
| EfficientNet-B2 | 9.2M | 80.1% | 1.0B | 0.77× |
| EfficientNet-B3 | 12M | 81.6% | 1.8B | 0.63× |
| EfficientNet-B4 | 19M | 82.9% | 4.2B | 0.46× |
| EfficientNet-B5 | 30M | 83.6% | 9.9B | 0.33× |
| EfficientNet-B6 | 43M | 84.0% | 19B | 0.25× |
| EfficientNet-B7 | 66M | 84.3% | 37B | 0.17× |

**Comparison:** EfficientNet-B7 (66M params) matches ResNet-152 (60M params) accuracy but 8× fewer FLOPs!

### EO Applications

#### High-Efficiency Scene Classification
**Use case:** Real-time land cover classification on edge devices (drones, mobile apps)

**Setup:**
- Model: EfficientNet-B0 (smallest, fastest)
- Input: 224×224 Sentinel-2 RGB composite
- Fine-tuning: Last 20 layers + new classification head
- Deployment: TensorFlow Lite for mobile/edge

**Performance:**
- **Accuracy:** 92% (comparable to ResNet-50's 93%)
- **Inference speed:** 50 ms on smartphone (vs 180 ms for ResNet-50)
- **Model size:** 21 MB (vs 98 MB for ResNet-50)

#### Balanced Performance for Production
**PhilSA National Land Cover System:**

**Requirements:**
- Process entire Philippines (~300,000 km²) quarterly
- 10m resolution Sentinel-2 (billions of pixels)
- 10 land cover classes
- Inference on GPU cluster (8× A100)

**Architecture choice:**
- EfficientNet-B4 (balanced accuracy and speed)
- Batch inference on 256×256 patches
- **Throughput:** 10,000 patches/sec (distributed)
- **Total time:** ~8 hours for full Philippines mosaic

**Why EfficientNet over ResNet?**
- 30% faster inference (more patches per hour)
- Lower GPU memory (can use larger batch sizes)
- Better accuracy per parameter (easier to deploy updates)

### Transfer Learning with EfficientNet

```python
from tensorflow.keras.applications import EfficientNetB3

# Load pre-trained EfficientNet-B3
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)

# Freeze early layers
base_model.trainable = False

# Add custom head
inputs = tf.keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
```

### Strengths
- Best accuracy-to-parameters ratio
- Faster inference than ResNet (for similar accuracy)
- Smaller model sizes (easier deployment)
- Compound scaling principle is elegant and effective
- Pre-trained weights available for all variants

### Weaknesses
- More complex architecture (harder to understand)
- Requires careful scaling hyperparameter tuning if modifying
- Some variants (B6, B7) are very slow despite efficiency claims
- Less mature ecosystem than ResNet (fewer tutorials)

### Philippine EO Recommendations

**Use EfficientNet-B0 to B2 when:**
- Deploying on resource-constrained devices (drones, mobile)
- Need fast inference with good accuracy
- Model size is critical (limited storage/bandwidth)

**Use EfficientNet-B3 to B5 when:**
- Production system with accuracy priority
- Have GPU resources for training/inference
- Want best accuracy-efficiency balance

**Avoid EfficientNet when:**
- Need pixel-level segmentation (use U-Net)
- Require maximum interpretability (use VGG or shallow ResNet)
- Working with very small datasets (<500 samples per class)

---

## Vision Transformers (2020)

### Overview
Vision Transformers (ViT) apply the Transformer architecture (from NLP) to images by treating image patches as "tokens." Challenges the dominance of CNNs, especially with very large datasets.

### Architecture

**Patch Embedding:**
1. Divide image into fixed-size patches (e.g., 16×16)
2. Flatten each patch into a vector
3. Linearly project to embedding dimension (768 for ViT-B)
4. Add positional embeddings (learnable)

```
Input Image: 224×224×3
    ↓ Split into patches
Patches: 196 patches of 16×16×3
    ↓ Flatten each patch
Vectors: 196 vectors of 768
    ↓ Add [CLS] token + positional embeddings
    ↓
Transformer Encoder (12 layers for ViT-B):
    - Multi-head self-attention (capture global context)
    - MLP feedforward
    - Layer normalization
    - Residual connections
    ↓
[CLS] token output → MLP Head → Classification
```

### ViT vs CNN

| Aspect | CNN | Vision Transformer |
|--------|-----|-------------------|
| **Inductive bias** | Strong (locality, translation invariance) | Weak (learns from data) |
| **Receptive field** | Grows with depth (local → global) | Global from layer 1 (self-attention) |
| **Data requirements** | Moderate (1K-10K samples) | Large (100K+ samples) |
| **Pre-training** | ImageNet (1.3M images) | ImageNet-21K (14M images) or JFT (300M) |
| **Interpretability** | Filter visualization | Attention maps |
| **Computational cost** | O(n²) for n×n image | O(p²) for p patches (where p << n) |

### EO Applications

#### Large-Scale Multi-Spectral Classification

**Advantages for EO:**
1. **Global context:** Self-attention sees entire image from layer 1 (useful for context-dependent classification)
2. **Multi-scale processing:** Patch size can be tuned (16×16 for fine detail, 32×32 for efficiency)
3. **Transfer learning:** Pre-trained ViT adapts well to multi-band imagery

**SatViT (Satellite Vision Transformer):**
- Specialized ViT variant for Sentinel-2
- Pre-trained on 1M Sentinel-2 images
- 10-band input (all Sentinel-2 bands)
- Fine-tunes for specific tasks (land cover, crop type, disaster mapping)

**Performance:**
- **EuroSAT:** 98.8% accuracy (vs 98.6% for ResNet-50)
- **fMoW (satellite dataset):** 84.3% (vs 81.7% for ResNet-101)

#### Time Series Analysis

**Temporal ViT:**
- Input: Sequence of image patches across time
- Self-attention captures temporal dependencies
- **Application:** Crop phenology tracking, deforestation progression

**Example:**
```
Input: 12 monthly Sentinel-2 images (256×256×10)
    ↓
Patchify: 256 patches per image × 12 months = 3,072 patches
    ↓
Temporal-spatial attention:
    - Attention across space (within each month)
    - Attention across time (same spatial location)
    ↓
Output: Annual crop type classification
```

**Advantage over CNN+LSTM:**
- ViT captures long-range temporal dependencies better
- Single architecture (no separate spatial + temporal modules)

### Philippine EO Context

**Current Status (2025):**
- **Research phase:** ViT showing promise on large Sentinel-2 datasets
- **Not yet operational:** Requires very large training datasets (PhilSA collecting)
- **Transfer learning:** Pre-trained ViT from ImageNet-21K can be fine-tuned

**Future Potential:**
- **National land cover:** ViT may replace ResNet once sufficient PH training data available
- **Multi-temporal monitoring:** Temporal ViT for crop classification, change detection
- **Foundation models:** Large ViT pre-trained on global satellite data, fine-tuned for Philippines

**Recommendations:**
- **Experiment:** Try ViT for research projects with large datasets (>10,000 samples)
- **Use pre-trained models:** Google's ViT or SatViT for transfer learning
- **Stick with CNNs for now:** For operational systems and small datasets

### Strengths
- State-of-the-art with very large datasets
- Global receptive field from layer 1
- Attention maps aid interpretability
- Scales well to massive models and data
- Strong performance on diverse tasks (classification, detection, segmentation)

### Weaknesses
- Requires enormous training data (100K+ samples)
- Slower training than CNNs (quadratic self-attention complexity)
- Larger models (ViT-Large: 300M+ parameters)
- Less mature for EO (fewer pre-trained models than ResNet)
- Weaker inductive biases (needs more data to learn spatial structures)

---

## Architecture Comparison Table

| Architecture | Parameters | Use Case | Data Need | Speed | Accuracy | EO Suitability |
|--------------|------------|----------|-----------|-------|----------|----------------|
| **LeNet-5** | 60K | Educational | Tiny | Very Fast | Low | ⭐ |
| **AlexNet** | 60M | Historical | Medium | Slow | Medium | ⭐⭐ |
| **VGG-16** | 138M | Transfer learning baseline | Medium | Slow | High | ⭐⭐⭐ |
| **ResNet-50** | 26M | Scene classification | Medium | Fast | Very High | ⭐⭐⭐⭐⭐ |
| **U-Net** | 31M | Semantic segmentation | Small-Medium | Fast | Very High | ⭐⭐⭐⭐⭐ |
| **EfficientNet-B3** | 12M | Production (balanced) | Medium | Very Fast | Very High | ⭐⭐⭐⭐⭐ |
| **EfficientNet-B0** | 5M | Edge deployment | Small-Medium | Ultra Fast | High | ⭐⭐⭐⭐ |
| **ViT-Base** | 86M | Large-scale research | Very Large | Slow | State-of-art | ⭐⭐⭐ |

### Legend:
- **Parameters:** Total number of learnable parameters
- **Data Need:** Training samples required (Tiny: <1K, Small: 1K-5K, Medium: 5K-50K, Large: 50K-500K, Very Large: 500K+)
- **Speed:** Inference speed on V100 GPU for 256×256 image
- **EO Suitability:** ⭐ (not recommended) to ⭐⭐⭐⭐⭐ (highly recommended for Philippine EO)

---

## Choosing the Right Architecture

### Decision Tree

```
What is your EO task?
│
├─ Scene-level classification (one label per image)
│  │
│  ├─ Have large dataset (>10K samples)?
│  │  ├─ Yes → ResNet-50 or EfficientNet-B3
│  │  └─ No → EfficientNet-B0 with transfer learning
│  │
│  ├─ Need edge deployment (drone, mobile)?
│  │  └─ EfficientNet-B0 or MobileNetV3
│  │
│  └─ Experimental (very large data)?
│     └─ Vision Transformer (ViT-Base)
│
├─ Pixel-level segmentation (classify every pixel)
│  │
│  ├─ Have 100-500 labeled images?
│  │  └─ Yes → U-Net (standard)
│  │
│  ├─ Need high precision boundaries?
│  │  └─ U-Net++ or Attention U-Net
│  │
│  └─ Multi-temporal segmentation?
│     └─ 3D U-Net or U-Net + LSTM
│
├─ Object detection (bounding boxes)
│  │
│  ├─ Need real-time performance?
│  │  └─ YOLOv8
│  │
│  ├─ Need maximum accuracy?
│  │  └─ Faster R-CNN (ResNet-50 backbone)
│  │
│  └─ Balanced?
│     └─ RetinaNet (EfficientNet backbone)
│
└─ Change detection
   │
   ├─ Binary change mask?
   │  └─ Siamese U-Net
   │
   ├─ Multi-class change?
   │  └─ Early fusion CNN (ResNet encoder + U-Net decoder)
   │
   └─ Time series analysis?
      └─ Temporal ViT or CNN + LSTM
```

### Philippine EO Recommendations by Stakeholder

#### PhilSA (National Scale, High Accuracy)
- **Land cover classification:** ResNet-50 or EfficientNet-B4
- **Cloud masking:** U-Net
- **Disaster rapid mapping:** U-Net (segmentation) + YOLOv8 (object detection)
- **Infrastructure:** EfficientNet-B0 for GPU clusters (throughput optimization)

#### DENR (Forest Monitoring, Moderate Resources)
- **Protected area monitoring:** ResNet-50 (scene classification)
- **Deforestation detection:** U-Net (segmentation) or Siamese CNN
- **Illegal logging:** YOLOv8 (detect clearings) + ResNet (classify severity)
- **Transfer learning:** Use PhilSA pre-trained weights, fine-tune for specific regions

#### LGUs (Local Scale, Limited Resources)
- **Land use planning:** EfficientNet-B0 (fast, small model)
- **Flood risk mapping:** U-Net (pre-trained on PhilSA national data)
- **Building detection:** U-Net or YOLOv8 (depending on high-res imagery availability)
- **Deployment:** Google Colab for training, TensorFlow Lite for local inference

#### Research Institutions (Universities, ASTI)
- **Experimentation:** Try all architectures, publish comparisons
- **Method development:** Custom architectures (e.g., Attention U-Net for specific PH challenges)
- **Foundation models:** Contribute to large-scale Sentinel-2 pre-training for SE Asia
- **Open source:** Share trained models for community use

---

## Further Reading

### Papers
- **LeNet:** LeCun et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*
- **AlexNet:** Krizhevsky et al. (2012). ImageNet classification with deep CNNs. *NeurIPS*
- **VGG:** Simonyan & Zisserman (2014). Very deep convolutional networks. *ICLR*
- **ResNet:** He et al. (2016). Deep residual learning for image recognition. *CVPR*
- **U-Net:** Ronneberger et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*
- **EfficientNet:** Tan & Le (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML*
- **Vision Transformer:** Dosovitskiy et al. (2020). An image is worth 16×16 words. *ICLR 2021*

### EO-Specific Reviews
- Zhu et al. (2017). Deep learning in remote sensing. *IEEE GRSM*
- Ma et al. (2019). Deep learning in remote sensing applications. *ISPRS*
- Yuan et al. (2021). Deep learning in environmental remote sensing. *Remote Sensing*

### Code Repositories
- [TensorFlow/Keras Models](https://keras.io/api/applications/)
- [PyTorch torchvision models](https://pytorch.org/vision/stable/models.html)
- [Segmentation Models (Keras)](https://github.com/qubvel/segmentation_models)
- [TorchGeo](https://torchgeo.readthedocs.io/) - Geospatial datasets + models

---

**Last Updated:** October 2025
**Course:** CopPhil Advanced Training - Day 2, Session 3
**Contact:** training@copphil.org
