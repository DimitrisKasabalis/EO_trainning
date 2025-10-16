# CNN Architecture Guide for Earth Observation
**Session 3: Deep Learning Theory**

---

## Overview

This guide provides detailed explanations of CNN architectures specifically for Earth Observation (EO) applications. It covers layer types, parameter calculations, architecture design principles, and EO-specific considerations.

---

## Table of Contents

1. [CNN Building Blocks](#building-blocks)
2. [Popular Architectures](#popular-architectures)
3. [Designing CNNs for EO](#designing-for-eo)
4. [Parameter Calculations](#parameter-calculations)
5. [Architecture Patterns](#architecture-patterns)
6. [Philippine EO Applications](#philippine-applications)

---

## <a name="building-blocks"></a>1. CNN Building Blocks

### 1.1 Convolutional Layers

**Purpose:** Extract spatial features from images

**Key Parameters:**

| Parameter | Description | Typical Values | Effect |
|-----------|-------------|----------------|--------|
| **Filters (Kernels)** | Number of feature detectors | 32, 64, 128, 256, 512 | More filters = more features learned |
| **Kernel Size** | Spatial extent of filter | 3×3, 5×5, 7×7 | Larger = bigger receptive field |
| **Stride** | Step size for sliding window | 1, 2 | Larger = smaller output size |
| **Padding** | Border handling | 'same', 'valid' | 'same' preserves dimensions |
| **Activation** | Non-linearity function | ReLU, LeakyReLU | ReLU is default |

**Mathematical Operation:**

```
Output[i,j,k] = Σ Σ Σ (Input[i+m, j+n, c] × Filter[m, n, c, k]) + Bias[k]
                m n c
```

Where:
- `i, j`: Spatial coordinates
- `k`: Output channel
- `m, n`: Filter spatial dimensions
- `c`: Input channels

**Example (TensorFlow/Keras):**

```python
from tensorflow.keras.layers import Conv2D

# First convolutional layer for Sentinel-2
conv1 = Conv2D(
    filters=32,           # Learn 32 different features
    kernel_size=(3, 3),   # 3×3 filter
    strides=(1, 1),       # Slide 1 pixel at a time
    padding='same',       # Keep same spatial dimensions
    activation='relu',    # ReLU activation
    input_shape=(64, 64, 10)  # 64×64 image, 10 bands
)
```

**Output Size Calculation:**

```
output_height = (input_height - kernel_height + 2 × padding) / stride + 1
output_width = (input_width - kernel_width + 2 × padding) / stride + 1
output_channels = number_of_filters
```

**For Sentinel-2 Example:**
- Input: 64×64×10
- Conv: 32 filters, 3×3, stride=1, padding='same'
- Output: 64×64×32

---

### 1.2 Pooling Layers

**Purpose:** Reduce spatial dimensions, increase receptive field, add translation invariance

**Types:**

#### Max Pooling
Takes maximum value in each window.

```python
MaxPooling2D(pool_size=(2, 2))
```

**Example:**
```
Input (4×4):          Max Pool 2×2:
[1  3  2  4]          [3  4]
[2  1  5  3]    →     [6  7]
[6  2  1  7]
[4  5  3  2]
```

**Advantages:**
- ✅ Keeps strongest activations
- ✅ Adds slight translation invariance
- ✅ Most common choice

#### Average Pooling
Takes mean value in each window.

```python
AveragePooling2D(pool_size=(2, 2))
```

**Advantages:**
- ✅ Smoother down-sampling
- ✅ Reduces noise
- ✅ Good for regression tasks

#### Global Average Pooling
Reduces each feature map to single value.

```python
GlobalAveragePooling2D()
```

**Input:** H×W×C  
**Output:** C (one value per channel)

**Advantages:**
- ✅ Reduces parameters dramatically
- ✅ No spatial information needed
- ✅ Common before final classification

**Output Size:**

```
output_height = input_height / pool_size
output_width = input_width / pool_size
channels unchanged
```

---

### 1.3 Fully Connected (Dense) Layers

**Purpose:** Learn non-linear combinations of features for classification

```python
Dense(units=128, activation='relu')
```

**Parameters:**
```
parameters = (input_size + 1) × output_size
```

**Example:**
- Input: 512 features
- Dense: 128 neurons
- Parameters: (512 + 1) × 128 = 65,664

**Typical Pattern:**
```
Conv → Pool → Conv → Pool → Flatten → Dense → Dense → Output
```

---

### 1.4 Batch Normalization

**Purpose:** Normalize activations, accelerate training, reduce overfitting

```python
BatchNormalization()
```

**What it does:**
1. Normalizes mini-batch to mean=0, std=1
2. Applies learnable scale and shift
3. Stabilizes training

**Benefits:**
- ✅ Faster convergence
- ✅ Higher learning rates possible
- ✅ Acts as regularization

**Placement:**
```python
Conv2D(64, (3,3)) → BatchNormalization() → ReLU
```

---

### 1.5 Dropout

**Purpose:** Regularization to prevent overfitting

```python
Dropout(rate=0.5)  # Drop 50% of neurons randomly
```

**How it works:**
- Training: Randomly sets 50% of inputs to 0
- Inference: Uses all neurons (scaled)

**When to use:**
- Large networks
- Small datasets
- Signs of overfitting

**Typical rate:** 0.2-0.5

---

## <a name="popular-architectures"></a>2. Popular Architectures

### 2.1 LeNet-5 (1998)

**Historical significance:** First successful CNN

**Architecture:**
```
Input (32×32×1)
    ↓
Conv: 6 filters, 5×5 → 28×28×6
    ↓
AvgPool: 2×2 → 14×14×6
    ↓
Conv: 16 filters, 5×5 → 10×10×16
    ↓
AvgPool: 2×2 → 5×5×16
    ↓
Flatten → 400
    ↓
Dense: 120
    ↓
Dense: 84
    ↓
Output: 10 classes
```

**Parameters:** ~60,000

**Use today:** Educational purposes only

---

### 2.2 AlexNet (2012)

**Significance:** Won ImageNet 2012, proved CNNs work at scale

**Architecture:**
```
Input (227×227×3)
    ↓
Conv: 96 filters, 11×11, stride=4 → 55×55×96
MaxPool: 3×3, stride=2 → 27×27×96
    ↓
Conv: 256 filters, 5×5 → 27×27×256
MaxPool: 3×3, stride=2 → 13×13×256
    ↓
Conv: 384 filters, 3×3 → 13×13×384
Conv: 384 filters, 3×3 → 13×13×384
Conv: 256 filters, 3×3 → 13×13×256
MaxPool: 3×3, stride=2 → 6×6×256
    ↓
Flatten → 9216
Dense: 4096 + Dropout
Dense: 4096 + Dropout
Output: 1000 classes
```

**Key Innovations:**
- ReLU activation
- Dropout regularization
- Data augmentation
- GPU training

**Parameters:** ~60 million

---

### 2.3 VGG16 (2014)

**Philosophy:** Simplicity - use only 3×3 convolutions

**Architecture:**
```
Input (224×224×3)
    ↓
Block 1:
  Conv: 64, 3×3 → 224×224×64
  Conv: 64, 3×3 → 224×224×64
  MaxPool: 2×2 → 112×112×64
    ↓
Block 2:
  Conv: 128, 3×3 → 112×112×128
  Conv: 128, 3×3 → 112×112×128
  MaxPool: 2×2 → 56×56×128
    ↓
Block 3:
  Conv: 256, 3×3 → 56×56×256
  Conv: 256, 3×3 → 56×56×256
  Conv: 256, 3×3 → 56×56×256
  MaxPool: 2×2 → 28×28×256
    ↓
Block 4:
  Conv: 512, 3×3 → 28×28×512
  Conv: 512, 3×3 → 28×28×512
  Conv: 512, 3×3 → 28×28×512
  MaxPool: 2×2 → 14×14×512
    ↓
Block 5:
  Conv: 512, 3×3 → 14×14×512
  Conv: 512, 3×3 → 14×14×512
  Conv: 512, 3×3 → 14×14×512
  MaxPool: 2×2 → 7×7×512
    ↓
Flatten → 25088
Dense: 4096
Dense: 4096
Output: 1000 classes
```

**Parameters:** ~138 million

**Advantages:**
- ✅ Simple, uniform architecture
- ✅ Excellent for transfer learning
- ✅ Strong feature representations

**Disadvantages:**
- ❌ Many parameters
- ❌ Memory intensive
- ❌ Slow training

**EO Applications:**
- Scene classification
- Transfer learning base
- Feature extraction

---

### 2.4 ResNet50 (2015)

**Key Innovation:** Skip connections solve vanishing gradient

**Skip Connection:**
```
x → [Conv → BN → ReLU → Conv → BN] → + → ReLU
 \_____________________________________/
            (shortcut)
```

**Residual Block:**
```python
def residual_block(x, filters):
    shortcut = x
    
    # Main path
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x
```

**Full ResNet50 Architecture:**

```
Input (224×224×3)
    ↓
Conv: 64, 7×7, stride=2 → 112×112×64
MaxPool: 3×3, stride=2 → 56×56×64
    ↓
Block 1: 3 residual blocks → 56×56×256
Block 2: 4 residual blocks → 28×28×512
Block 3: 6 residual blocks → 14×14×1024
Block 4: 3 residual blocks → 7×7×2048
    ↓
GlobalAveragePool → 2048
Output: 1000 classes
```

**Parameters:** ~25 million

**Advantages:**
- ✅ Very deep (50-152 layers possible)
- ✅ No vanishing gradient
- ✅ State-of-art accuracy
- ✅ Efficient (fewer parameters than VGG)

**EO Applications:**
- High-accuracy land cover classification
- Complex scene understanding
- Pre-trained on ImageNet → fine-tune on EO

---

### 2.5 U-Net (2015)

**Purpose:** Semantic segmentation (pixel-level classification)

**Architecture:**

```
                    Encoder (Contracting Path)
Input (256×256×1)
    ↓
Conv→Conv → 256×256×64 ──────────────┐
    ↓ MaxPool                        │
Conv→Conv → 128×128×128 ────────┐    │
    ↓ MaxPool                   │    │
Conv→Conv → 64×64×256 ──────┐   │    │
    ↓ MaxPool               │   │    │
Conv→Conv → 32×32×512 ──┐   │   │    │
    ↓ MaxPool           │   │   │    │
Conv→Conv → 16×16×1024  │   │   │    │
                        │   │   │    │
                Decoder (Expanding Path)
                        ↓   │   │    │
UpConv → 32×32×512 ─────┴→ Concat
Conv→Conv → 32×32×512
    ↓
UpConv → 64×64×256 ─────────┴→ Concat
Conv→Conv → 64×64×256
    ↓
UpConv → 128×128×128 ───────────┴→ Concat
Conv→Conv → 128×128×128
    ↓
UpConv → 256×256×64 ─────────────────┴→ Concat
Conv→Conv → 256×256×64
    ↓
Conv 1×1 → 256×256×n_classes
```

**Key Features:**
- **Encoder:** Captures context (what)
- **Decoder:** Enables precise localization (where)
- **Skip connections:** Combines low-level and high-level features

**Parameters:** ~31 million (original)

**EO Applications:**
- Building footprint extraction
- Agricultural field boundaries
- Forest/non-forest segmentation
- Flood extent mapping
- Road network extraction

**Philippine Use Cases:**
- Informal settlement mapping
- Mangrove boundary delineation
- Rice field segmentation
- Post-disaster damage assessment

---

## <a name="designing-for-eo"></a>3. Designing CNNs for EO

### 3.1 Input Considerations

**Sentinel-2 Specifics:**

| Aspect | Natural Images (ImageNet) | Sentinel-2 |
|--------|---------------------------|------------|
| **Channels** | 3 (RGB) | 10-13 bands |
| **Bit Depth** | 8-bit (0-255) | 12-bit (0-4095) or float |
| **Spatial Resolution** | Variable | 10m, 20m, 60m |
| **Typical Patch Size** | 224×224, 299×299 | 64×64, 128×128, 256×256 |
| **Normalization** | ImageNet mean/std | Per-band or global |

**Adaptation Strategies:**

1. **Channel Handling:**
```python
# Option 1: Use all 10 bands
input_shape = (64, 64, 10)

# Option 2: Select key bands (RGB + NIR)
selected_bands = [2, 3, 4, 8]  # Blue, Green, Red, NIR
input_shape = (64, 64, 4)

# Option 3: Add derived indices
# [Bands + NDVI + NDWI + ...]
input_shape = (64, 64, 13)
```

2. **First Layer Design:**
```python
# Large receptive field for multispectral
Conv2D(32, kernel_size=(5, 5), input_shape=(64, 64, 10))

# Or standard 3×3
Conv2D(64, kernel_size=(3, 3), input_shape=(64, 64, 10))
```

---

### 3.2 Architecture Design Patterns

#### Pattern 1: Simple CNN for Scene Classification

**Task:** Classify 64×64 patches into land cover classes

```python
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64,64,10)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    # Classifier
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # 8 land cover classes
])
```

**Parameters:** ~500K  
**Training time:** 2-4 hours (GPU)  
**Accuracy:** 85-92% (with good data)

---

#### Pattern 2: Transfer Learning with ResNet50

**Task:** Fine-tune pre-trained model for Philippine land cover

```python
from tensorflow.keras.applications import ResNet50

# Load pre-trained ResNet (without top)
base_model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)  # Note: 3 channels for ImageNet
)

# Freeze base model
base_model.trainable = False

# Add custom head
model = Sequential([
    # Adapter for 10 Sentinel-2 bands → 3 channels
    Conv2D(3, (1,1), input_shape=(224, 224, 10)),
    
    # Pre-trained ResNet
    base_model,
    
    # Custom classifier
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

# Train top layers first
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)

# Fine-tune: unfreeze last few layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Retrain with lower learning rate
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=20)
```

**Advantages:**
- ✅ Faster convergence
- ✅ Better accuracy with less data
- ✅ Leverages ImageNet knowledge

---

#### Pattern 3: U-Net for Segmentation

**Task:** Pixel-level forest classification

```python
def unet_model(input_shape=(256, 256, 10), n_classes=2):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = UpSampling2D((2,2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(c5)
    
    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(c7)
    
    # Output
    outputs = Conv2D(n_classes, (1,1), activation='softmax')(c7)
    
    return Model(inputs, outputs)
```

---

### 3.3 Design Heuristics

**Rule of Thumb:**

| Aspect | Guideline |
|--------|-----------|
| **First Conv Layer** | 32-64 filters, 3×3 or 5×5 |
| **Filter Doubling** | Double filters after each pool (32→64→128→256) |
| **Pool Frequency** | Every 2-3 conv layers |
| **Dense Layer Size** | 128-512 neurons |
| **Dropout Rate** | 0.25 after conv, 0.5 after dense |
| **Batch Size** | 16-64 for training |
| **Learning Rate** | 0.001 (Adam optimizer) |

**Total Parameters Budget:**

| Dataset Size | Max Parameters | Rationale |
|--------------|----------------|-----------|
| <1K images | <100K | High overfitting risk |
| 1K-10K | 100K-1M | Moderate capacity |
| 10K-100K | 1M-10M | Standard CNNs |
| >100K | 10M+ | Can use large models |

---

## <a name="parameter-calculations"></a>4. Parameter Calculations

### 4.1 Convolutional Layer

```
params = (kernel_h × kernel_w × in_channels + 1) × out_channels
```

**Example:**
```python
Conv2D(filters=64, kernel_size=(3,3), input_shape=(64, 64, 10))

params = (3 × 3 × 10 + 1) × 64 = 91 × 64 = 5,824
```

---

### 4.2 Dense Layer

```
params = (input_size + 1) × output_size
```

**Example:**
```python
Dense(256, input_shape=(512,))

params = (512 + 1) × 256 = 131,328
```

---

### 4.3 Full Network Example

**Architecture:**
```
Input: 64×64×10
Conv1: 32 filters, 3×3 → 64×64×32
MaxPool: 2×2 → 32×32×32
Conv2: 64 filters, 3×3 → 32×32×64
MaxPool: 2×2 → 16×16×64
GlobalAvgPool → 64
Dense: 8 outputs
```

**Calculations:**
```
Conv1: (3×3×10 + 1) × 32 = 2,912
Conv2: (3×3×32 + 1) × 64 = 18,496
Dense: (64 + 1) × 8 = 520

Total: 21,928 parameters
```

---

## <a name="architecture-patterns"></a>5. Architecture Patterns

### 5.1 Increasing Depth

**Shallow (3-5 layers):**
- Fast training
- Limited capacity
- Good for simple tasks

**Medium (10-20 layers):**
- Balanced performance
- Most common choice
- VGG-like

**Deep (50+ layers):**
- Best accuracy
- Requires skip connections (ResNet)
- Longer training

---

### 5.2 Width vs Depth

**Wider (more filters per layer):**
- Captures more diverse features
- More parameters
- Example: 128→256→512 filters

**Deeper (more layers):**
- More abstract features
- Skip connections needed
- Example: ResNet, DenseNet

**Best Practice:** Balance both

---

## <a name="philippine-applications"></a>6. Philippine EO Applications

### 6.1 Mangrove Mapping

**Task:** Segment mangrove forests along coastlines

**Architecture:** U-Net
- Input: 256×256×10 (Sentinel-2)
- Output: Binary mask (mangrove/non-mangrove)
- Augmentation: Flip, rotate (coastal patterns)

**Dataset:**
- ~500-1000 labeled patches
- Palawan, Mindanao, Luzon coasts
- Mixed with non-mangrove

**Performance:** 92-95% IoU

---

### 6.2 Rice Paddy Detection

**Task:** Classify agricultural areas as rice/non-rice

**Architecture:** ResNet50 transfer learning
- Input: 128×128×4 (RGB + NIR)
- Temporal: Dry vs wet season
- Output: Binary classification

**Dataset:**
- ~2000 patches from Central Luzon
- Phenology-aware sampling

**Performance:** 93-96% accuracy

---

### 6.3 Informal Settlement Mapping

**Task:** Detect urban informal settlements

**Architecture:** Object detection (RetinaNet)
- Input: 512×512×3 (High-res RGB)
- Output: Bounding boxes + confidence
- Multi-scale detection

**Dataset:**
- Metro Manila, Cebu, Davao
- ~5000 annotated structures

**Performance:** 85-90% mAP

---

## Summary Table

| Architecture | Parameters | Best For | Training Time | Accuracy |
|--------------|------------|----------|---------------|----------|
| **Simple CNN** | 100K-1M | Scene classification | 2-4 hours | 85-92% |
| **VGG16** | 138M | Transfer learning base | 12-24 hours | 90-95% |
| **ResNet50** | 25M | Complex tasks, fine-tuning | 8-16 hours | 92-97% |
| **U-Net** | 31M | Segmentation | 10-20 hours | 90-95% IoU |
| **MobileNet** | 4M | Edge deployment | 4-8 hours | 88-93% |

---

## Recommended Reading

- **Papers:**
  - LeCun et al. (1998) - Gradient-Based Learning
  - Krizhevsky et al. (2012) - AlexNet (ImageNet Classification)
  - Simonyan & Zisserman (2014) - VGG
  - He et al. (2016) - ResNet (Deep Residual Learning)
  - Ronneberger et al. (2015) - U-Net

- **EO-Specific:**
  - Helber et al. (2019) - EuroSAT Dataset
  - Zhang et al. (2020) - Remote Sensing Deep Learning Survey

---

**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 3
