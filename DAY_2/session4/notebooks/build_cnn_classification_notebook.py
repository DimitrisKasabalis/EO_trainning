"""
Build Session 4 CNN Classification Notebook (Part A)
Comprehensive hands-on lab for TensorFlow/Keras CNN implementation
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =============================================================================
# TITLE AND INTRODUCTION
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""# Session 4: CNN Hands-On Lab - EuroSAT Classification

## Building CNNs with TensorFlow/Keras

**Duration:** 90 minutes | **Difficulty:** Intermediate  
**Dataset:** EuroSAT (27,000 Sentinel-2 images, 10 classes)

---

## 🎯 Objectives

By the end of this lab, you will:

1. ✅ Build a CNN from scratch using TensorFlow/Keras
2. ✅ Train on real satellite imagery (EuroSAT dataset)
3. ✅ Achieve >90% accuracy on land use classification
4. ✅ Evaluate model performance comprehensively
5. ✅ Understand training dynamics and hyperparameters
6. ✅ Apply data augmentation for better generalization

---

## 📋 Lab Structure

| Step | Activity | Duration |
|------|----------|----------|
| **1** | Environment Setup & GPU Check | 5 min |
| **2** | Dataset Download & Exploration | 15 min |
| **3** | Data Preprocessing & Augmentation | 15 min |
| **4** | Build CNN Architecture | 20 min |
| **5** | Training & Monitoring | 20 min |
| **6** | Evaluation & Analysis | 15 min |

---

## 🌍 About EuroSAT Dataset

**What:** Benchmark dataset for satellite image classification  
**Source:** Sentinel-2 RGB and multi-spectral  
**Images:** 27,000 labeled patches (64×64 pixels)  
**Classes:** 10 land use/land cover types  
**Purpose:** Standardized comparison of classification methods

**Classes:**
1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. SeaLake

---

Let's build your first CNN! 🚀
"""))

# =============================================================================
# STEP 1: ENVIRONMENT SETUP
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 1: Environment Setup (5 minutes)

First, let's import libraries and check if GPU is available.
"""))

cells.append(nbf.v4.new_code_cell("""# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# scikit-learn for metrics
from sklearn.metrics import classification_report, confusion_matrix

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ Keras version: {keras.__version__}")
print(f"✓ NumPy version: {np.__version__}")"""))

cells.append(nbf.v4.new_markdown_cell("""### Check GPU Availability

GPUs dramatically speed up training. Let's check if one is available.
"""))

cells.append(nbf.v4.new_code_cell("""# Check for GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\\n✓ GPU(s) Available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    
    # Enable memory growth (prevents TensorFlow from allocating all GPU memory)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\\n✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("\\n⚠️  No GPU found - training will use CPU (slower)")
    print("   Consider using Google Colab with GPU runtime")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\\n✓ Environment ready!")"""))

# =============================================================================
# STEP 2: DATASET DOWNLOAD
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 2: Dataset Download & Exploration (15 minutes)

We'll download the EuroSAT RGB dataset from TensorFlow Datasets.
"""))

cells.append(nbf.v4.new_code_cell("""# Option 1: Using TensorFlow Datasets (easiest)
# If this doesn't work, we'll provide manual download instructions

try:
    import tensorflow_datasets as tfds
    print("✓ TensorFlow Datasets available")
    USE_TFDS = True
except ImportError:
    print("⚠️  TensorFlow Datasets not installed")
    print("   Installing: pip install tensorflow-datasets")
    USE_TFDS = False"""))

cells.append(nbf.v4.new_code_cell("""# Download EuroSAT dataset
if USE_TFDS:
    print("Downloading EuroSAT RGB dataset...")
    print("(This may take a few minutes on first run - ~90MB)")
    
    # Load dataset
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'eurosat/rgb',
        split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
        as_supervised=True,
        with_info=True
    )
    
    print(f"\\n✓ Dataset loaded successfully!")
    print(f"  Total images: {ds_info.splits['train'].num_examples}")
    print(f"  Train split: 70% ({ds_train.cardinality().numpy()} images)")
    print(f"  Val split: 15% ({ds_val.cardinality().numpy()} images)")
    print(f"  Test split: 15% ({ds_test.cardinality().numpy()} images)")
    
    # Class names
    class_names = ds_info.features['label'].names
    num_classes = len(class_names)
    
    print(f"\\n  Classes ({num_classes}):")
    for i, name in enumerate(class_names):
        print(f"    {i}: {name}")
else:
    print("Manual dataset download required")
    print("See: https://github.com/phelber/EuroSAT")"""))

cells.append(nbf.v4.new_markdown_cell("""### Explore Dataset"""))

cells.append(nbf.v4.new_code_cell("""# Visualize sample images
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

# Take 20 samples
sample_images = list(ds_train.take(20))

for idx, (image, label) in enumerate(sample_images):
    ax = axes[idx]
    
    # Display image
    ax.imshow(image.numpy())
    ax.set_title(f"{class_names[label.numpy()]}\\n(Class {label.numpy()})",
                 fontsize=10, fontweight='bold')
    ax.axis('off')

plt.suptitle('EuroSAT Sample Images', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\\n✓ Dataset looks good!")
print("  Notice the variety of land use patterns")"""))

# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 3: Data Preprocessing & Augmentation (15 minutes)

We need to:
1. Normalize pixel values (0-255 → 0-1)
2. Batch the data for efficient training
3. Apply data augmentation to prevent overfitting
"""))

cells.append(nbf.v4.new_code_cell("""# Preprocessing function
def preprocess(image, label):
    \"\"\"
    Normalize image to [0, 1] range
    \"\"\"
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

print("✓ Images normalized to [0, 1]")"""))

cells.append(nbf.v4.new_markdown_cell("""### Data Augmentation

Augmentation creates variations of training images to improve generalization.

**Techniques we'll use:**
- Random horizontal flips
- Random vertical flips
- Random rotations (90°, 180°, 270°)
- Random brightness adjustment

**Why safe for satellite imagery:**
- Land use patterns have no preferred orientation
- Small brightness variations are realistic
"""))

cells.append(nbf.v4.new_code_cell("""# Data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),  # Up to 90 degrees
    layers.RandomBrightness(0.1),  # ±10% brightness
    layers.RandomContrast(0.1),
], name='data_augmentation')

# Augmentation function
def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label

# Apply augmentation to training data only
ds_train_augmented = ds_train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

print("✓ Data augmentation configured")
print("  Augmentation applied to training data only")
print("  Validation and test data unchanged (for fair evaluation)")"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualize Augmentation"""))

cells.append(nbf.v4.new_code_cell("""# Show original vs augmented
sample_image, sample_label = next(iter(ds_train))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Original
axes[0, 0].imshow(sample_image.numpy())
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')

# Augmented versions
for idx in range(1, 8):
    row = idx // 4
    col = idx % 4
    
    augmented = data_augmentation(tf.expand_dims(sample_image, 0), training=True)[0]
    axes[row, col].imshow(augmented.numpy())
    axes[row, col].set_title(f'Augmented {idx}', fontweight='bold')
    axes[row, col].axis('off')

plt.suptitle(f'Data Augmentation Examples\\nClass: {class_names[sample_label.numpy()]}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\n✓ Augmentation creates realistic variations")"""))

cells.append(nbf.v4.new_markdown_cell("""### Create Batched Datasets"""))

cells.append(nbf.v4.new_code_cell("""# Configuration
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000

# Batch and prefetch for performance
ds_train_final = ds_train_augmented.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val_final = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test_final = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"✓ Datasets configured")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(list(ds_train_final))}")
print(f"  Val batches: {len(list(ds_val_final))}")
print(f"  Test batches: {len(list(ds_test_final))}")"""))

# Continue with notebook...
nb['cells'] = cells

# Write notebook
output_path = 'session4_cnn_classification_STUDENT.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"\\n✓ Session 4 Notebook (Part 1) Created!")
print(f"  Total cells so far: {len(cells)}")
print(f"  File: {output_path}")
print("\\nNext: Adding CNN architecture, training, and evaluation...")
