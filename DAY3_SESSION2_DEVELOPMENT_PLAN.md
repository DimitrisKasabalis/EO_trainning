# Day 3 Session 2 - Development Plan
**Session:** Hands-on Flood Mapping with U-Net and Sentinel-1 SAR  
**Duration:** 2.5 hours  
**Format:** Hands-on Lab  
**Date:** October 15, 2025

---

## Session Overview

**Case Study:** Flood Mapping in Central Luzon (Pampanga River Basin)  
**Typhoon Event:** Ulysses (2020) or Karding (2022)  
**Platform:** Google Colab with GPU acceleration  
**Framework:** TensorFlow/Keras or PyTorch (consistent with Day 2)

---

## Deliverables to Create

### 1. **Quarto Session Page** (`session2.qmd`)
- Lab introduction and learning objectives
- Dataset description
- Step-by-step workflow explanation
- Embedded or linked Jupyter notebook
- Troubleshooting guide
- Expected results and interpretation
- Resources and next steps

### 2. **Jupyter Notebook** (`Day3_Session2_Flood_Mapping_UNet.ipynb`)
- Complete executable notebook
- Pre-filled code cells
- Markdown explanations
- Visualization outputs
- Exercise sections for students
- Solutions (optional separate file)

### 3. **Dataset Documentation** (`DATASET_README.md`)
- Data source and preparation details
- File structure and formats
- Loading instructions
- Data characteristics
- Citation and credits

### 4. **Supporting Materials**
- Sample data (or download instructions)
- Pre-trained model weights (optional)
- Troubleshooting FAQ
- Additional resources

---

## Session 2 Content Structure

### **Part 1: Introduction (15 min)**
- Lab objectives and expected outcomes
- Philippine context: Central Luzon flood risk
- Dataset overview and preparation notes
- Platform setup (Colab, GPU check)

### **Part 2: Data Loading and Exploration (20 min)**
- Load Sentinel-1 SAR patches
- Load binary flood masks
- Visualize sample data
- Understand SAR backscatter values
- Data statistics and distribution

### **Part 3: Data Preprocessing (20 min)**
- Normalization strategies for SAR data
- Train/validation/test split
- Data augmentation (rotation, flip)
- Creating TensorFlow/PyTorch datasets
- Batch preparation

### **Part 4: U-Net Model Implementation (30 min)**
- Define U-Net architecture
- Encoder block implementation
- Decoder block implementation
- Skip connections
- Final output layer (sigmoid for binary)
- Model summary and parameter count

### **Part 5: Model Compilation and Training (30 min)**
- Loss function selection (Dice or Combined)
- Optimizer configuration (Adam)
- Metrics (IoU, Dice, accuracy)
- Callbacks (ModelCheckpoint, EarlyStopping, TensorBoard)
- Training loop execution
- Monitoring training/validation curves

### **Part 6: Model Evaluation (20 min)**
- Load best model weights
- Evaluate on test set
- Calculate IoU, F1-score, precision, recall
- Confusion matrix
- Per-class metrics

### **Part 7: Visualization and Interpretation (15 min)**
- Predict on test images
- Visualize: SAR image | Ground truth | Prediction
- Overlay predictions on original images
- Identify successes and failures
- Error analysis

### **Part 8: Export and Integration (10 min)**
- Export predictions as GeoTIFF (if georeferenced)
- Save model for future use
- Create flood extent polygons
- Prepare for GIS integration

---

## Dataset Specifications

### **Input Data: Sentinel-1 SAR**

**File Format:** NumPy arrays (`.npy`) or GeoTIFF (`.tif`)

**Structure:**
```
flood_mapping_dataset/
├── train/
│   ├── images/
│   │   ├── patch_001.npy  # Shape: (256, 256, 2) - VV, VH
│   │   ├── patch_002.npy
│   │   └── ...
│   └── masks/
│       ├── mask_001.npy   # Shape: (256, 256, 1) - Binary 0/1
│       ├── mask_002.npy
│       └── ...
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
└── metadata.json
```

**Characteristics:**
- **Patch size:** 256×256 pixels
- **Bands:** 2 (VV and VH polarizations)
- **Number of patches:** 
  - Train: ~600-800
  - Validation: ~150-200
  - Test: ~150-200
- **Data type:** Float32 (dB values, typically -30 to 10)
- **Labels:** Binary (0 = non-flood, 1 = flood)
- **Class distribution:** Imbalanced (~5-15% flood pixels)

**Preprocessing Applied:**
- Speckle filtering (Lee filter or similar)
- Radiometric calibration to dB
- Geometric terrain correction
- Resampling to 10m resolution
- Patch extraction with overlap

---

## Notebook Structure

### **Section 1: Setup and Imports**
```python
# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Deep learning framework (TensorFlow example)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Metrics and utilities
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```

### **Section 2: Configuration**
```python
# Paths
DATA_DIR = '/content/drive/MyDrive/flood_mapping_dataset'
MODEL_DIR = '/content/models'
OUTPUT_DIR = '/content/outputs'

# Hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 2  # VV, VH
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# Class weights for imbalance
CLASS_WEIGHTS = {0: 1.0, 1: 10.0}  # Adjust based on actual distribution
```

### **Section 3: Data Loading Functions**
```python
def load_sar_data(data_dir, subset='train'):
    """Load SAR images and masks from directory"""
    pass

def normalize_sar(image):
    """Normalize SAR backscatter values"""
    pass

def create_tf_dataset(images, masks, batch_size, augment=False):
    """Create TensorFlow dataset with augmentation"""
    pass
```

### **Section 4: Data Exploration**
```python
# Visualize sample patches
# Plot histograms of VV, VH values
# Show class distribution
# Display augmented examples
```

### **Section 5: U-Net Architecture**
```python
def unet_model(input_shape=(256, 256, 2), num_classes=1):
    """Build U-Net architecture"""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    # ... (detailed implementation)
    
    # Bottleneck
    # ...
    
    # Decoder with skip connections
    # ...
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='unet')
```

### **Section 6: Loss Functions**
```python
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for binary segmentation"""
    pass

def combined_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy + Dice Loss"""
    pass
```

### **Section 7: Training**
```python
# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=combined_loss,
    metrics=['accuracy', dice_coefficient, iou_score]
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(...),
    keras.callbacks.EarlyStopping(...),
    keras.callbacks.ReduceLROnPlateau(...),
    keras.callbacks.TensorBoard(...)
]

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)
```

### **Section 8: Evaluation and Visualization**
```python
# Load best model
# Predict on test set
# Calculate metrics
# Plot confusion matrix
# Visualize predictions
# Error analysis
```

---

## Key Conceptual Hurdles to Address

### 1. **SAR Data Understanding**
- Explain backscatter values (dB scale)
- VV vs VH polarization significance
- Why water appears dark in SAR (low backscatter)
- Speckle noise characteristics

### 2. **U-Net Implementation Details**
- How skip connections are coded (concatenate layers)
- Transpose convolution vs upsampling
- Output activation (sigmoid for binary, softmax for multi-class)
- Why same padding is crucial

### 3. **Training Challenges**
- Class imbalance handling (weighted loss, Dice loss)
- Overfitting prevention (dropout, batch norm, early stopping)
- Learning rate scheduling
- How to interpret training curves

### 4. **Evaluation Interpretation**
- What does IoU=0.7 mean in practice?
- Precision vs Recall trade-off for flood mapping
- False positives (water classified as land) vs False negatives (land as water)
- Spatial patterns in errors

---

## Expected Learning Outcomes

By the end of Session 2, students will be able to:

1. **Load and preprocess** Sentinel-1 SAR data for segmentation
2. **Implement** U-Net architecture in TensorFlow/PyTorch
3. **Train** a segmentation model with appropriate loss functions
4. **Evaluate** model performance using IoU, Dice, and confusion matrix
5. **Visualize** flood extent predictions overlaid on SAR imagery
6. **Interpret** model results and identify limitations
7. **Export** predictions for GIS integration

---

## Troubleshooting Guide Topics

### Common Issues:
1. **Out of Memory (OOM) errors**
   - Reduce batch size
   - Use smaller patches
   - Enable mixed precision training

2. **Model not learning (loss plateau)**
   - Check data normalization
   - Verify labels are correct
   - Adjust learning rate
   - Check for data leakage

3. **Poor generalization (high train, low val accuracy)**
   - Increase augmentation
   - Add regularization (dropout)
   - Reduce model complexity
   - Get more diverse training data

4. **Predictions all black or all white**
   - Check output activation function
   - Verify loss function handles class imbalance
   - Inspect data distribution

5. **Colab disconnections**
   - Save checkpoints frequently
   - Use Google Drive for data persistence
   - Keep browser tab active

---

## Resources to Include

### Datasets:
- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11
- NASA Flood datasets
- Local Philippine flood datasets (if available)

### Code Repositories:
- U-Net implementations in TensorFlow/PyTorch
- SAR preprocessing pipelines
- Segmentation evaluation utilities

### Papers and Tutorials:
- U-Net paper (Ronneberger et al. 2015)
- SAR flood mapping papers
- TensorFlow segmentation tutorials

### Philippine Context:
- PAGASA flood bulletins
- PhilSA disaster response data
- DOST-ASTI DATOS rapid mapping

---

## Next Steps

1. **Create session2.qmd** - Quarto page with lab instructions
2. **Develop Jupyter notebook** - Complete executable code
3. **Prepare sample dataset** - Or provide download instructions
4. **Test notebook** - Execute end-to-end to verify
5. **Create troubleshooting FAQ** - Based on testing
6. **Peer review** - Have someone test the notebook

---

## Timeline

**Immediate (Today):**
- [ ] Create session2.qmd Quarto page
- [ ] Outline complete Jupyter notebook structure

**This Week:**
- [ ] Implement full Jupyter notebook with code
- [ ] Test notebook execution
- [ ] Create dataset documentation

**Before Deployment:**
- [ ] Verify dataset availability
- [ ] Test on fresh Colab instance
- [ ] Peer review and feedback
- [ ] Finalize troubleshooting guide

---

**Status:** Ready to begin development  
**Priority:** HIGH - Session 2 is critical hands-on component  
**Dependencies:** None (Session 1 complete)
