# Session 4 CNN Notebook - Test Execution Guide

**Test Date:** October 15, 2025  
**Notebook:** `session4_cnn_classification_STUDENT.ipynb`  
**Cells:** 45 total (23 code, 22 markdown)  
**Expected Duration:** 90-120 minutes  
**Critical Requirements:** GPU, Internet connection

---

## 🚀 Pre-Test Setup (5 minutes)

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google account
3. File → Upload notebook
4. Navigate to: `/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site/day2/notebooks/session4_cnn_classification_STUDENT.ipynb`

**OR** use direct path:
```
File → Open notebook → Upload → Select file
```

### Step 2: Enable GPU Runtime (CRITICAL)
1. **Runtime** → **Change runtime type**
2. **Hardware accelerator:** Select **GPU** (not None, not TPU)
3. **GPU type:** T4 (default for free Colab) or P100/V100 if available
4. Click **Save**
5. Runtime will restart

### Step 3: Verify GPU is Enabled
Run this verification cell:
```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
```

**Expected Output:**
```
TensorFlow version: 2.x.x
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Num GPUs: 1
```

⚠️ **If GPU shows as empty list `[]`:**
- Go back to Runtime → Change runtime type
- Ensure GPU is selected
- Restart runtime
- Try again

---

## 📝 Section-by-Section Test Execution

### PART 1: Environment Setup & GPU Check (Expected: 5 min)

#### Cell 1-3: Imports and Setup
**What to check:**
- ✅ All imports execute without errors
- ✅ No version conflicts
- ✅ GPU detected

**Common issues:**
- ImportError for tensorflow → Already installed in Colab
- Module not found errors → Run `!pip install <package>`

**Document:**
```
Cell 1-3 Status: PASS / FAIL
GPU Detected: YES / NO
TensorFlow Version: _______
Issues: _______
```

---

### PART 2: Dataset Download & Exploration (Expected: 15 min)

#### Cells 4-8: EuroSAT Download

**What happens:**
- Downloads ~90 MB dataset
- Extracts to `/content/eurosat/`
- Verifies 10 class directories exist

**What to check:**
- ✅ Download completes successfully (watch progress bar)
- ✅ Extraction successful
- ✅ All 10 class directories created
- ✅ Total images: 27,000

**Time tracking:**
- Download time: _______ minutes (Expected: 2-5 min depending on connection)
- Extraction time: _______ seconds

**If download fails:**
```python
# Alternative download method
!gdown --id 1234567890abcdef  # Use alternative link
# OR
!wget https://alternative-mirror.com/eurosat.zip
```

**Document:**
```
Download Status: PASS / FAIL
Download Time: _______ min
Total Images Found: _______
All 10 Classes: YES / NO
Issues: _______
```

#### Cells 9-12: Data Exploration

**What to check:**
- ✅ Sample images display correctly (one from each class)
- ✅ Class distribution bar chart renders
- ✅ Dataset statistics calculated
- ✅ Images are 64×64×3 (RGB)

**Expected class distribution:**
Each class should have ~2,700 images (balanced dataset)

**Document:**
```
Visualization Status: PASS / FAIL
Images Display: YES / NO
Class Balance: BALANCED / IMBALANCED
Issues: _______
```

---

### PART 3: Data Preprocessing & Augmentation (Expected: 15 min)

#### Cells 13-16: Train/Val/Test Split

**What to check:**
- ✅ Split ratios correct (70% train, 15% val, 15% test)
- ✅ Stratified split (each class proportionally represented)
- ✅ No data leakage between sets

**Expected numbers:**
- Training: ~18,900 images
- Validation: ~4,050 images
- Test: ~4,050 images

**Document:**
```
Split Status: PASS / FAIL
Training Set: _______ images (Expected: ~18,900)
Validation Set: _______ images (Expected: ~4,050)
Test Set: _______ images (Expected: ~4,050)
Stratified: YES / NO
Issues: _______
```

#### Cells 17-20: Data Pipeline Creation

**What to check:**
- ✅ `tf.data.Dataset` created successfully
- ✅ Normalization applied (pixel values 0-1)
- ✅ Batching configured (batch_size=32)
- ✅ Shuffling enabled for training
- ✅ Prefetching for performance

**Verify batch shape:**
```python
for images, labels in train_dataset.take(1):
    print(f"Batch shape: {images.shape}")  # Should be (32, 64, 64, 3)
    print(f"Label shape: {labels.shape}")   # Should be (32, 10) - one-hot
```

**Document:**
```
Pipeline Status: PASS / FAIL
Batch Shape: _______
Batches per Epoch: _______
Issues: _______
```

#### Cells 21-23: Data Augmentation

**What to check:**
- ✅ RandomFlip (horizontal)
- ✅ RandomRotation (small angles)
- ✅ RandomZoom (slight zoom)
- ✅ Augmented examples display correctly

**Document:**
```
Augmentation Status: PASS / FAIL
Augmented Images Display: YES / NO
Issues: _______
```

---

### PART 4: Build CNN Architecture (Expected: 20 min)

#### Cells 24-27: Model Definition

**What to check:**
- ✅ Architecture defined correctly
- ✅ Model compiles without errors
- ✅ Model summary displays

**Expected architecture:**
```
Model: "eurosat_cnn"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)          (None, 62, 62, 32)        896       
activation_1 (ReLU)        (None, 62, 62, 32)        0         
max_pooling2d_1            (None, 31, 31, 32)        0         
conv2d_2 (Conv2D)          (None, 29, 29, 64)        18,496    
activation_2 (ReLU)        (None, 29, 29, 64)        0         
max_pooling2d_2            (None, 14, 14, 64)        0         
conv2d_3 (Conv2D)          (None, 12, 12, 128)       73,856    
activation_3 (ReLU)        (None, 12, 12, 128)       0         
max_pooling2d_3            (None, 6, 6, 128)         0         
flatten (Flatten)          (None, 4608)              0         
dropout (Dropout)          (None, 4608)              0         
dense_1 (Dense)            (None, 128)               589,952   
activation_4 (ReLU)        (None, 128)               0         
dropout_2 (Dropout)        (None, 128)               0         
dense_2 (Dense)            (None, 10)                1,290     
activation_5 (Softmax)     (None, 10)                0         
=================================================================
Total params: 684,490
Trainable params: 684,490
Non-trainable params: 0
```

**Verify:**
- Total parameters: ~684,490
- All layers present
- Output shape: (None, 10)

**Document:**
```
Model Build Status: PASS / FAIL
Total Parameters: _______
Trainable Parameters: _______
Output Classes: _______
Issues: _______
```

#### Cells 28-29: Model Compilation

**What to check:**
- ✅ Loss: categorical_crossentropy
- ✅ Optimizer: Adam
- ✅ Metrics: accuracy
- ✅ Learning rate: 0.001 (default)

**Document:**
```
Compilation Status: PASS / FAIL
Loss Function: _______
Optimizer: _______
Learning Rate: _______
Issues: _______
```

---

### PART 5: Training & Monitoring (Expected: 20 min) ⏱️ CRITICAL SECTION

#### Cells 30-32: Callback Configuration

**What to check:**
- ✅ EarlyStopping configured (patience=5, monitor='val_loss')
- ✅ ModelCheckpoint configured (save_best_only=True)
- ✅ ReduceLROnPlateau configured (factor=0.5, patience=3)

**Document:**
```
Callbacks Status: PASS / FAIL
EarlyStopping: CONFIGURED / NOT CONFIGURED
ModelCheckpoint: CONFIGURED / NOT CONFIGURED
ReduceLROnPlateau: CONFIGURED / NOT CONFIGURED
Issues: _______
```

#### Cell 33-34: MODEL TRAINING 🚨 MAIN TEST

**What happens:**
- Training begins (20-30 epochs expected, may stop early)
- Each epoch takes ~30-60 seconds with GPU (10-15 min total)
- Progress bar shows loss and accuracy

**WATCH FOR:**

**Epoch 1:**
```
Epoch 1/30
590/590 [==============================] - 45s 76ms/step - 
loss: 1.8234 - accuracy: 0.3456 - val_loss: 1.4532 - val_accuracy: 0.5234
```
- Initial accuracy: 30-40% (random is 10%)
- Loss should be decreasing

**Epoch 5-10:**
```
Epoch 10/30
590/590 [==============================] - 42s 71ms/step - 
loss: 0.4321 - accuracy: 0.8567 - val_loss: 0.3987 - val_accuracy: 0.8756
```
- Accuracy climbing: 80-90%
- Train/val gap should be small (<5%)

**Final Epoch (15-25):**
```
Epoch 18/30
590/590 [==============================] - 41s 69ms/step - 
loss: 0.1234 - accuracy: 0.9523 - val_loss: 0.2345 - val_accuracy: 0.9234
Restoring model weights from the end of the best epoch: 13.
```
- Training accuracy: 93-98%
- Validation accuracy: 90-95% ✅ **TARGET: >90%**
- Early stopping triggered

**TIME TRACKING:**
- Start time: _______
- End time: _______
- Total training time: _______ minutes (Expected: 10-15 min with GPU)
- Epochs completed: _______

**GPU UTILIZATION:**
- Check GPU usage during training:
  - Runtime → Manage sessions → View details
  - GPU memory used: _______ GB / 15 GB

**Document:**
```
Training Status: PASS / FAIL / IN PROGRESS
Epochs Completed: _______
Training Time: _______ minutes
Final Training Accuracy: _______ %
Final Validation Accuracy: _______ % (TARGET: >90%)
Best Epoch: _______
Early Stopping: TRIGGERED / NOT TRIGGERED
GPU Used: YES / NO
GPU Memory: _______ GB
Issues: _______
```

**RED FLAGS:**
- ❌ Training accuracy stuck at 10% (random guessing) → Model not learning
- ❌ Loss = NaN → Exploding gradients, reduce learning rate
- ❌ Validation accuracy << training accuracy → Overfitting
- ❌ Training very slow (>5 min/epoch) → GPU not enabled
- ❌ Out of memory error → Reduce batch size to 16

#### Cells 35-36: Learning Curves

**What to check:**
- ✅ Training/validation loss plot displays
- ✅ Training/validation accuracy plot displays
- ✅ Curves show convergence
- ✅ No severe overfitting (curves should be close)

**Healthy curves:**
- Loss: Both decrease, converge, small gap
- Accuracy: Both increase, plateau, small gap

**Document:**
```
Learning Curves Status: PASS / FAIL
Loss Curve: HEALTHY / OVERFITTING / UNDERFITTING
Accuracy Curve: HEALTHY / OVERFITTING / UNDERFITTING
Convergence: YES / NO
Issues: _______
```

---

### PART 6: Evaluation & Analysis (Expected: 15 min)

#### Cells 37-38: Test Set Evaluation

**What to check:**
- ✅ Best model loaded
- ✅ Test accuracy calculated
- ✅ Test loss calculated

**Expected results:**
- Test Accuracy: 90-95%
- Test Loss: 0.2-0.4
- Should be similar to validation accuracy

**Document:**
```
Test Evaluation Status: PASS / FAIL
Test Accuracy: _______ % (TARGET: >90%)
Test Loss: _______
Similar to Validation: YES / NO
Issues: _______
```

#### Cells 39-40: Confusion Matrix

**What to check:**
- ✅ 10×10 confusion matrix generated
- ✅ Heatmap visualization displays
- ✅ Diagonal is darkest (correct predictions)
- ✅ Off-diagonal shows confusions

**Analyze confusions:**
- Which classes are most confused?
- Is there a pattern (e.g., vegetation classes)?

**Document:**
```
Confusion Matrix Status: PASS / FAIL
Matrix Generated: YES / NO
Visualization: CLEAR / UNCLEAR
Most Confused Pair: _______ ↔ _______
Issues: _______
```

#### Cells 41-42: Per-Class Metrics

**What to check:**
- ✅ Classification report displays
- ✅ Precision, recall, F1 for all 10 classes
- ✅ Macro average ~0.92-0.95
- ✅ Weighted average ~0.92-0.95

**Expected:**
```
              precision    recall  f1-score   support
  AnnualCrop       0.93      0.94      0.93       478
      Forest       0.96      0.97      0.97       507
  Herbaceous       0.91      0.92      0.92       481
  ...
    accuracy                           0.94      4850
```

**Document:**
```
Per-Class Metrics Status: PASS / FAIL
Macro Avg F1: _______
Weighted Avg F1: _______
Lowest Performing Class: _______ (F1: _______)
Highest Performing Class: _______ (F1: _______)
Issues: _______
```

#### Cells 43-45: Error Analysis

**What to check:**
- ✅ Misclassified examples displayed
- ✅ True vs predicted labels shown
- ✅ Patterns visible

**Document:**
```
Error Analysis Status: PASS / FAIL
Misclassifications Displayed: YES / NO
Patterns Identified: _______
Issues: _______
```

---

## 📊 FINAL RESULTS SUMMARY

### Performance Achieved

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Training Accuracy | >90% | ___% | PASS/FAIL |
| Validation Accuracy | >90% | ___% | PASS/FAIL |
| Test Accuracy | >90% | ___% | PASS/FAIL |
| Macro F1-Score | >0.90 | _____ | PASS/FAIL |
| Training Time (GPU) | 10-20 min | ___min | PASS/FAIL |

### Execution Summary

- **Total Cells:** 45
- **Cells Executed:** _______
- **Cells Failed:** _______
- **Total Time:** _______ minutes
- **GPU Used:** YES / NO
- **Notebook Complete:** YES / NO

### Issues Encountered

**Critical Issues (Prevent completion):**
1. _______
2. _______

**Major Issues (Affect performance):**
1. _______
2. _______

**Minor Issues (Cosmetic):**
1. _______
2. _______

---

## ✅ PASS/FAIL Criteria

**PASS if:**
- ✅ All 45 cells execute without critical errors
- ✅ Test accuracy >90%
- ✅ Training completed with GPU
- ✅ All visualizations render
- ✅ Timing reasonable (total <120 min)

**FAIL if:**
- ❌ Test accuracy <85%
- ❌ Multiple cells fail to execute
- ❌ GPU not used (training >60 min)
- ❌ Data download fails

**OVERALL NOTEBOOK STATUS:** ⬜ PASS / ⬜ FAIL / ⬜ PARTIAL

---

## 🔧 Troubleshooting Quick Reference

### GPU Not Detected
```python
# Verify GPU
!nvidia-smi  # Should show GPU info
# If fails: Runtime → Change runtime type → GPU → Save
```

### Download Timeout
```python
# Increase timeout
!wget --timeout=300 --tries=5 <url>
```

### Out of Memory
```python
# Reduce batch size
batch_size = 16  # Instead of 32
```

### Training Not Improving
```python
# Reduce learning rate
optimizer = Adam(learning_rate=0.0001)  # Instead of 0.001
```

### Import Errors
```python
!pip install tensorflow==2.13.0
# Restart runtime after install
```

---

## 📝 Next Steps After Testing

1. **Copy results** to `DAY2_NOTEBOOK_TEST_TEMPLATE.md` Session 4 section
2. **Save notebook** with outputs (File → Download → .ipynb)
3. **Take screenshots** of:
   - GPU verification
   - Final training output
   - Confusion matrix
   - Learning curves
4. **Document issues** in detail
5. **Update Session 4 QMD** troubleshooting if new issues found

---

## 🎯 Expected Completion Checklist

After successful test, you should have:
- ✅ Executed notebook with all outputs visible
- ✅ Test accuracy >90% achieved
- ✅ GPU confirmation screenshot
- ✅ Learning curves saved
- ✅ Confusion matrix image
- ✅ Test results documented in template
- ✅ Issues logged (if any)
- ✅ Recommendations for improvements

---

**Test Started:** _______  
**Test Completed:** _______  
**Tester:** _______  
**Overall Status:** ⬜ PASS / ⬜ FAIL

---

**Ready to test? Open Colab and follow this guide step-by-step! 🚀**
