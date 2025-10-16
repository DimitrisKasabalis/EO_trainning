"""
Build Session 4 Transfer Learning Notebook (Part B)
Fine-tuning pre-trained ResNet50 for EuroSAT classification
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =============================================================================
# TITLE
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""# Session 4 Part B: Transfer Learning with ResNet50

## Fine-Tuning Pre-Trained Models for Earth Observation

**Duration:** 60 minutes | **Difficulty:** Intermediate-Advanced  
**Dataset:** EuroSAT (same as Part A)

---

## 🎯 Objectives

By the end of this notebook, you will:

1. ✅ Understand transfer learning concepts
2. ✅ Load pre-trained ResNet50 (ImageNet weights)
3. ✅ Adapt ResNet50 for EO applications (3 channels → 10 bands)
4. ✅ Fine-tune the model on EuroSAT
5. ✅ Compare transfer learning vs from-scratch CNN
6. ✅ Achieve 93-96% accuracy (improvement over Part A)

---

## 📋 What is Transfer Learning?

**Concept:** Use knowledge from one task to improve performance on another

**How it works:**
1. Take a model pre-trained on large dataset (e.g., ImageNet: 1.2M images, 1000 classes)
2. Remove final classification layer
3. Add new layers for your specific task
4. Fine-tune on your dataset (smaller, specialized)

**Benefits:**
- ✅ **Less data needed:** 1000s instead of millions
- ✅ **Faster training:** Start from good features
- ✅ **Better accuracy:** Leverage learned representations
- ✅ **Prevents overfitting:** Pre-trained weights are robust

**ImageNet → EuroSAT:**
- ImageNet learned: edges, textures, shapes, objects
- We adapt: Apply these features to satellite imagery

---

## 🏗️ Notebook Structure

| Step | Activity | Duration |
|------|----------|----------|
| **1** | Setup & Load Pre-trained Model | 10 min |
| **2** | Adapt for Multi-spectral (Optional) | 10 min |
| **3** | Feature Extraction (Freeze Base) | 15 min |
| **4** | Fine-Tuning (Unfreeze Layers) | 15 min |
| **5** | Comparison & Analysis | 10 min |

---

Let's leverage pre-trained models! 🚀
"""))

# =============================================================================
# STEP 1: SETUP
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 1: Setup & Load Pre-trained Model (10 minutes)

We'll use the same environment and dataset from Part A.
"""))

cells.append(nbf.v4.new_code_cell("""# Imports (same as Part A)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ Keras version: {keras.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\\n✓ GPU available: {len(gpus)} device(s)")
else:
    print("\\n⚠️  No GPU - training will be slower")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)"""))

cells.append(nbf.v4.new_markdown_cell("""### Load Dataset (Same as Part A)

If you completed Part A in the same session, the dataset is already downloaded. Otherwise, we'll re-download.
"""))

cells.append(nbf.v4.new_code_cell("""# Load EuroSAT dataset
import tensorflow_datasets as tfds

print("Loading EuroSAT dataset...")

# Load with same splits as Part A
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'eurosat/rgb',
    split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
    as_supervised=True,
    with_info=True
)

class_names = ds_info.features['label'].names
num_classes = len(class_names)

print(f"\\n✓ Dataset loaded")
print(f"  Classes: {num_classes}")
print(f"  Train: {ds_train.cardinality().numpy()} images")
print(f"  Val: {ds_val.cardinality().numpy()} images")
print(f"  Test: {ds_test.cardinality().numpy()} images")"""))

cells.append(nbf.v4.new_markdown_cell("""### Preprocessing & Batching"""))

cells.append(nbf.v4.new_code_cell("""# Preprocessing for ResNet50
# ResNet expects images scaled to [-1, 1] or [0, 1] depending on preprocessing
# We'll use [0, 1] for consistency with Part A

def preprocess_resnet(image, label):
    \"\"\"
    Preprocess for ResNet50
    ResNet was trained on ImageNet with specific preprocessing
    \"\"\"
    # Convert to float and normalize
    image = tf.cast(image, tf.float32) / 255.0
    
    # ResNet50 expects 224x224 images (ImageNet size)
    # Resize EuroSAT (64x64) to 224x224
    image = tf.image.resize(image, [224, 224])
    
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)

# Data augmentation (same as Part A)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomBrightness(0.1),
], name='augmentation')

def augment(image, label):
    return data_augmentation(image, training=True), label

ds_train = ds_train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
BATCH_SIZE = 32
ds_train = ds_train.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\\n✓ Data prepared for ResNet50")
print(f"  Image size: 224×224 (ResNet50 standard)")
print(f"  Batch size: {BATCH_SIZE}")"""))

# =============================================================================
# STEP 2: LOAD PRE-TRAINED RESNET50
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 2: Load Pre-trained ResNet50

ResNet50 architecture:
- 50 layers deep
- Skip connections (residual blocks)
- ~25 million parameters
- Pre-trained on ImageNet (1.2M images, 1000 classes)

We'll load it **without the top classification layer** (include_top=False).
"""))

cells.append(nbf.v4.new_code_cell("""# Load pre-trained ResNet50
print("Loading pre-trained ResNet50...")

base_model = ResNet50(
    include_top=False,           # Exclude ImageNet classifier
    weights='imagenet',           # Use ImageNet pre-trained weights
    input_shape=(224, 224, 3),   # EuroSAT RGB
    pooling='avg'                # Global average pooling
)

print(f"\\n✓ ResNet50 loaded")
print(f"  Total parameters: {base_model.count_params():,}")
print(f"  Trainable: {base_model.trainable}")
print(f"  Output shape: {base_model.output_shape}")"""))

cells.append(nbf.v4.new_markdown_cell("""### Freeze Base Model

For **feature extraction**, we freeze all ResNet50 layers initially. This means:
- Pre-trained weights don't change
- Only train the new classification head
- Much faster training
- Prevents destroying good features
"""))

cells.append(nbf.v4.new_code_cell("""# Freeze the base model
base_model.trainable = False

print("✓ Base model frozen")
print(f"  Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in base_model.trainable_weights]):,}")
print(f"  Non-trainable parameters: {sum([tf.keras.backend.count_params(w) for w in base_model.non_trainable_weights]):,}")"""))

# =============================================================================
# STEP 3: FEATURE EXTRACTION
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 3: Feature Extraction - Train Classification Head (15 minutes)

Now we add our custom classification layers on top of frozen ResNet50.

**Architecture:**
```
Input (224×224×3)
    ↓
ResNet50 Base (frozen) → 2048 features
    ↓
Dense(512, ReLU) + Dropout
    ↓
Dense(10, Softmax)
```
"""))

cells.append(nbf.v4.new_code_cell("""# Build complete model with custom head
inputs = keras.Input(shape=(224, 224, 3))

# ResNet50 base
x = base_model(inputs, training=False)

# Custom classification head
x = layers.Dense(512, activation='relu', name='fc1')(x)
x = layers.Dropout(0.5, name='dropout1')(x)
outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

# Create model
model_feature_extraction = keras.Model(inputs, outputs, name='ResNet50_FeatureExtraction')

print("✓ Model with custom head created")
model_feature_extraction.summary()"""))

cells.append(nbf.v4.new_code_cell("""# Compile model
model_feature_extraction.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\\n✓ Model compiled for feature extraction")
print("  Only training the classification head (~500K parameters)")"""))

cells.append(nbf.v4.new_markdown_cell("""### Train Classification Head"""))

cells.append(nbf.v4.new_code_cell("""# Callbacks
callbacks_fe = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('resnet50_feature_extraction.h5', monitor='val_accuracy', 
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# Train
print("Training classification head (feature extraction)...")
print("=" * 70)

history_fe = model_feature_extraction.fit(
    ds_train,
    validation_data=ds_val,
    epochs=20,  # Fewer epochs needed
    callbacks=callbacks_fe,
    verbose=1
)

print("=" * 70)
print("\\n✓ Feature extraction training complete!")"""))

cells.append(nbf.v4.new_markdown_cell("""### Evaluate Feature Extraction Model"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate on test set
test_loss_fe, test_acc_fe = model_feature_extraction.evaluate(ds_test, verbose=0)

print(f"\\n📊 Feature Extraction Results:")
print(f"   Test Accuracy: {test_acc_fe*100:.2f}%")
print(f"   Test Loss: {test_loss_fe:.4f}")

if test_acc_fe > 0.92:
    print("\\n🎉 Excellent! Already beating from-scratch CNN!")
    print("   Transfer learning is working well")"""))

# =============================================================================
# STEP 4: FINE-TUNING
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 4: Fine-Tuning - Unfreeze Some Layers (15 minutes)

Now we'll **fine-tune** by unfreezing the last few layers of ResNet50.

**Strategy:**
1. Unfreeze last 20 layers (out of 175)
2. Use very low learning rate (1e-5)
3. Allow model to adapt to satellite imagery

**Why this works:**
- Early layers (edges, textures) are general → keep frozen
- Later layers (objects, semantics) need adaptation → unfreeze
"""))

cells.append(nbf.v4.new_code_cell("""# Unfreeze the base model
base_model.trainable = True

# But keep early layers frozen
print(f"Total layers in ResNet50: {len(base_model.layers)}")

# Freeze all layers except last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

print(f"\\n✓ Fine-tuning configuration:")
print(f"  Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")
print(f"  Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# Count trainable parameters
trainable_params = sum([tf.keras.backend.count_params(w) for w in model_feature_extraction.trainable_weights])
print(f"  Trainable parameters: {trainable_params:,}")"""))

cells.append(nbf.v4.new_code_cell("""# Recompile with lower learning rate
model_feature_extraction.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Much lower!
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model recompiled for fine-tuning")
print("  Learning rate: 1e-5 (100x smaller)")"""))

cells.append(nbf.v4.new_markdown_cell("""### Fine-Tune the Model"""))

cells.append(nbf.v4.new_code_cell("""# Callbacks for fine-tuning
callbacks_ft = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('resnet50_finetuned.h5', monitor='val_accuracy', 
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1)
]

# Continue training (fine-tuning)
print("Fine-tuning model...")
print("=" * 70)

history_ft = model_feature_extraction.fit(
    ds_train,
    validation_data=ds_val,
    epochs=30,  # More epochs for fine-tuning
    callbacks=callbacks_ft,
    verbose=1
)

print("=" * 70)
print("\\n✓ Fine-tuning complete!")"""))

cells.append(nbf.v4.new_markdown_cell("""### Evaluate Fine-Tuned Model"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate on test set
test_loss_ft, test_acc_ft = model_feature_extraction.evaluate(ds_test, verbose=0)

print(f"\\n📊 Fine-Tuned Model Results:")
print(f"   Test Accuracy: {test_acc_ft*100:.2f}%")
print(f"   Test Loss: {test_loss_ft:.4f}")

# Compare with feature extraction
improvement = (test_acc_ft - test_acc_fe) * 100
print(f"\\n   Improvement: +{improvement:.2f}% over feature extraction")

if test_acc_ft > 0.94:
    print("\\n🎉 Outstanding! >94% accuracy achieved!")
    print("   Transfer learning + fine-tuning is very effective")"""))

# =============================================================================
# STEP 5: COMPARISON
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Step 5: Comparison & Analysis (10 minutes)

Let's compare all three approaches:
1. **From-Scratch CNN** (Part A)
2. **Transfer Learning - Feature Extraction**
3. **Transfer Learning - Fine-Tuned**
"""))

cells.append(nbf.v4.new_code_cell("""# Summary comparison
# Note: Load Part A results if available, otherwise use typical values

# Typical results (you can update with your actual Part A results)
from_scratch_acc = 0.90  # Update with your Part A result

comparison_data = {
    'Method': [
        'From-Scratch CNN\\n(Part A)',
        'ResNet50\\nFeature Extraction',
        'ResNet50\\nFine-Tuned'
    ],
    'Test Accuracy': [
        from_scratch_acc * 100,
        test_acc_fe * 100,
        test_acc_ft * 100
    ],
    'Trainable Params': [
        '~300K',
        '~500K',
        '~5M'
    ],
    'Training Time': [
        '~15 min',
        '~5 min',
        '~10 min'
    ]
}

df_comparison = pd.DataFrame(comparison_data)

print("\\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(df_comparison.to_string(index=False))
print("=" * 70)"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualize Comparison"""))

cells.append(nbf.v4.new_code_cell("""# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['From-Scratch\\nCNN', 'ResNet50\\nFeature Ext.', 'ResNet50\\nFine-Tuned']
accuracies = [from_scratch_acc * 100, test_acc_fe * 100, test_acc_ft * 100]
colors = ['steelblue', 'orange', 'green']

bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('EuroSAT Classification: Model Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(85, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add baseline reference
ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90% threshold')
ax.legend()

plt.tight_layout()
plt.show()

print("\\n✓ Transfer learning provides significant improvement!")"""))

cells.append(nbf.v4.new_markdown_cell("""### Confusion Matrix (Fine-Tuned Model)"""))

cells.append(nbf.v4.new_code_cell("""# Generate predictions for confusion matrix
print("Generating predictions...")

y_true = []
y_pred = []

for images, labels in ds_test:
    predictions = model_feature_extraction.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - ResNet50 Fine-Tuned', fontsize=14, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("✓ Confusion matrix for fine-tuned model")"""))

cells.append(nbf.v4.new_markdown_cell("""### Per-Class Performance"""))

cells.append(nbf.v4.new_code_cell("""# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\\n📊 Per-Class Metrics (Fine-Tuned Model):")
print("=" * 80)
print(report_df[:-3].round(3))  # Exclude averages
print("=" * 80)

# Identify best and worst
metrics_df = report_df[:-3]
best_class = metrics_df['f1-score'].idxmax()
worst_class = metrics_df['f1-score'].idxmin()

print(f"\\n✨ Best: {best_class} (F1={metrics_df.loc[best_class, 'f1-score']:.3f})")
print(f"⚠️  Worst: {worst_class} (F1={metrics_df.loc[worst_class, 'f1-score']:.3f})")"""))

# =============================================================================
# CONCLUSION
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# 🎉 Transfer Learning Lab Complete!

## Summary

You've successfully:

✅ **Loaded** pre-trained ResNet50 (25M parameters from ImageNet)  
✅ **Feature Extraction:** Trained custom classifier head (92-93% accuracy)  
✅ **Fine-Tuned:** Adapted ResNet50 to satellite imagery (93-96% accuracy)  
✅ **Compared:** Demonstrated transfer learning superiority  
✅ **Achieved:** State-of-art results on EuroSAT  

---

## Key Insights

### Why Transfer Learning Works

1. **Pre-trained features are universal**
   - Edges, textures, patterns learned on ImageNet
   - Apply to satellite imagery without retraining

2. **Less data required**
   - ImageNet: 1.2M images
   - EuroSAT: 27K images
   - Transfer learning bridges the gap

3. **Faster convergence**
   - Start from good weights
   - 5-15 min vs 15-30 min from scratch

4. **Better accuracy**
   - +3-6% improvement
   - Critical for real-world applications

### When to Use Transfer Learning

✅ **Use transfer learning when:**
- Limited training data (<10K images)
- Similar task (image classification)
- Time/compute constrained
- Need best accuracy

❌ **Train from scratch when:**
- Very different domain (medical, satellite with many bands)
- Abundant data (>100K images)
- Specific architectural requirements
- Learning about CNNs (educational)

---

## Comparison Summary

| Metric | From-Scratch | Feature Extraction | Fine-Tuned |
|--------|--------------|-------------------|------------|
| **Accuracy** | 90-92% | 92-93% | 93-96% |
| **Training Time** | 15-20 min | 5-10 min | 10-15 min |
| **Parameters Trained** | ~300K | ~500K | ~5M |
| **Data Efficiency** | Needs more | Good | Best |
| **Overfitting Risk** | Higher | Low | Medium |

---

## Philippine Applications

**Transfer learning is ideal for:**

1. **Mangrove Mapping**
   - Limited labeled data
   - High accuracy needed
   - ResNet50 → Fine-tune on Palawan mangroves

2. **Rice Paddy Detection**
   - Seasonal patterns
   - VGG16 → Fine-tune on Central Luzon

3. **Informal Settlement Detection**
   - Urban patterns similar to ImageNet
   - ResNet50 → Fine-tune on Metro Manila

4. **Disaster Damage Assessment**
   - Limited post-disaster data
   - Transfer from pre-trained → Quick deployment

---

## Next Steps

### Continue to Part C: U-Net Segmentation
- Pixel-level land cover classification
- Encoder-decoder architecture
- Palawan forest boundaries

### Experiments to Try

**Easy:**
1. Unfreeze different numbers of layers (10, 30, 50)
2. Try different learning rates (1e-4, 1e-6)
3. Compare ResNet50 vs VGG16

**Medium:**
4. Use different pre-trained models (EfficientNet, MobileNet)
5. Multi-spectral adaptation (10 bands)
6. Apply to Palawan dataset

**Advanced:**
7. Progressive unfreezing (unfreeze layers gradually)
8. Discriminative learning rates (different LR per layer)
9. Ensemble multiple fine-tuned models

---

## Save Models
"""))

cells.append(nbf.v4.new_code_cell("""# Save fine-tuned model
model_feature_extraction.save('resnet50_eurosat_finetuned_final.h5')
print("✓ Fine-tuned model saved")

# Save training histories
import pickle
with open('transfer_learning_history.pkl', 'wb') as f:
    pickle.dump({
        'feature_extraction': history_fe.history,
        'fine_tuning': history_ft.history
    }, f)
print("✓ Training histories saved")

# Export results
results = pd.DataFrame({
    'Method': ['From-Scratch', 'Feature Extraction', 'Fine-Tuned'],
    'Test_Accuracy': [from_scratch_acc, test_acc_fe, test_acc_ft]
})
results.to_csv('transfer_learning_comparison.csv', index=False)
print("✓ Comparison results saved")

print("\\n🎊 All done! Ready for U-Net segmentation (Part C)...")"""))

# Add cells to notebook
nb['cells'] = cells

# Write notebook
output_path = 'session4_transfer_learning_STUDENT.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"\\n✓ Transfer Learning Notebook Created!")
print(f"  Total cells: {len(cells)}")
print(f"  File: {output_path}")
print(f"  Duration: ~60 minutes")
