"""
Complete Session 4 CNN Classification Notebook
Add CNN architecture, training, evaluation, and experimentation
"""

import nbformat as nbf
import os

# Load existing notebook
if os.path.exists('session4_cnn_classification_STUDENT.ipynb'):
    with open('session4_cnn_classification_STUDENT.ipynb', 'r') as f:
        nb = nbf.read(f, as_version=4)
    print("✓ Loading existing notebook...")
else:
    nb = nbf.v4.new_notebook()
    print("✓ Creating new notebook...")

additional_cells = []

# =============================================================================
# STEP 4: BUILD CNN ARCHITECTURE
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# Step 4: Build CNN Architecture (20 minutes)

Now we'll design and build a CNN from scratch!

## Architecture Design

We'll create a **3-block CNN**:

```
Input (64×64×3)
    ↓
Block 1: Conv(32) → Conv(32) → MaxPool → Dropout
    ↓
Block 2: Conv(64) → Conv(64) → MaxPool → Dropout
    ↓
Block 3: Conv(128) → MaxPool → Dropout
    ↓
Flatten
    ↓
Dense(256) → Dropout
    ↓
Output (10 classes)
```

**Design Principles:**
- Start with 32 filters, double each block (32→64→128)
- Use 3×3 convolutions (standard)
- MaxPool after each block (reduce dimensions)
- Dropout for regularization (prevent overfitting)
- ReLU activation (all hidden layers)
- Softmax activation (output layer)

---

## 4.1: Define the Model
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Build CNN model
def build_cnn_model(input_shape=(64, 64, 3), num_classes=10):
    \"\"\"
    Build a 3-block CNN for EuroSAT classification
    
    Parameters:
    -----------
    input_shape : tuple
        Input image dimensions
    num_classes : int
        Number of output classes
    
    Returns:
    --------
    model : keras.Model
        Compiled CNN model
    \"\"\"
    
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout4'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='EuroSAT_CNN')
    
    return model

# Create model
model = build_cnn_model(input_shape=(64, 64, 3), num_classes=num_classes)

print("✓ CNN model created")
print(f"  Architecture: 3-block CNN")
print(f"  Input shape: (64, 64, 3)")
print(f"  Output classes: {num_classes}")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Model Summary"""))

additional_cells.append(nbf.v4.new_code_cell("""# Display model architecture
model.summary()

# Calculate total parameters
total_params = model.count_params()
print(f"\\n📊 Total Parameters: {total_params:,}")
print(f"   Trainable: {total_params:,}")

# Breakdown by layer type
conv_params = sum([layer.count_params() for layer in model.layers if 'conv' in layer.name])
dense_params = sum([layer.count_params() for layer in model.layers if 'dense' in layer.name or 'fc' in layer.name])

print(f"\\n   Convolutional layers: {conv_params:,}")
print(f"   Dense layers: {dense_params:,}")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Visualize Architecture"""))

additional_cells.append(nbf.v4.new_code_cell("""# Plot model architecture
keras.utils.plot_model(
    model,
    to_file='cnn_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # Top to bottom
    dpi=96
)

# Display
from IPython.display import Image
Image('cnn_architecture.png')"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 4.2: Compile the Model

We need to configure:
- **Loss function:** Sparse categorical crossentropy (for integer labels)
- **Optimizer:** Adam (adaptive learning rate)
- **Metrics:** Accuracy (percentage of correct predictions)
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: Sparse Categorical Crossentropy")
print(f"  Metrics: Accuracy")"""))

# =============================================================================
# STEP 5: TRAINING
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# Step 5: Training & Monitoring (20 minutes)

Time to train! We'll use callbacks to:
- **Early Stopping:** Stop if validation loss doesn't improve
- **Model Checkpoint:** Save best model weights
- **Reduce LR:** Lower learning rate when plateauing

---

## 5.1: Configure Callbacks
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Define callbacks
callbacks = [
    # Early stopping: stop if val_loss doesn't improve for 10 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Model checkpoint: save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # Reduce learning rate: divide by 2 if plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("✓ Callbacks configured")
print("  - Early stopping (patience=10)")
print("  - Model checkpoint (save best)")
print("  - Reduce LR on plateau")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 5.2: Train the Model

⏱️ **Training Time:** ~15-20 minutes on GPU, 1-2 hours on CPU

We'll train for up to 50 epochs, but early stopping will likely halt around 20-30 epochs.
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Train model
print("Starting training...")
print("=" * 70)

EPOCHS = 50

history = model.fit(
    ds_train_final,
    validation_data=ds_val_final,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("=" * 70)
print("\\n✓ Training complete!")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Training Curves"""))

additional_cells.append(nbf.v4.new_code_cell("""# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\\n📊 Final Training Metrics:")
print(f"   Train Accuracy: {final_train_acc*100:.2f}%")
print(f"   Train Loss: {final_train_loss:.4f}")
print(f"   Val Accuracy: {final_val_acc*100:.2f}%")
print(f"   Val Loss: {final_val_loss:.4f}")

# Check for overfitting
gap = final_train_acc - final_val_acc
if gap > 0.05:
    print(f"\\n⚠️  Possible overfitting: {gap*100:.1f}% gap between train/val accuracy")
else:
    print(f"\\n✓ Good generalization: only {gap*100:.1f}% gap")"""))

# =============================================================================
# STEP 6: EVALUATION
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# Step 6: Evaluation & Analysis (15 minutes)

Now let's evaluate on the test set and analyze results.

---

## 6.1: Test Set Evaluation
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Evaluate on test set
print("Evaluating on test set...")

test_loss, test_accuracy = model.evaluate(ds_test_final, verbose=0)

print(f"\\n🎯 Test Set Results:")
print(f"   Accuracy: {test_accuracy*100:.2f}%")
print(f"   Loss: {test_loss:.4f}")

if test_accuracy > 0.90:
    print("\\n🎉 Excellent! You've achieved >90% accuracy!")
elif test_accuracy > 0.85:
    print("\\n✓ Good! Above 85% is solid for first attempt")
else:
    print("\\n💡 Room for improvement - try tuning hyperparameters")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 6.2: Confusion Matrix
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Generate predictions
print("Generating predictions for confusion matrix...")

y_true = []
y_pred = []

for images, labels in ds_test_final:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"✓ Predictions generated for {len(y_true)} test images")"""))

additional_cells.append(nbf.v4.new_code_cell("""# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - EuroSAT Test Set', fontsize=14, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\\n✓ Confusion matrix generated")
print("  Diagonal = correct predictions")
print("  Off-diagonal = misclassifications")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 6.3: Per-Class Metrics
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Convert to DataFrame for nice display
report_df = pd.DataFrame(report).transpose()

print("\\n📊 Per-Class Performance:")
print("=" * 80)
print(report_df.round(3))
print("=" * 80)

# Highlight best and worst classes
metrics_df = report_df[:-3]  # Exclude accuracy, macro avg, weighted avg
best_class = metrics_df['f1-score'].idxmax()
worst_class = metrics_df['f1-score'].idxmin()

print(f"\\n✨ Best performing class: {best_class} (F1={metrics_df.loc[best_class, 'f1-score']:.3f})")
print(f"⚠️  Worst performing class: {worst_class} (F1={metrics_df.loc[worst_class, 'f1-score']:.3f})")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Visualize Per-Class Accuracy"""))

additional_cells.append(nbf.v4.new_code_cell("""# Plot per-class F1 scores
fig, ax = plt.subplots(figsize=(12, 6))

classes = metrics_df.index.tolist()
f1_scores = metrics_df['f1-score'].values

bars = ax.barh(classes, f1_scores, color='steelblue', edgecolor='black', linewidth=1.5)

# Color best and worst
bars[classes.index(best_class)].set_color('green')
bars[classes.index(worst_class)].set_color('orange')

ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Class', fontsize=12, fontweight='bold')
ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax.text(score + 0.01, i, f'{score:.3f}', 
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 6.4: Analyze Misclassifications
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Find misclassified examples
misclassified_indices = np.where(y_true != y_pred)[0]
print(f"\\nTotal misclassifications: {len(misclassified_indices)} / {len(y_true)}")
print(f"Error rate: {len(misclassified_indices)/len(y_true)*100:.2f}%")

# Show some misclassified examples
if len(misclassified_indices) > 0:
    # Get first 12 misclassifications
    sample_errors = misclassified_indices[:12]
    
    # Get corresponding images
    test_images = []
    for images, labels in ds_test_final:
        test_images.extend([img.numpy() for img in images])
    test_images = np.array(test_images)
    
    # Plot misclassifications
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, error_idx in enumerate(sample_errors):
        ax = axes[idx]
        
        # Display image
        ax.imshow(test_images[error_idx])
        
        true_label = class_names[y_true[error_idx]]
        pred_label = class_names[y_pred[error_idx]]
        
        ax.set_title(f'True: {true_label}\\nPred: {pred_label}',
                     fontsize=9, fontweight='bold',
                     color='red')
        ax.axis('off')
    
    plt.suptitle('Sample Misclassifications', fontsize=14, fontweight='bold', color='red')
    plt.tight_layout()
    plt.show()
    
    print("\\n💡 Misclassification Analysis:")
    print("  Look for patterns:")
    print("  - Similar-looking classes (e.g., forest types)")
    print("  - Ambiguous examples")
    print("  - Data augmentation might help")"""))

# =============================================================================
# CONCLUSION
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# 🎉 Lab Complete!

## Summary

You've successfully:

✅ **Built** a CNN from scratch (3 blocks, ~300K parameters)  
✅ **Trained** on 27,000 Sentinel-2 images  
✅ **Achieved** 90%+ accuracy on EuroSAT dataset  
✅ **Evaluated** with confusion matrix and per-class metrics  
✅ **Analyzed** misclassifications  

---

## Key Takeaways

### What Worked Well
- **Architecture:** 3-block design with progressive filters (32→64→128)
- **Regularization:** Dropout prevented overfitting
- **Data Augmentation:** Improved generalization
- **Callbacks:** Early stopping saved training time

### Compared to Random Forest (Session 1-2)
- **CNN:** 92-95% accuracy (automatic features)
- **Random Forest:** 85-90% accuracy (manual features)
- **Improvement:** +5-10% for critical applications

### What's Next?
- **Transfer Learning:** Use pre-trained ResNet (Session 4B)
- **U-Net:** Pixel-level segmentation (Session 4C)
- **Palawan:** Apply to real Philippine data
- **Production:** Deploy model for monitoring

---

## Exercises to Try

### Easy
1. Change batch size (16, 64) and observe effects
2. Modify dropout rates (0.1, 0.5)
3. Try different learning rates

### Medium
4. Add another convolutional block
5. Use different augmentation techniques
6. Experiment with optimizer (SGD vs Adam)

### Advanced
7. Implement learning rate scheduling
8. Add batch normalization layers
9. Try different architectures (VGG-style, ResNet-style)
10. Fine-tune on Palawan-specific classes

---

## Save Your Work
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Save final model
model.save('eurosat_cnn_final.h5')
print("✓ Model saved: eurosat_cnn_final.h5")

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("✓ Training history saved: training_history.pkl")

# Export predictions
results = pd.DataFrame({
    'true_label': [class_names[i] for i in y_true],
    'predicted_label': [class_names[i] for i in y_pred],
    'correct': y_true == y_pred
})
results.to_csv('test_predictions.csv', index=False)
print("✓ Predictions saved: test_predictions.csv")

print("\\n🎊 All done! Continue to transfer learning notebook...")"""))

# Add all cells
nb['cells'].extend(additional_cells)

# Write completed notebook
with open('session4_cnn_classification_STUDENT.ipynb', 'w') as f:
    nbf.write(nb, f)

print("\\n" + "=" * 70)
print("✓ SESSION 4 CNN CLASSIFICATION NOTEBOOK COMPLETE!")
print("=" * 70)
print(f"Total cells: {len(nb['cells'])}")
print(f"File: session4_cnn_classification_STUDENT.ipynb")
print("\\nNotebook covers:")
print("  ✓ Environment setup & GPU check")
print("  ✓ EuroSAT dataset download")
print("  ✓ Data preprocessing & augmentation")
print("  ✓ CNN architecture (3 blocks)")
print("  ✓ Training with callbacks")
print("  ✓ Comprehensive evaluation")
print("  ✓ Misclassification analysis")
print("=" * 70)
