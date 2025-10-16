"""Build U-Net Segmentation Notebook - Part 2 (Training & Evaluation)"""
import nbformat as nbf

# Load existing
with open('session4_unet_segmentation_STUDENT.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

cells = []

# Training
cells.append(nbf.v4.new_markdown_cell("# Step 3: Training (20 min)"))

cells.append(nbf.v4.new_code_cell("""from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('unet_best.h5', monitor='val_mean_io_u', save_best_only=True)
]
history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                    epochs=50, batch_size=8, callbacks=callbacks)
print("Training complete!")"""))

# Evaluation
cells.append(nbf.v4.new_markdown_cell("# Step 4: Evaluation (10 min)"))

cells.append(nbf.v4.new_code_cell("""test_loss, test_acc, test_iou = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test IoU: {test_iou:.4f}")"""))

# Visualization
cells.append(nbf.v4.new_markdown_cell("# Step 5: Visualization (5 min)"))

cells.append(nbf.v4.new_code_cell("""predictions = model.predict(X_test)
pred_masks = np.argmax(predictions, axis=-1)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(3):
    idx = i
    axes[i, 0].imshow(X_test[idx])
    axes[i, 0].set_title('Input')
    axes[i, 1].imshow(y_test[idx], cmap='Greens')
    axes[i, 1].set_title('Ground Truth')
    axes[i, 2].imshow(pred_masks[idx], cmap='Greens')
    axes[i, 2].set_title('Prediction')
    overlay = X_test[idx].copy()
    overlay[pred_masks[idx]==1] = overlay[pred_masks[idx]==1]*0.5 + [0,0.5,0]*0.5
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title('Overlay')
    for ax in axes[i]: ax.axis('off')
plt.tight_layout()
plt.show()"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""---
# 🎉 U-Net Segmentation Complete!

## Summary
✅ Built U-Net encoder-decoder architecture  
✅ Trained on forest segmentation  
✅ Achieved {test_iou:.2f} IoU  
✅ Pixel-level classification successful  

## Philippine Applications
- **Mangrove mapping** - Palawan coastlines
- **Rice field delineation** - Central Luzon
- **Building footprints** - Metro Manila
- **Flood extent** - Pampanga Basin

## Key Insights
U-Net excels at:
- Precise boundary delineation
- Spatial detail preservation
- Works with limited data (skip connections help)

**Next:** Apply to real Palawan Sentinel-2 imagery!
"""))

nb['cells'].extend(cells)
with open('session4_unet_segmentation_STUDENT.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f"U-Net notebook complete: {len(nb['cells'])} cells total")
