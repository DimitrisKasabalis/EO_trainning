"""Build U-Net Segmentation Notebook - Part 1 (Setup & Architecture)"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# Session 4 Part C: U-Net for Semantic Segmentation
## Pixel-Level Land Cover Classification

**Duration:** 60 minutes | **Difficulty:** Advanced  
**Task:** Forest boundary delineation in Palawan

---

## 🎯 Objectives

1. ✅ Understand semantic segmentation vs classification
2. ✅ Implement U-Net encoder-decoder architecture
3. ✅ Train on pixel-level masks
4. ✅ Evaluate with IoU metric
5. ✅ Visualize segmentation results

---

## 🌳 What is Semantic Segmentation?

**Classification:** One label per image  
**Segmentation:** One label per pixel

**Applications:** Forest boundaries, field delineation, building footprints

---

Let's build a segmentation model! 🚀
"""))

# Setup
cells.append(nbf.v4.new_markdown_cell("# Step 1: Setup & Dataset (10 min)"))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

print(f"TensorFlow: {tf.__version__}")
np.random.seed(42)
tf.random.set_seed(42)"""))

# Generate data
cells.append(nbf.v4.new_code_cell("""def generate_forest_data(n=500, size=256):
    images, masks = [], []
    for _ in range(n):
        img = np.random.rand(size, size, 3) * 0.3
        mask = np.zeros((size, size), dtype=np.uint8)
        for _ in range(np.random.randint(2, 6)):
            cx, cy = np.random.randint(50, size-50, 2)
            r = np.random.randint(30, 80)
            Y, X = np.ogrid[:size, :size]
            forest = (np.sqrt((X-cx)**2 + (Y-cy)**2) + np.random.randn(size,size)*10) < r
            mask[forest] = 1
            img[forest, 1] += 0.4
        images.append(np.clip(img, 0, 1))
        masks.append(mask)
    return np.array(images), np.array(masks)

images, masks = generate_forest_data(500, 256)
print(f"Dataset: {images.shape}, {masks.shape}")"""))

# Split
cells.append(nbf.v4.new_code_cell("""X_temp, X_test, y_temp, y_test = train_test_split(images, masks, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")"""))

# Architecture
cells.append(nbf.v4.new_markdown_cell("# Step 2: Build U-Net (15 min)"))

cells.append(nbf.v4.new_code_cell("""def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    conv = conv_block(x, filters)
    pool = layers.MaxPooling2D(2)(conv)
    return conv, pool

def decoder_block(x, skip, filters):
    up = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    concat = layers.Concatenate()([up, skip])
    return conv_block(concat, filters)

def build_unet(input_shape=(256,256,3), num_classes=2):
    inputs = layers.Input(input_shape)
    # Encoder
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)
    # Bottleneck
    b = conv_block(p4, 1024)
    # Decoder
    d4 = decoder_block(b, c4, 512)
    d3 = decoder_block(d4, c3, 256)
    d2 = decoder_block(d3, c2, 128)
    d1 = decoder_block(d2, c1, 64)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
    return models.Model(inputs, outputs, name='U-Net')

model = build_unet()
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)])
print("U-Net ready!")"""))

nb['cells'] = cells
with open('session4_unet_segmentation_STUDENT.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f"Part 1 created: {len(cells)} cells")
