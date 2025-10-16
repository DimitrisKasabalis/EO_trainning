"""
Finalize Session 3 Theory Notebook - Add Parts 4-5
Convolution operations and CNN exploration
"""

import nbformat as nbf
import os

# Load existing notebook
if os.path.exists('session3_theory_interactive.ipynb'):
    with open('session3_theory_interactive.ipynb', 'r') as f:
        nb = nbf.read(f, as_version=4)
    print("✓ Loading notebook with Parts 1-3...")
else:
    print("⚠️ Base notebook not found. Run build script first.")
    exit(1)

final_cells = []

# =============================================================================
# PART 4: CONVOLUTION OPERATIONS
# =============================================================================

final_cells.append(nbf.v4.new_markdown_cell("""---

# Part 4: Convolution Operations (20 minutes)

## What is Convolution?

**Convolution** is the core operation in CNNs. It:
1. Takes a small filter (kernel) like 3×3
2. Slides it across an image
3. Performs element-wise multiplication
4. Sums the results
5. Creates a feature map

**Why convolution for images?**
- ✅ **Spatial locality:** Nearby pixels are related
- ✅ **Parameter sharing:** Same filter across entire image
- ✅ **Translation invariance:** Detects patterns anywhere
- ✅ **Hierarchical learning:** Builds from simple to complex features

---

## 4.1: Manual Convolution Implementation
"""))

final_cells.append(nbf.v4.new_code_cell("""def convolve2d(image, kernel):
    \"\"\"
    Apply 2D convolution manually
    
    Parameters:
    -----------
    image : 2D array
        Input image
    kernel : 2D array
        Convolution filter
    
    Returns:
    --------
    output : 2D array
        Convolved feature map
    \"\"\"
    # Get dimensions
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output size
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    
    # Initialize output
    output = np.zeros((output_h, output_w))
    
    # Slide kernel across image
    for i in range(output_h):
        for j in range(output_w):
            # Extract region
            region = image[i:i+kernel_h, j:j+kernel_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)
    
    return output

print("✓ Convolution function defined")
print("  This mimics how CNNs process images!")"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

## 4.2: Classic Image Filters

Let's apply different filters to understand what CNNs learn!
"""))

final_cells.append(nbf.v4.new_code_cell("""# Create a simple test image (simulating Sentinel-2 NIR band)
# Simulate forest (bright) vs non-forest (dark) with edges
test_image = np.zeros((50, 50))
test_image[10:40, 10:25] = 0.8  # Forest patch (high NIR)
test_image[10:40, 25:40] = 0.2  # Urban/bare soil (low NIR)

# Add some noise for realism
test_image += np.random.normal(0, 0.05, test_image.shape)
test_image = np.clip(test_image, 0, 1)

# Define classic filters
filters = {
    'Vertical Edge': np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    'Horizontal Edge': np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]),
    'Edge Detection (Sobel)': np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]),
    'Sharpen': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ]),
    'Blur (Smoothing)': np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9,
    'Identity': np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
}

print("✓ Test image and filters created")
print(f"  Image size: {test_image.shape}")
print(f"  Number of filters: {len(filters)}")"""))

final_cells.append(nbf.v4.new_markdown_cell("""### Apply Filters and Visualize"""))

final_cells.append(nbf.v4.new_code_cell("""# Apply all filters
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Original image
axes[0].imshow(test_image, cmap='gray')
axes[0].set_title('Original Image\\n(Simulated NIR Band)', fontsize=11, fontweight='bold')
axes[0].axis('off')

# Apply each filter
for idx, (name, kernel) in enumerate(filters.items(), start=1):
    # Convolve
    filtered = convolve2d(test_image, kernel)
    
    # Display
    axes[idx].imshow(filtered, cmap='gray')
    axes[idx].set_title(f'{name}\\nFilter', fontsize=11, fontweight='bold')
    axes[idx].axis('off')
    
    # Show kernel as text
    kernel_text = f"Kernel:\\n{kernel}"
    axes[idx].text(0.5, -0.15, kernel_text, transform=axes[idx].transAxes,
                   fontsize=7, ha='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()

print("\\n✓ Filters applied successfully!")
print("\\n🔍 Observations:")
print("  • Vertical Edge: Detects vertical boundaries (forest | urban)")
print("  • Horizontal Edge: Detects horizontal boundaries")
print("  • Sharpen: Enhances edges and details")
print("  • Blur: Smooths out noise")
print("  • Identity: Passes through unchanged")"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

## 4.3: Simulate Sentinel-2 Image

Let's apply filters to a more realistic Sentinel-2-like image!
"""))

final_cells.append(nbf.v4.new_code_cell("""# Create synthetic Sentinel-2 NIR band (64x64)
np.random.seed(42)

# Simulate different land covers
s2_image = np.zeros((64, 64))

# Forest blocks (high NIR)
s2_image[5:25, 5:25] = 0.8 + np.random.normal(0, 0.05, (20, 20))
s2_image[40:60, 40:60] = 0.75 + np.random.normal(0, 0.05, (20, 20))

# Water (very low NIR)
s2_image[5:25, 40:60] = 0.1 + np.random.normal(0, 0.02, (20, 20))

# Agriculture (medium NIR)
s2_image[40:60, 5:25] = 0.5 + np.random.normal(0, 0.08, (20, 20))

# Urban/bare soil (low NIR)
s2_image[25:40, 25:40] = 0.25 + np.random.normal(0, 0.05, (15, 15))

# Clip to valid range
s2_image = np.clip(s2_image, 0, 1)

print(f"✓ Synthetic Sentinel-2 image created: {s2_image.shape}")
print("  Contains: Forest, Water, Agriculture, Urban")"""))

final_cells.append(nbf.v4.new_code_cell("""# Apply multiple edge detection filters
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(s2_image, cmap='RdYlGn', vmin=0, vmax=1)
axes[0, 0].set_title('Original Sentinel-2 NIR\\n(Synthetic)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Vertical edges
vert_edges = convolve2d(s2_image, filters['Vertical Edge'])
axes[0, 1].imshow(vert_edges, cmap='seismic')
axes[0, 1].set_title('Vertical Edge Detection\\n(Forest | Water boundary)', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Horizontal edges
horiz_edges = convolve2d(s2_image, filters['Horizontal Edge'])
axes[0, 2].imshow(horiz_edges, cmap='seismic')
axes[0, 2].set_title('Horizontal Edge Detection', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# Sobel (combined edges)
sobel = convolve2d(s2_image, filters['Edge Detection (Sobel)'])
axes[1, 0].imshow(sobel, cmap='hot')
axes[1, 0].set_title('Sobel Edge Detection\\n(All edges)', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Blur (texture smoothing)
blurred = convolve2d(s2_image, filters['Blur (Smoothing)'])
axes[1, 1].imshow(blurred, cmap='RdYlGn')
axes[1, 1].set_title('Blur Filter\\n(Noise reduction)', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Sharpen
sharpened = convolve2d(s2_image, filters['Sharpen'])
axes[1, 2].imshow(sharpened, cmap='RdYlGn')
axes[1, 2].set_title('Sharpen Filter\\n(Detail enhancement)', fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("\\n✓ Convolution filters applied to Sentinel-2-like image!")
print("\\n🎯 This is what CNNs do automatically:")
print("  • Learn optimal filters (not pre-defined)")
print("  • Stack multiple filters (32, 64, 128...)")
print("  • Build hierarchical features (edges → textures → objects)")"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

### Understanding Feature Maps

When a CNN applies a filter, it creates a **feature map**. Multiple filters = multiple feature maps.

**Example:** First convolutional layer in ResNet
- Input: 64×64×10 (Sentinel-2 image)
- Filter: 64 filters of size 3×3
- Output: 64×64×64 (64 feature maps)

Each feature map responds to different patterns!
"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

### 🎯 Key Takeaways - Part 4

✅ **Convolution = filter sliding across image**  
✅ **Filters detect specific patterns** (edges, textures)  
✅ **CNNs learn optimal filters** during training  
✅ **Feature maps** are outputs of convolution  
✅ **Multiple filters** capture different features  

**Connection to EO:**
- Layer 1 filters: Water/land boundaries, forest edges
- Layer 2 filters: Vegetation textures, urban patterns
- Layer 3 filters: Agricultural fields, forest stands

---
"""))

# =============================================================================
# PART 5: CNN ARCHITECTURE EXPLORATION
# =============================================================================

final_cells.append(nbf.v4.new_markdown_cell("""---

# Part 5: CNN Architecture Exploration (15 minutes)

Now let's explore real CNN architectures and understand their components!

## 5.1: Build a Simple CNN (Conceptually)

Let's design a CNN for Sentinel-2 scene classification:

**Task:** Classify 64×64 Sentinel-2 patches into 8 land cover classes

**Architecture:**
```
Input: 64×64×10 (10 Sentinel-2 bands)
    ↓
Conv1: 32 filters, 3×3 → 64×64×32
ReLU activation
MaxPool: 2×2 → 32×32×32
    ↓
Conv2: 64 filters, 3×3 → 32×32×64
ReLU activation
MaxPool: 2×2 → 16×16×64
    ↓
Conv3: 128 filters, 3×3 → 16×16×128
ReLU activation
GlobalAveragePool → 128
    ↓
Dense (Fully Connected): 128 → 8
Softmax activation
    ↓
Output: 8 class probabilities
```

---

## 5.2: Calculate Parameters
"""))

final_cells.append(nbf.v4.new_code_cell("""def calculate_cnn_parameters(architecture):
    \"\"\"
    Calculate number of trainable parameters in CNN
    \"\"\"
    total_params = 0
    
    print("CNN Architecture Analysis")
    print("=" * 70)
    
    for layer_name, layer_info in architecture.items():
        if 'conv' in layer_name.lower():
            # Convolution layer: (filter_h * filter_w * in_channels + 1) * out_channels
            kernel_h, kernel_w = layer_info['kernel_size']
            in_channels = layer_info['in_channels']
            out_channels = layer_info['out_channels']
            
            params = (kernel_h * kernel_w * in_channels + 1) * out_channels
            total_params += params
            
            print(f"{layer_name}:")
            print(f"  Kernel: {kernel_h}×{kernel_w}, In: {in_channels}, Out: {out_channels}")
            print(f"  Parameters: {params:,}")
            
        elif 'dense' in layer_name.lower():
            # Dense layer: (input_size + 1) * output_size
            input_size = layer_info['input_size']
            output_size = layer_info['output_size']
            
            params = (input_size + 1) * output_size
            total_params += params
            
            print(f"{layer_name}:")
            print(f"  Input: {input_size}, Output: {output_size}")
            print(f"  Parameters: {params:,}")
        
        print()
    
    print("=" * 70)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 70)
    
    return total_params

# Define our CNN architecture
our_cnn = {
    'Conv1': {'kernel_size': (3, 3), 'in_channels': 10, 'out_channels': 32},
    'Conv2': {'kernel_size': (3, 3), 'in_channels': 32, 'out_channels': 64},
    'Conv3': {'kernel_size': (3, 3), 'in_channels': 64, 'out_channels': 128},
    'Dense': {'input_size': 128, 'output_size': 8}
}

params = calculate_cnn_parameters(our_cnn)

print(f"\\n💡 For comparison:")
print(f"  ResNet50: ~25 million parameters")
print(f"  VGG16: ~138 million parameters")
print(f"  Our simple CNN: {params:,} parameters")
print(f"\\n  → Lightweight, suitable for small datasets!")"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

## 5.3: Visualize CNN Architecture
"""))

final_cells.append(nbf.v4.new_code_cell("""# Visualize the architecture flow
fig, ax = plt.subplots(figsize=(14, 8))

# Define layer positions and sizes
layers = [
    {'name': 'Input\\n64×64×10', 'x': 0, 'y': 0.5, 'w': 0.8, 'h': 0.8, 'color': 'lightblue'},
    {'name': 'Conv1 + ReLU\\n64×64×32', 'x': 1.5, 'y': 0.5, 'w': 0.7, 'h': 0.7, 'color': 'lightcoral'},
    {'name': 'MaxPool\\n32×32×32', 'x': 2.8, 'y': 0.5, 'w': 0.6, 'h': 0.6, 'color': 'lightyellow'},
    {'name': 'Conv2 + ReLU\\n32×32×64', 'x': 4.0, 'y': 0.5, 'w': 0.6, 'h': 0.6, 'color': 'lightcoral'},
    {'name': 'MaxPool\\n16×16×64', 'x': 5.2, 'y': 0.5, 'w': 0.5, 'h': 0.5, 'color': 'lightyellow'},
    {'name': 'Conv3 + ReLU\\n16×16×128', 'x': 6.4, 'y': 0.5, 'w': 0.5, 'h': 0.5, 'color': 'lightcoral'},
    {'name': 'Global\\nAvgPool', 'x': 7.6, 'y': 0.5, 'w': 0.3, 'h': 0.8, 'color': 'lightyellow'},
    {'name': 'Dense\\n8 classes', 'x': 8.5, 'y': 0.5, 'w': 0.3, 'h': 0.6, 'color': 'lightgreen'},
]

# Draw layers
for layer in layers:
    rect = plt.Rectangle((layer['x'] - layer['w']/2, layer['y'] - layer['h']/2),
                          layer['w'], layer['h'], 
                          facecolor=layer['color'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(layer['x'], layer['y'], layer['name'], 
            ha='center', va='center', fontsize=9, fontweight='bold')

# Draw arrows
for i in range(len(layers) - 1):
    ax.arrow(layers[i]['x'] + layers[i]['w']/2 + 0.05, 
             layers[i]['y'],
             layers[i+1]['x'] - layers[i+1]['w']/2 - layers[i]['x'] - layers[i]['w']/2 - 0.15,
             0, head_width=0.1, head_length=0.1, fc='gray', ec='gray')

ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')
ax.set_title('CNN Architecture for Sentinel-2 Scene Classification', 
             fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='lightblue', ec='black', label='Input'),
    plt.Rectangle((0, 0), 1, 1, fc='lightcoral', ec='black', label='Convolution + ReLU'),
    plt.Rectangle((0, 0), 1, 1, fc='lightyellow', ec='black', label='Pooling'),
    plt.Rectangle((0, 0), 1, 1, fc='lightgreen', ec='black', label='Dense/Output')
]
ax.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)

plt.tight_layout()
plt.show()

print("\\n✓ CNN architecture visualized!")
print("\\n📐 Layer Dimensions:")
print("  Notice how spatial dimensions decrease (64→32→16)")
print("  While channels increase (10→32→64→128)")
print("  This is typical: trade spatial resolution for semantic features")"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

## 5.4: Compare with Random Forest

Let's understand when to use CNNs vs Random Forest for EO tasks.
"""))

final_cells.append(nbf.v4.new_code_cell("""# Create comparison table
comparison_data = {
    'Aspect': [
        'Input Type',
        'Feature Engineering',
        'Spatial Context',
        'Training Data Needed',
        'Training Time',
        'Inference Speed',
        'Interpretability',
        'Typical Accuracy',
        'Hardware',
        'Best Use Case'
    ],
    'Random Forest\\n(Sessions 1-2)': [
        'Pixel features',
        'Manual (GLCM, NDVI, etc.)',
        'Limited (neighborhood)',
        '100-1000 samples',
        'Minutes',
        'Very fast (ms)',
        'High (feature importance)',
        '80-90%',
        'CPU sufficient',
        'Quick prototypes, small areas'
    ],
    'CNN\\n(Sessions 3-4)': [
        'Image patches',
        'Automatic (learned)',
        'Hierarchical (receptive field)',
        '1000-100K+ images',
        'Hours-Days',
        'Fast with GPU (10-100ms)',
        'Low (black box)',
        '90-98%',
        'GPU recommended',
        'Production, large areas, high accuracy'
    ]
}

# Display as formatted table
print("\\n" + "=" * 100)
print("RANDOM FOREST vs CONVOLUTIONAL NEURAL NETWORKS")
print("=" * 100)

for i, aspect in enumerate(comparison_data['Aspect']):
    rf_value = comparison_data['Random Forest\\n(Sessions 1-2)'][i]
    cnn_value = comparison_data['CNN\\n(Sessions 3-4)'][i]
    
    print(f"\\n{aspect}:")
    print(f"  RF:  {rf_value}")
    print(f"  CNN: {cnn_value}")

print("\\n" + "=" * 100)

print("\\n🎯 Decision Guide:")
print("\\n  Use RANDOM FOREST when:")
print("    • You have <1000 training samples")
print("    • Quick results needed (hours, not days)")
print("    • Interpretability is important")
print("    • No GPU available")
print("\\n  Use CNN when:")
print("    • You have >1000 labeled images")
print("    • Highest accuracy is critical")
print("    • Production deployment planned")
print("    • GPU resources available")
print("\\n  🌟 BEST PRACTICE: Start with RF, upgrade to CNN if needed!")"""))

final_cells.append(nbf.v4.new_markdown_cell("""---

### 🎯 Key Takeaways - Part 5

✅ **CNN architecture:** Input → Conv → Pool → ... → Dense → Output  
✅ **Parameters scale quickly:** Deeper networks = more parameters  
✅ **Spatial dimensions decrease:** While semantic depth increases  
✅ **Choose wisely:** RF for quick work, CNN for production  
✅ **Transfer learning helps:** Use pre-trained models  

---
"""))

# =============================================================================
# CONCLUSION
# =============================================================================

final_cells.append(nbf.v4.new_markdown_cell("""---

# 🎓 Session Complete! Summary

## What You've Learned

### Part 1: Perceptron
- ✅ Built artificial neuron from scratch
- ✅ Understood weights, bias, activation
- ✅ Trained using gradient descent
- ✅ Visualized decision boundary

### Part 2: Activation Functions
- ✅ Explored ReLU, Sigmoid, Tanh
- ✅ Understood non-linearity importance
- ✅ Saw vanishing gradient problem
- ✅ Learned why ReLU is popular

### Part 3: Neural Networks
- ✅ Built multi-layer network
- ✅ Implemented forward propagation
- ✅ Understood backpropagation
- ✅ Compared with perceptron

### Part 4: Convolution Operations
- ✅ Applied filters to images manually
- ✅ Visualized edge detection
- ✅ Processed Sentinel-2-like data
- ✅ Understood feature maps

### Part 5: CNN Architectures
- ✅ Designed CNN for EO classification
- ✅ Calculated parameters
- ✅ Visualized architecture flow
- ✅ Compared RF vs CNN

---

## Ready for Session 4!

In the next session, you'll:
- 🔨 Build actual CNNs with TensorFlow/Keras
- 🌲 Train on real Palawan land cover data
- 🎯 Implement U-Net for segmentation
- 📊 Compare results with Random Forest
- 🚀 Apply transfer learning

---

## 📚 Additional Practice (Optional)

**Exercises to Try:**

1. **Modify the Perceptron**
   - Add a third feature (elevation)
   - Try different learning rates
   - Visualize in 3D

2. **Experiment with Activations**
   - Replace ReLU with Tanh in the neural network
   - Compare training dynamics
   - Plot accuracy curves

3. **Custom Filters**
   - Design your own 3×3 filter
   - Test on the Sentinel-2 image
   - Explain what pattern it detects

4. **Architecture Design**
   - Design a CNN for 10-class classification
   - Calculate total parameters
   - Keep it under 100K parameters!

---

## 🌟 Key Concepts to Remember

**From Random Forest to CNNs:**
- RF: Manual features → Tree ensemble → Classification
- CNN: Raw pixels → Learned filters → Feature hierarchy → Classification

**Why CNNs Excel at Images:**
- Spatial locality (nearby pixels related)
- Parameter sharing (same filter everywhere)
- Hierarchical features (edges → textures → objects)
- End-to-end learning (optimize everything together)

**When CNNs Are Worth It:**
- Large labeled dataset (>1000 images)
- GPU available
- Accuracy is critical
- Production deployment

---

## 📖 Resources for Deeper Learning

**Interactive:**
- [TensorFlow Playground](https://playground.tensorflow.org/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Distill.pub Feature Visualization](https://distill.pub/2017/feature-visualization/)

**Courses:**
- Deep Learning Specialization (Coursera) - Andrew Ng
- Fast.ai Practical Deep Learning
- CS231n (Stanford) - CNNs for Visual Recognition

**Papers:**
- LeCun et al. (1998) - Gradient-Based Learning
- Krizhevsky et al. (2012) - AlexNet
- He et al. (2016) - ResNet

**EO-Specific:**
- EuroSAT Dataset
- TorchGeo Library
- Awesome Satellite Imagery Repo

---

**Congratulations! 🎉**

You now understand the fundamentals of deep learning and CNNs. Time to put it into practice in Session 4!

[Continue to Session 4 →](../../session4/notebooks/)

---

*Session 3 Theory Notebook - CopPhil Advanced Training Program*
"""))

# Add final cells
nb['cells'].extend(final_cells)

# Write complete notebook
with open('session3_theory_interactive.ipynb', 'w') as f:
    nbf.write(nb, f)

print("\\n" + "=" * 70)
print("✓ SESSION 3 THEORY NOTEBOOK COMPLETE!")
print("=" * 70)
print(f"Total cells: {len(nb['cells'])}")
print(f"File: session3_theory_interactive.ipynb")
print("\\nNotebook includes:")
print("  ✓ Part 1: Build Perceptron (20 min)")
print("  ✓ Part 2: Activation Functions (15 min)")
print("  ✓ Part 3: Neural Networks (20 min)")
print("  ✓ Part 4: Convolution Operations (20 min)")
print("  ✓ Part 5: CNN Architecture Exploration (15 min)")
print("\\nTotal duration: ~90 minutes")
print("=" * 70)
