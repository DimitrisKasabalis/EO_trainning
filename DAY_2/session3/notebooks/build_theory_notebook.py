"""
Build Session 3 Interactive Theory Notebook
Comprehensive hands-on exploration of neural networks and CNNs
"""

import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

cells = []

# =============================================================================
# TITLE AND INTRODUCTION
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""# Session 3: Deep Learning & CNN Theory - Interactive Notebook

## From Random Forest to Neural Networks

**Duration:** 90 minutes | **Type:** Interactive Theory | **Difficulty:** Intermediate

---

## 🎯 Learning Objectives

By the end of this notebook, you will:

1. ✅ Build a perceptron from scratch using NumPy
2. ✅ Understand and visualize activation functions
3. ✅ Implement forward propagation manually
4. ✅ Apply convolution operations to Sentinel-2 imagery
5. ✅ Explore pre-trained CNN architectures
6. ✅ Visualize learned feature maps
7. ✅ Understand the transition from RF to deep learning

---

## 📋 Notebook Structure

| Part | Topic | Duration |
|------|-------|----------|
| **1** | Build Perceptron from Scratch | 20 min |
| **2** | Activation Functions | 15 min |
| **3** | Simple Neural Network | 20 min |
| **4** | Convolution Operations | 20 min |
| **5** | CNN Architecture Exploration | 15 min |

---

## 🔑 Key Concepts Preview

**What you already know (from Sessions 1-2):**
- Random Forest classification
- Feature engineering (GLCM, NDVI, temporal)
- Accuracy assessment
- Palawan land cover mapping

**What you'll learn today:**
- How neural networks learn from data
- Why convolution is perfect for images
- How CNNs build feature hierarchies
- When to use CNNs vs Random Forest

---

Let's dive in! 🚀
"""))

# =============================================================================
# SETUP
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Setup and Imports

First, let's import the libraries we'll need.
"""))

cells.append(nbf.v4.new_code_cell("""# Core libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, signal
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# For reproducibility
np.random.seed(42)

print("✓ Libraries imported successfully")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")"""))

# =============================================================================
# PART 1: BUILD PERCEPTRON
# =============================================================================

cells.append(nbf.v4.new_markdown_cell("""---

# Part 1: Build a Perceptron from Scratch (20 minutes)

## What is a Perceptron?

A **perceptron** is the simplest artificial neuron. It:
1. Takes multiple inputs (x₁, x₂, ..., xₙ)
2. Multiplies each by a weight (w₁, w₂, ..., wₙ)
3. Adds a bias term (b)
4. Applies an activation function
5. Outputs a prediction

**Mathematical formula:**
```
z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b
output = activation(z)
```

**Analogy for EO:**
Think of classifying a pixel as "forest" or "not forest":
- x₁ = NDVI value
- x₂ = texture measure
- x₃ = elevation
- Weights determine how important each feature is
- Output: probability of being forest

---

## 1.1: Implement the Perceptron Class
"""))

cells.append(nbf.v4.new_code_cell("""class Perceptron:
    \"\"\"
    Simple perceptron implementation
    \"\"\"
    
    def __init__(self, n_inputs, learning_rate=0.01):
        \"\"\"
        Initialize perceptron with random weights
        
        Parameters:
        -----------
        n_inputs : int
            Number of input features
        learning_rate : float
            Step size for weight updates
        \"\"\"
        # Initialize weights randomly (small values)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        
        # Track training history
        self.errors = []
    
    def sigmoid(self, z):
        \"\"\"
        Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
        Maps any value to range (0, 1)
        \"\"\"
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        \"\"\"
        Make predictions for input data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Binary predictions (0 or 1)
        \"\"\"
        # Calculate weighted sum
        z = np.dot(X, self.weights) + self.bias
        
        # Apply sigmoid activation
        probabilities = self.sigmoid(z)
        
        # Convert to binary (threshold at 0.5)
        predictions = (probabilities >= 0.5).astype(int)
        
        return predictions
    
    def train(self, X, y, epochs=100):
        \"\"\"
        Train perceptron using gradient descent
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels (0 or 1)
        epochs : int
            Number of training iterations
        \"\"\"
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Calculate error
            errors = y - predictions
            
            # Update weights (gradient descent)
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * np.sum(errors)
            
            # Track mean squared error
            mse = np.mean(errors ** 2)
            self.errors.append(mse)
            
            if (epoch + 1) % 20 == 0:
                accuracy = np.mean(self.predict(X) == y) * 100
                print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.4f} - Accuracy: {accuracy:.1f}%")

print("✓ Perceptron class defined")
print("  Methods: __init__, sigmoid, predict, train")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 1.2: Generate Simple Training Data

Let's create a toy dataset that mimics forest classification:
- **Feature 1:** NDVI (high for forest)
- **Feature 2:** Texture contrast (medium for forest)
- **Label:** Forest (1) or Not Forest (0)
"""))

cells.append(nbf.v4.new_code_cell("""# Generate synthetic "forest" vs "non-forest" data
np.random.seed(42)

# Forest: high NDVI (0.6-0.9), medium texture (20-50)
n_forest = 50
forest_ndvi = np.random.uniform(0.6, 0.9, n_forest)
forest_texture = np.random.uniform(20, 50, n_forest)
forest_data = np.column_stack([forest_ndvi, forest_texture])
forest_labels = np.ones(n_forest)

# Non-forest: low NDVI (0.1-0.4), high texture (40-80)
n_non_forest = 50
non_forest_ndvi = np.random.uniform(0.1, 0.4, n_non_forest)
non_forest_texture = np.random.uniform(40, 80, n_non_forest)
non_forest_data = np.column_stack([non_forest_ndvi, non_forest_texture])
non_forest_labels = np.zeros(n_non_forest)

# Combine datasets
X_train = np.vstack([forest_data, non_forest_data])
y_train = np.concatenate([forest_labels, non_forest_labels])

# Shuffle
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"Training data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")
print(f"\\nClass distribution:")
print(f"  Forest (1): {np.sum(y_train == 1)} samples")
print(f"  Non-forest (0): {np.sum(y_train == 0)} samples")"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualize Training Data"""))

cells.append(nbf.v4.new_code_cell("""# Plot the training data
fig, ax = plt.subplots(figsize=(10, 6))

# Separate classes for plotting
forest_mask = y_train == 1
non_forest_mask = y_train == 0

ax.scatter(X_train[forest_mask, 0], X_train[forest_mask, 1], 
           c='darkgreen', s=100, alpha=0.6, edgecolors='black', 
           label='Forest', marker='o')
ax.scatter(X_train[non_forest_mask, 0], X_train[non_forest_mask, 1], 
           c='orange', s=100, alpha=0.6, edgecolors='black', 
           label='Non-Forest', marker='s')

ax.set_xlabel('NDVI (Normalized Difference Vegetation Index)', fontsize=12, fontweight='bold')
ax.set_ylabel('Texture Contrast', fontsize=12, fontweight='bold')
ax.set_title('Forest vs Non-Forest Training Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✓ Training data visualized")
print("  Notice: Forest = high NDVI, moderate texture")
print("          Non-forest = low NDVI, high texture")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 1.3: Train the Perceptron

Now let's train our perceptron to classify forest vs non-forest!
"""))

cells.append(nbf.v4.new_code_cell("""# Create and train perceptron
print("Training perceptron...")
print("=" * 60)

perceptron = Perceptron(n_inputs=2, learning_rate=0.1)
perceptron.train(X_train, y_train, epochs=100)

print("=" * 60)
print("\\n✓ Training complete!")

# Final accuracy
final_predictions = perceptron.predict(X_train)
final_accuracy = np.mean(final_predictions == y_train) * 100
print(f"\\nFinal Training Accuracy: {final_accuracy:.1f}%")

print(f"\\nLearned Weights:")
print(f"  NDVI weight: {perceptron.weights[0]:.4f}")
print(f"  Texture weight: {perceptron.weights[1]:.4f}")
print(f"  Bias: {perceptron.bias:.4f}")"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualize Learning Progress"""))

cells.append(nbf.v4.new_code_cell("""# Plot training curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Error over time
ax1.plot(perceptron.errors, linewidth=2, color='darkred')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax1.set_title('Learning Curve: Error Decreases Over Time', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Decision boundary
# Create mesh for decision boundary
x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
y_min, y_max = X_train[:, 1].min() - 5, X_train[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict for mesh
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision regions
ax2.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[0, 0.5, 1])
ax2.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])

# Plot training points
ax2.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='darkgreen', s=100, alpha=0.8, edgecolors='black', label='Forest')
ax2.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            c='orange', s=100, alpha=0.8, edgecolors='black', label='Non-Forest')

ax2.set_xlabel('NDVI', fontsize=12, fontweight='bold')
ax2.set_ylabel('Texture Contrast', fontsize=12, fontweight='bold')
ax2.set_title('Decision Boundary Learned by Perceptron', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✓ Perceptron successfully learned to separate forest from non-forest!")
print("  The black line shows the decision boundary")
print("  Green region = predicted as forest")
print("  Red region = predicted as non-forest")"""))

cells.append(nbf.v4.new_markdown_cell("""---

### 🎯 Key Takeaways - Part 1

✅ **A perceptron is the building block** of neural networks  
✅ **Weights determine feature importance** (like feature importance in RF)  
✅ **Training adjusts weights** to minimize error  
✅ **Activation functions** map outputs to desired range  
✅ **Decision boundary** separates classes (linear for perceptron)  

**Limitation:** Perceptrons can only learn linear decision boundaries. For complex patterns (like in satellite images), we need deeper networks with non-linear activations!

---
"""))

# Add cells to notebook
nb['cells'] = cells

# Write notebook
with open('session3_theory_interactive.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✓ Session 3 Theory Notebook (Part 1) Created!")
print(f"  Total cells: {len(cells)}")
print(f"  File: session3_theory_interactive.ipynb")
print("\\nNext: Adding Parts 2-5...")
