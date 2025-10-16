# Session 3 Jupyter Notebook Outlines

## Overview

This document provides detailed outlines for the two interactive Jupyter notebooks for Session 3. These outlines specify the content, code, and exercises for each cell, enabling instructors or developers to create the complete notebooks.

---

## Notebook 1: session3_theory_STUDENT.ipynb

**Duration:** 45 minutes
**Objective:** Build intuition for neural networks through hands-on NumPy implementations
**Total Cells:** ~50 (25 markdown, 25 code)

### Part 1: The Perceptron (15 minutes, ~15 cells)

#### Cell 1 (Markdown)
```markdown
# Session 3: Neural Network Fundamentals

## Part 1: The Perceptron - Building Block of Neural Networks

### Objectives:
- Understand the mathematical foundation of artificial neurons
- Build a perceptron from scratch using NumPy
- Visualize decision boundaries
- Train a perceptron with gradient descent

### What is a Perceptron?
The simplest artificial neuron, inspired by biological neurons:
1. **Inputs**: Features (x₁, x₂, ..., xₙ)
2. **Weights**: Learned parameters (w₁, w₂, ..., wₙ)
3. **Weighted sum**: z = Σ(wᵢxᵢ) + b
4. **Activation**: Output = f(z)

**EO Example**: Classify "Water vs Non-Water" using NDVI and NDWI
```

#### Cell 2 (Code) - Setup
```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

print("Libraries imported successfully!")
print(f"NumPy version: {np.__version__}")
```

#### Cell 3 (Markdown)
```markdown
### Generate Synthetic EO Data

We'll create a 2D dataset simulating:
- **Feature 1**: NDVI (Normalized Difference Vegetation Index)
- **Feature 2**: NDWI (Normalized Difference Water Index)
- **Classes**: Water (0) vs Non-Water (1)

**Real-world analogy**:
- Water has low NDVI (~0) and high NDWI (~0.5-1.0)
- Vegetation has high NDVI (~0.7-0.9) and low NDWI (~-0.5)
```

#### Cell 4 (Code) - Generate Data
```python
# Generate 2D classification dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

# Scale to realistic NDVI/NDWI ranges
X[:, 0] = X[:, 0] * 0.3 + 0.5  # NDVI: 0.2 to 0.8
X[:, 1] = X[:, 1] * 0.4   # NDWI: -0.4 to 0.4

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nFeature ranges:")
print(f"  NDVI: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
print(f"  NDWI: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
```

#### Cell 5 (Code) - Visualize Data
```python
# Visualize the dataset
plt.figure(figsize=(10, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Water', alpha=0.6, s=50)
plt.scatter(X[y==1, 0], X[y==1, 1], c='green', label='Non-Water', alpha=0.6, s=50)
plt.xlabel('NDVI (Vegetation Index)', fontsize=12)
plt.ylabel('NDWI (Water Index)', fontsize=12)
plt.title('Synthetic EO Data: Water vs Non-Water Classification', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# TODO: What do you notice about the separability of classes?
# Answer: Classes are linearly separable (perfect for perceptron!)
```

#### Cell 6 (Markdown)
```markdown
### Perceptron Class Implementation

**Mathematical Definition**:
```
y = f(w₁x₁ + w₂x₂ + b)
```

Where:
- `w₁, w₂`: Weights (learned during training)
- `b`: Bias term (shifts decision boundary)
- `f`: Activation function (step function for binary classification)

**Training Rule** (Gradient Descent):
```
w_new = w_old - α × error × input
b_new = b_old - α × error
```

Where:
- `α`: Learning rate (step size)
- `error`: (predicted - actual)
```

#### Cell 7 (Code) - Perceptron Class
```python
class Perceptron:
    """Simple perceptron for binary classification"""

    def __init__(self, input_dim, learning_rate=0.01):
        """
        Initialize perceptron with random weights

        Args:
            input_dim (int): Number of input features
            learning_rate (float): Learning rate for gradient descent
        """
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate

    def activation(self, z):
        """Step activation function: 1 if z > 0, else 0"""
        return np.where(z > 0, 1, 0)

    def predict(self, X):
        """
        Make predictions

        Args:
            X (np.array): Input features, shape (n_samples, n_features)

        Returns:
            np.array: Predictions (0 or 1), shape (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def fit(self, X, y, epochs=100):
        """
        Train perceptron using gradient descent

        Args:
            X (np.array): Training features
            y (np.array): Training labels
            epochs (int): Number of training iterations
        """
        self.losses = []

        for epoch in range(epochs):
            # Forward pass
            predictions = self.predict(X)

            # Calculate error
            errors = y - predictions

            # Update weights and bias
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * np.sum(errors)

            # Track loss (mean absolute error)
            loss = np.mean(np.abs(errors))
            self.losses.append(loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

        print(f"\nFinal weights: {self.weights}")
        print(f"Final bias: {self.bias:.4f}")

print("Perceptron class defined successfully!")
```

#### Cell 8 (Code) - Train Perceptron
```python
# Create and train perceptron
model = Perceptron(input_dim=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=100)

# Evaluate on test set
y_pred_test = model.predict(X_test)
accuracy = np.mean(y_pred_test == y_test)
print(f"\nTest Accuracy: {accuracy:.2%}")
```

#### Cell 9 (Code) - Visualize Training Loss
```python
# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(model.losses, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.title('Perceptron Training Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# TODO: What does the loss curve tell you about convergence?
```

#### Cell 10 (Code) - Visualize Decision Boundary
```python
def plot_decision_boundary(model, X, y):
    """Plot decision boundary learned by perceptron"""

    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(12, 6))

    # Decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'green'])

    # Data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Water', edgecolors='k', s=60)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='green', label='Non-Water', edgecolors='k', s=60)

    # Decision boundary line (w₁x₁ + w₂x₂ + b = 0)
    if model.weights[1] != 0:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(model.weights[0] * x_line + model.bias) / model.weights[1]
        plt.plot(x_line, y_line, 'r--', linewidth=2, label='Decision Boundary')

    plt.xlabel('NDVI', fontsize=12)
    plt.ylabel('NDWI', fontsize=12)
    plt.title('Perceptron Decision Boundary', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_decision_boundary(model, X_test, y_test)

# TODO: How would you interpret this decision boundary for real Sentinel-2 imagery?
```

#### Cells 11-15 (Exercise): TODO - Experiment with Learning Rate
```markdown
### 🎯 EXERCISE 1: Effect of Learning Rate

Try training the perceptron with different learning rates and observe:
- Convergence speed
- Final accuracy
- Decision boundary stability

**TODO**: Complete the code below
```

```python
learning_rates = [0.001, 0.01, 0.1, 0.5]

# TODO: Train 4 perceptrons with different learning rates
# Plot their loss curves on the same plot
# Which learning rate works best?

# SOLUTION (hidden from students, available in INSTRUCTOR notebook):
plt.figure(figsize=(12, 6))

for lr in learning_rates:
    model = Perceptron(input_dim=2, learning_rate=lr)
    model.fit(X_train, y_train, epochs=100)
    plt.plot(model.losses, label=f'LR = {lr}', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Effect of Learning Rate on Convergence', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

### Part 2: Activation Functions (10 minutes, ~10 cells)

#### Cell 16 (Markdown)
```markdown
## Part 2: Activation Functions - Adding Non-Linearity

### Why Activation Functions?

Without activation functions, neural networks are just linear transformations!

**Problem**: Stacking linear layers yields another linear layer:
```
h₁ = W₁x
h₂ = W₂h₁ = W₂(W₁x) = (W₂W₁)x = W_combined x
```

**Solution**: Add non-linear activation functions between layers!

### Common Activation Functions:

1. **Sigmoid**: σ(z) = 1 / (1 + e^(-z))
   - Range: (0, 1)
   - Use: Output layer for binary classification

2. **ReLU**: ReLU(z) = max(0, z)
   - Range: [0, ∞)
   - Use: Hidden layers (most popular!)

3. **Tanh**: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
   - Range: (-1, 1)
   - Use: When zero-centered outputs needed

4. **Softmax**: softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
   - Range: (0, 1), sums to 1
   - Use: Multi-class classification output
```

#### Cell 17 (Code) - Implement Activation Functions
```python
def sigmoid(z):
    """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)

def tanh(z):
    """Tanh activation: (e^z - e^(-z)) / (e^z + e^(-z))"""
    return np.tanh(z)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: max(αz, z)"""
    return np.where(z > 0, z, alpha * z)

def softmax(z):
    """Softmax activation for multi-class classification"""
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

print("Activation functions implemented!")
```

#### Cell 18 (Code) - Visualize Activation Functions
```python
# Generate input range
z = np.linspace(-5, 5, 200)

# Calculate activations
sigmoid_out = sigmoid(z)
relu_out = relu(z)
tanh_out = tanh(z)
leaky_relu_out = leaky_relu(z)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sigmoid
axes[0, 0].plot(z, sigmoid_out, 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid: σ(z) = 1 / (1 + e^(-z))', fontsize=12)
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('σ(z)')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(0, color='k', linewidth=0.5)
axes[0, 0].axvline(0, color='k', linewidth=0.5)

# ReLU
axes[0, 1].plot(z, relu_out, 'r-', linewidth=2)
axes[0, 1].set_title('ReLU: max(0, z)', fontsize=12)
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('ReLU(z)')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].axhline(0, color='k', linewidth=0.5)
axes[0, 1].axvline(0, color='k', linewidth=0.5)

# Tanh
axes[1, 0].plot(z, tanh_out, 'g-', linewidth=2)
axes[1, 0].set_title('Tanh: (e^z - e^(-z)) / (e^z + e^(-z))', fontsize=12)
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('tanh(z)')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axhline(0, color='k', linewidth=0.5)
axes[1, 0].axvline(0, color='k', linewidth=0.5)

# Leaky ReLU
axes[1, 1].plot(z, leaky_relu_out, 'm-', linewidth=2)
axes[1, 1].set_title('Leaky ReLU: max(0.01z, z)', fontsize=12)
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('Leaky ReLU(z)')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(0, color='k', linewidth=0.5)
axes[1, 1].axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

#### Cell 19 (Markdown)
```markdown
### Activation Function Properties

| Function | Range | Gradient (derivative) | Pros | Cons | EO Use Case |
|----------|-------|----------------------|------|------|-------------|
| **Sigmoid** | (0, 1) | σ'(z) = σ(z)(1-σ(z)) | Smooth, interpretable as probability | Vanishing gradient, slow convergence | Cloud probability output |
| **ReLU** | [0, ∞) | 1 if z>0, else 0 | Fast, sparse, no vanishing gradient | Dead ReLU (neurons stuck at 0) | Hidden layers (most CNNs) |
| **Tanh** | (-1, 1) | 1 - tanh²(z) | Zero-centered, stronger gradients than sigmoid | Still suffers from vanishing gradient | Older networks, RNNs |
| **Leaky ReLU** | (-∞, ∞) | α if z<0, 1 if z>0 | No dead neurons | Extra hyperparameter (α) | When ReLU causes dead neurons |

**Key Insight**: ReLU is the default choice for hidden layers in modern CNNs due to simplicity and effectiveness!
```

#### Cell 20 (Code) - Apply to Sentinel-2 Reflectance Values
```python
# Simulate Sentinel-2 reflectance values (typical range: 0-1)
sentinel2_values = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2])

print("Sentinel-2 Reflectance Values (0-1 range):")
print(sentinel2_values)
print("\nAfter Activations:")
print(f"Sigmoid:     {sigmoid(sentinel2_values)}")
print(f"ReLU:        {relu(sentinel2_values)}")
print(f"Tanh:        {tanh(sentinel2_values)}")
print(f"Leaky ReLU:  {leaky_relu(sentinel2_values)}")

# TODO: Which activation preserves the most information for EO data?
# Answer: ReLU (doesn't compress to 0-1 range like sigmoid/tanh)
```

#### Cells 21-25 (Exercise): TODO - Compare Activations on Real Data
```markdown
### 🎯 EXERCISE 2: Activation Functions on NDVI Time Series

You have NDVI time series data for a rice field (120 days):
- Days 0-30: Land preparation (NDVI ~0.2)
- Days 30-60: Transplanting (NDVI 0.2 → 0.6)
- Days 60-90: Vegetative (NDVI 0.6 → 0.9)
- Days 90-120: Maturity (NDVI 0.9 → 0.4)

**TODO**: Apply different activations and visualize the effect
```

---

### Part 3: Multi-Layer Network (15 minutes, ~15 cells)

#### Cell 26 (Markdown)
```markdown
## Part 3: Multi-Layer Networks - Learning Complex Patterns

### Why Multiple Layers?

**Single perceptron limitation**: Can only learn linearly separable patterns

**XOR Problem** (classic example):
```
Input (x₁, x₂) | Output
   (0, 0)      |   0
   (0, 1)      |   1
   (1, 0)      |   1
   (1, 1)      |   0
```

**Not linearly separable!** No single line can separate classes.

**Solution**: Stack multiple layers with non-linear activations!

### Multi-Layer Perceptron (MLP) Architecture:

```
Input Layer (2 features: NDVI, NDWI)
    ↓
Hidden Layer 1 (4 neurons, ReLU)
    ↓
Hidden Layer 2 (2 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid for binary classification)
```
```

#### Cell 27 (Code) - Implement 2-Layer Network
```python
class TwoLayerNetwork:
    """Simple 2-layer neural network for binary classification"""

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        """
        Initialize network with random weights

        Architecture: Input → Hidden (ReLU) → Output (Sigmoid)
        """
        # Layer 1: Input → Hidden
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        # Layer 2: Hidden → Output
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        self.learning_rate = learning_rate

    def forward(self, X):
        """Forward propagation through network"""

        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
        self.a1 = relu(self.z1)                 # ReLU activation

        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)              # Sigmoid activation

        return self.a2

    def backward(self, X, y, output):
        """Backpropagation: Compute gradients"""

        m = X.shape[0]  # Number of samples

        # Output layer gradients
        dz2 = output - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        """Gradient descent weight update"""

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(self, X, y, epochs=1000, verbose=True):
        """Train network"""

        self.losses = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1-y) * np.log(1-output + 1e-8))
            self.losses.append(loss)

            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, output)

            # Update weights
            self.update_weights(dW1, db1, dW2, db2)

            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

        if verbose:
            print(f"Final Loss: {loss:.4f}")

    def predict(self, X):
        """Make predictions (threshold at 0.5)"""
        output = self.forward(X)
        return (output > 0.5).astype(int).flatten()

print("TwoLayerNetwork class implemented!")
```

#### Cell 28 (Code) - Train 2-Layer Network
```python
# Create and train network
model = TwoLayerNetwork(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.1)
model.fit(X_train, y_train, epochs=2000, verbose=True)

# Evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.2%}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(model.losses, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
plt.title('Training Loss: 2-Layer Network', fontsize=14)
plt.grid(alpha=0.3)
plt.show()
```

#### Cell 29 (Code) - Visualize Decision Boundary (Non-Linear)
```python
def plot_nn_decision_boundary(model, X, y):
    """Plot decision boundary for neural network"""

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'green'])
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Water', edgecolors='k', s=60)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='green', label='Non-Water', edgecolors='k', s=60)

    plt.xlabel('NDVI', fontsize=12)
    plt.ylabel('NDWI', fontsize=12)
    plt.title('2-Layer Network Decision Boundary (Non-Linear!)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_nn_decision_boundary(model, X_test, y_test)

# TODO: Compare this to the perceptron decision boundary. What's different?
# Answer: Non-linear boundary can fit more complex patterns!
```

#### Cells 30-40 (Exercises and Visualizations)
- Exercise: Experiment with hidden layer size
- Visualization: Weight distributions
- Exercise: Effect of learning rate on multi-layer networks
- Visualization: Activation patterns in hidden layer

---

### Part 4: Learning Rate Exploration (5 minutes, ~10 cells)

#### Cells 41-50: Interactive learning rate comparison
- Train networks with lr = [0.001, 0.01, 0.1, 1.0]
- Visualize convergence differences
- Explore overfitting vs underfitting
- Connection to Session 4 (Adam optimizer)

---

## Notebook 2: session3_cnn_operations_STUDENT.ipynb

**Duration:** 55 minutes
**Objective:** Understand convolution operations and CNN components through hands-on implementations
**Total Cells:** ~45 (20 markdown, 25 code)

### Part 1: Manual Convolution on Sentinel-2 (15 minutes, ~12 cells)

#### Cell 1 (Markdown)
```markdown
# Session 3: CNN Operations for Earth Observation

## Part 1: Manual Convolution - The Heart of CNNs

### Objectives:
- Understand convolution as a sliding filter operation
- Implement convolution from scratch using NumPy
- Apply edge detection filters to Sentinel-2 imagery
- Visualize feature maps

### What is Convolution?

**Mathematical Definition**:
```
(I * K)(i,j) = ΣΣ I(i+m, j+n) × K(m,n)
```

Where:
- `I`: Input image
- `K`: Filter/kernel (small matrix, e.g., 3×3)
- `*`: Convolution operator

**Intuition**: Slide filter across image, compute element-wise product and sum at each position.

**Why it works**: Filters detect patterns (edges, textures) regardless of position!
```

#### Cell 2 (Code) - Load Sample Sentinel-2 Patch
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from skimage import data, color
import requests
from io import BytesIO
from PIL import Image

# For this demo, we'll create a synthetic Sentinel-2 NIR band
# In real Session 4, you'll load actual Sentinel-2 imagery from GEE

# Create synthetic land cover scene (256×256)
np.random.seed(42)
image = np.zeros((256, 256))

# Forest (high NIR)
image[50:150, 50:150] = 0.8 + np.random.normal(0, 0.05, (100, 100))

# Water (low NIR)
image[160:230, 30:200] = 0.1 + np.random.normal(0, 0.02, (70, 170))

# Urban (medium NIR)
image[20:80, 180:240] = 0.4 + np.random.normal(0, 0.03, (60, 60))

# Clip to valid range
image = np.clip(image, 0, 1)

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.colorbar(label='NIR Reflectance')
plt.title('Synthetic Sentinel-2 NIR Band (simulating forest, water, urban)', fontsize=14)
plt.axis('off')
plt.show()

print(f"Image shape: {image.shape}")
print(f"Value range: [{image.min():.2f}, {image.max():.2f}]")
```

#### Cell 3 (Code) - Implement Manual Convolution
```python
def manual_convolution(image, kernel):
    """
    Manually implement 2D convolution (for learning purposes)

    Args:
        image (np.array): Input image (H×W)
        kernel (np.array): Filter (typically 3×3 or 5×5)

    Returns:
        np.array: Feature map (H×W) - same size with zero padding
    """

    # Get dimensions
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate padding (to maintain same size)
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad image with zeros
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize output feature map
    output = np.zeros((img_h, img_w))

    # Perform convolution (sliding window)
    for i in range(img_h):
        for j in range(img_w):
            # Extract patch
            patch = padded_image[i:i+kernel_h, j:j+kernel_w]

            # Element-wise multiplication and sum
            output[i, j] = np.sum(patch * kernel)

    return output

print("Manual convolution implemented!")
print("Note: This is O(N²) slow - production CNNs use FFT-based convolution (1000× faster on GPU)")
```

#### Cell 4 (Code) - Define Edge Detection Filters
```python
# Classic edge detection filters

# Horizontal edge detector (detects horizontal lines)
horizontal_edge_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Vertical edge detector
vertical_edge_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Sobel X (horizontal edges)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Sobel Y (vertical edges)
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Laplacian (detects edges in all directions)
laplacian = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])

# Gaussian blur (smoothing filter)
gaussian = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

print("Edge detection filters defined!")
```

#### Cell 5 (Code) - Apply Filters and Visualize
```python
# Apply filters to our Sentinel-2 NIR image
horizontal_edges = manual_convolution(image, horizontal_edge_kernel)
vertical_edges = manual_convolution(image, vertical_edge_kernel)
sobel_edges = np.sqrt(manual_convolution(image, sobel_x)**2 + manual_convolution(image, sobel_y)**2)
laplacian_edges = manual_convolution(image, laplacian)
gaussian_blurred = manual_convolution(image, gaussian)

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original NIR Band', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(horizontal_edges, cmap='gray')
axes[0, 1].set_title('Horizontal Edges (water/forest boundary)', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(vertical_edges, cmap='gray')
axes[0, 2].set_title('Vertical Edges', fontsize=12)
axes[0, 2].axis('off')

axes[1, 0].imshow(sobel_edges, cmap='gray')
axes[1, 0].set_title('Sobel Edges (all directions)', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(laplacian_edges, cmap='gray')
axes[1, 1].set_title('Laplacian Edges', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(gaussian_blurred, cmap='gray')
axes[1, 2].set_title('Gaussian Blur (smoothed)', fontsize=12)
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# TODO: Which filter best highlights the water/forest boundary?
# Answer: Horizontal edge detector! (forest-water boundary is horizontal)
```

### Part 2: Max Pooling (10 minutes, ~8 cells)

### Part 3: Architecture Comparison (15 minutes, ~12 cells)

### Part 4: Feature Map Visualization (15 minutes, ~13 cells)

---

## Summary

These notebook outlines provide:
1. **Complete cell-by-cell structure** with markdown and code
2. **Progressive difficulty** from simple perceptron to multi-layer networks
3. **EO-specific examples** using NDVI, NDWI, Sentinel-2 imagery
4. **TODO exercises** for students to complete
5. **Visualizations** at each step for intuition building
6. **Connections to Session 4** where TensorFlow/Keras will automate everything

**Next Steps:**
- Instructors can copy these outlines into Jupyter notebooks
- Fill in remaining exercises and visualizations
- Add solutions in separate INSTRUCTOR notebooks
- Test all code cells execute successfully in Colab
- Upload to course repository with Colab badges

---

**Estimated Development Time:**
- Notebook 1: 4-6 hours (complete implementation + testing)
- Notebook 2: 4-6 hours (complete implementation + testing)
- Total: 8-12 hours for full interactive notebooks

**Contact:** training@copphil.org
