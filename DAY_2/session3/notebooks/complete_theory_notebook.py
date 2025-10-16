"""
Complete Session 3 Theory Notebook - Add Parts 2-5
Activation functions, neural networks, convolutions, and CNN exploration
"""

import nbformat as nbf
import os

# Check if base notebook exists
if os.path.exists('session3_theory_interactive.ipynb'):
    with open('session3_theory_interactive.ipynb', 'r') as f:
        nb = nbf.read(f, as_version=4)
    print("✓ Loading existing notebook...")
else:
    # Create new if doesn't exist
    nb = nbf.v4.new_notebook()
    print("✓ Creating new notebook...")

additional_cells = []

# =============================================================================
# PART 2: ACTIVATION FUNCTIONS
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# Part 2: Activation Functions (15 minutes)

## Why Activation Functions?

Without activation functions, neural networks would just be linear models (like linear regression). Activation functions introduce **non-linearity**, allowing networks to learn complex patterns.

**Analogy:** 
- Linear model: Can only draw straight lines to separate classes
- With activation: Can draw curves, circles, any shape!

---

## 2.1: Implement Common Activation Functions
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Define activation functions
def sigmoid(x):
    \"\"\"
    Sigmoid: σ(x) = 1 / (1 + e^(-x))
    Range: (0, 1)
    Use: Output probabilities, binary classification
    \"\"\"
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow

def relu(x):
    \"\"\"
    ReLU: f(x) = max(0, x)
    Range: [0, ∞)
    Use: Most popular for hidden layers
    \"\"\"
    return np.maximum(0, x)

def tanh(x):
    \"\"\"
    Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)
    Use: Hidden layers, zero-centered
    \"\"\"
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    \"\"\"
    Leaky ReLU: f(x) = x if x > 0 else alpha * x
    Range: (-∞, ∞)
    Use: Solves "dying ReLU" problem
    \"\"\"
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    \"\"\"
    Softmax: Converts vector to probability distribution
    Use: Multi-class classification output
    \"\"\"
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0)

print("✓ Activation functions defined")
print("  Functions: sigmoid, relu, tanh, leaky_relu, softmax")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 2.2: Visualize Activation Functions
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Generate input range
x = np.linspace(-5, 5, 1000)

# Calculate activations
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_leaky_relu = leaky_relu(x)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Sigmoid
axes[0].plot(x, y_sigmoid, linewidth=3, color='blue')
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Sigmoid Function', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Input (z)', fontsize=11)
axes[0].set_ylabel('Output σ(z)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].text(0.5, 0.05, 'Range: (0, 1)\\nUse: Binary classification output', 
             transform=axes[0].transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ReLU
axes[1].plot(x, y_relu, linewidth=3, color='red')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_title('ReLU (Rectified Linear Unit)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Input (z)', fontsize=11)
axes[1].set_ylabel('Output ReLU(z)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].text(0.5, 0.05, 'Range: [0, ∞)\\nUse: Hidden layers (most popular)', 
             transform=axes[1].transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# Tanh
axes[2].plot(x, y_tanh, linewidth=3, color='green')
axes[2].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[2].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[2].set_title('Tanh (Hyperbolic Tangent)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Input (z)', fontsize=11)
axes[2].set_ylabel('Output tanh(z)', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].text(0.5, 0.05, 'Range: (-1, 1)\\nUse: Hidden layers (zero-centered)', 
             transform=axes[2].transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Leaky ReLU
axes[3].plot(x, y_leaky_relu, linewidth=3, color='purple')
axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[3].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[3].set_title('Leaky ReLU', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Input (z)', fontsize=11)
axes[3].set_ylabel('Output Leaky ReLU(z)', fontsize=11)
axes[3].grid(True, alpha=0.3)
axes[3].text(0.5, 0.05, 'Range: (-∞, ∞)\\nUse: Avoids "dying ReLU" problem', 
             transform=axes[3].transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

plt.tight_layout()
plt.show()

print("\\n✓ Activation functions visualized!")
print("\\n📊 Key Observations:")
print("  • Sigmoid: S-shaped curve, squashes to (0,1)")
print("  • ReLU: Simple, fast, most popular")
print("  • Tanh: Similar to sigmoid but zero-centered")
print("  • Leaky ReLU: Allows small negative values")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Compare Derivatives (Gradients)

The **derivative** determines how fast the neuron learns during backpropagation.
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Calculate derivatives
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Compute derivatives
dy_sigmoid = sigmoid_derivative(x)
dy_relu = relu_derivative(x)
dy_tanh = tanh_derivative(x)

# Plot derivatives
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x, dy_sigmoid, linewidth=3, label='Sigmoid derivative', color='blue')
ax.plot(x, dy_relu, linewidth=3, label='ReLU derivative', color='red')
ax.plot(x, dy_tanh, linewidth=3, label='Tanh derivative', color='green')

ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Input (z)', fontsize=12, fontweight='bold')
ax.set_ylabel('Gradient (derivative)', fontsize=12, fontweight='bold')
ax.set_title('Activation Function Derivatives (Gradients for Backpropagation)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.2, 1.2)

plt.tight_layout()
plt.show()

print("\\n✓ Gradients visualized!")
print("\\n🔑 Why ReLU is Popular:")
print("  • Gradient is either 0 or 1 (simple computation)")
print("  • No vanishing gradient for x > 0")
print("  • Much faster than sigmoid/tanh")
print("\\n⚠️ Vanishing Gradient Problem:")
print("  • Sigmoid/Tanh: gradients → 0 for large |x|")
print("  • Deep networks can't learn (gradients disappear)")
print("  • ReLU solves this for positive inputs")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

### 🎯 Key Takeaways - Part 2

✅ **Activation functions introduce non-linearity**  
✅ **ReLU is the default choice** for hidden layers  
✅ **Sigmoid/Softmax for output** layers (probabilities)  
✅ **Derivatives matter** for learning speed  
✅ **Vanishing gradient** is why we prefer ReLU  

---
"""))

# =============================================================================
# PART 3: SIMPLE NEURAL NETWORK
# =============================================================================

additional_cells.append(nbf.v4.new_markdown_cell("""---

# Part 3: Build a Simple Neural Network (20 minutes)

Now let's connect multiple perceptrons to create a **multi-layer neural network**!

## Architecture

```
Input Layer (2 neurons) → Hidden Layer (4 neurons) → Output Layer (1 neuron)
        ↓                        ↓                         ↓
    [NDVI, Texture]         [ReLU activation]      [Sigmoid activation]
```

This is a **2-4-1 network**: 2 inputs, 4 hidden neurons, 1 output.

---

## 3.1: Implement Neural Network Class
"""))

additional_cells.append(nbf.v4.new_code_cell("""class SimpleNeuralNetwork:
    \"\"\"
    2-layer neural network with one hidden layer
    \"\"\"
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        \"\"\"
        Initialize network with random weights
        \"\"\"
        # Layer 1: input → hidden
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        
        # Layer 2: hidden → output
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        self.learning_rate = learning_rate
        self.losses = []
    
    def forward(self, X):
        \"\"\"
        Forward propagation
        \"\"\"
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)  # Hidden layer uses ReLU
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # Output layer uses Sigmoid
        
        return self.a2
    
    def backward(self, X, y):
        \"\"\"
        Backpropagation (gradient calculation)
        \"\"\"
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0)
        
        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=1000):
        \"\"\"
        Train the network
        \"\"\"
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(predictions + 1e-8) + 
                           (1 - y) * np.log(1 - predictions + 1e-8))
            self.losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            if (epoch + 1) % 200 == 0:
                accuracy = np.mean((predictions > 0.5).flatten() == y) * 100
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.1f}%")
    
    def predict(self, X):
        \"\"\"
        Make predictions
        \"\"\"
        probabilities = self.forward(X)
        return (probabilities > 0.5).astype(int).flatten()

print("✓ Neural Network class defined")
print("  Architecture: Input → Hidden (ReLU) → Output (Sigmoid)")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

## 3.2: Train Neural Network

Let's train on the same forest/non-forest data and compare with the perceptron!
"""))

additional_cells.append(nbf.v4.new_code_cell("""# Create and train neural network
print("Training 2-layer Neural Network...")
print("=" * 60)
print("Architecture: 2 inputs → 4 hidden (ReLU) → 1 output (Sigmoid)")
print("=" * 60)

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
nn.train(X_train, y_train, epochs=1000)

print("=" * 60)
print("\\n✓ Training complete!")

# Final accuracy
final_predictions = nn.predict(X_train)
final_accuracy = np.mean(final_predictions == y_train) * 100
print(f"\\nFinal Training Accuracy: {final_accuracy:.1f}%")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""### Compare: Perceptron vs Neural Network"""))

additional_cells.append(nbf.v4.new_code_cell("""# Create comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Learning curves comparison
axes[0].plot(perceptron.errors, label='Perceptron (MSE)', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Error', fontsize=11, fontweight='bold')
axes[0].set_title('Perceptron Learning Curve', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(nn.losses, label='Neural Network (Cross-Entropy)', linewidth=2, 
             alpha=0.7, color='darkgreen')
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
axes[1].set_title('Neural Network Learning Curve', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 2: Decision boundaries
x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
y_min, y_max = X_train[:, 1].min() - 5, X_train[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Neural network predictions
Z_nn = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_nn = Z_nn.reshape(xx.shape)

axes[2].contourf(xx, yy, Z_nn, alpha=0.3, cmap='RdYlGn', levels=[0, 0.5, 1])
axes[2].contour(xx, yy, Z_nn, colors='black', linewidths=2, levels=[0.5])
axes[2].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                c='darkgreen', s=80, alpha=0.8, edgecolors='black', label='Forest')
axes[2].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                c='orange', s=80, alpha=0.8, edgecolors='black', label='Non-Forest')
axes[2].set_xlabel('NDVI', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Texture', fontsize=11, fontweight='bold')
axes[2].set_title('Neural Network Decision Boundary', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✓ Comparison complete!")
print(f"\\nPerceptron accuracy: {np.mean(perceptron.predict(X_train) == y_train)*100:.1f}%")
print(f"Neural Network accuracy: {final_accuracy:.1f}%")
print("\\n💡 Neural network can learn more complex decision boundaries!")"""))

additional_cells.append(nbf.v4.new_markdown_cell("""---

### 🎯 Key Takeaways - Part 3

✅ **Multi-layer networks** learn complex patterns  
✅ **Hidden layers** create feature representations  
✅ **Different activations** for different layers  
✅ **Backpropagation** trains all layers together  
✅ **Deeper ≠ always better** for simple problems  

**Next:** Apply these concepts to images using convolution!

---
"""))

# Add all new cells to notebook
nb['cells'].extend(additional_cells)

# Write updated notebook
with open('session3_theory_interactive.ipynb', 'w') as f:
    nbf.write(nb, f)

print("\\n✓ Session 3 Notebook Extended (Parts 2-3)!")
print(f"  Total cells: {len(nb['cells'])}")
print("  Next: Adding Parts 4-5 (Convolutions & CNN Exploration)...")
