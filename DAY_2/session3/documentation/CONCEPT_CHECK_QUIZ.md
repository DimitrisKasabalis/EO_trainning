# Session 3: Concept Check Quiz
**Deep Learning & CNN Theory Assessment**

---

## Instructions

- **Total Questions:** 20
- **Time Limit:** 30 minutes (recommended)
- **Passing Score:** 14/20 (70%)
- **Format:** Multiple choice (4 options each)
- **Open Notes:** Yes (this is for learning, not testing)

**Scoring:**
- Review answers immediately after completing
- Revisit topics where you scored <70%
- Retake after reviewing materials

---

## Questions

### Part A: Neural Network Fundamentals (Questions 1-6)

#### Question 1
**What is the primary purpose of an activation function in a neural network?**

A) To initialize weights randomly  
B) To introduce non-linearity into the model  
C) To reduce the number of parameters  
D) To normalize input data  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

Activation functions introduce non-linearity, allowing neural networks to learn complex patterns. Without activation functions, even deep networks would behave like linear models.

**Why others are wrong:**
- A: Weight initialization is separate from activation
- C: Activation functions don't reduce parameters
- D: Normalization is done by BatchNorm or preprocessing
</details>

---

#### Question 2
**Which activation function is most commonly used in hidden layers of modern CNNs?**

A) Sigmoid  
B) Tanh  
C) ReLU  
D) Softmax  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**ReLU (Rectified Linear Unit)** is the default choice for hidden layers because:
- Fast computation: f(x) = max(0, x)
- Solves vanishing gradient problem
- Sparse activation (many zeros)

**When to use others:**
- Sigmoid: Output layer for binary classification
- Tanh: Hidden layers when zero-centered outputs needed
- Softmax: Output layer for multi-class classification
</details>

---

#### Question 3
**What problem do skip connections (residual connections) in ResNet solve?**

A) Overfitting  
B) Vanishing gradient  
C) Slow inference  
D) High memory usage  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

Skip connections allow gradients to flow directly through the network, solving the **vanishing gradient problem** that prevents very deep networks from training effectively.

**Formula:** `output = F(x) + x`

This enables networks with 50, 101, or even 152 layers to train successfully.
</details>

---

#### Question 4
**In forward propagation, data flows:**

A) From output layer to input layer  
B) From input layer to output layer  
C) Bidirectionally through all layers  
D) Only through convolutional layers  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

**Forward propagation:** Input → Hidden Layers → Output

**Process:**
1. Feed input data
2. Compute weighted sums and activations
3. Pass through each layer sequentially
4. Generate predictions at output

**Backward propagation** (training only) flows the opposite direction with gradients.
</details>

---

#### Question 5
**What is the purpose of the learning rate in gradient descent?**

A) To determine batch size  
B) To control the step size of weight updates  
C) To set the number of neurons  
D) To initialize weights  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

Learning rate controls how much weights change during each update:
- **Too high:** Training unstable, overshoots minimum
- **Too low:** Training very slow, may get stuck
- **Typical values:** 0.001 - 0.01

**Formula:** `new_weight = old_weight - learning_rate × gradient`
</details>

---

#### Question 6
**Which optimizer is most commonly recommended for training CNNs?**

A) SGD (Stochastic Gradient Descent)  
B) Momentum  
C) Adam  
D) RMSprop  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**Adam (Adaptive Moment Estimation)** is the default choice because:
- Adaptive learning rates per parameter
- Combines benefits of Momentum and RMSprop
- Works well out-of-the-box
- Requires minimal tuning

**When to use others:**
- SGD with momentum: When you have lots of time to tune
- RMSprop: Older, less common now
</details>

---

### Part B: Convolutional Operations (Questions 7-12)

#### Question 7
**What is the primary purpose of convolution in CNNs?**

A) To reduce image size  
B) To extract spatial features  
C) To normalize pixel values  
D) To add noise for regularization  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

Convolution extracts **spatial features** by sliding filters across images:
- Detects patterns (edges, textures, objects)
- Preserves spatial relationships
- Parameter sharing (same filter everywhere)
- Translation invariance

**Size reduction** is done by pooling, not convolution (with padding='same').
</details>

---

#### Question 8
**A 3×3 convolutional filter with 64 output channels applied to an input with 32 channels has how many parameters (including bias)?**

A) 288  
B) 2,304  
C) 18,432  
D) 18,496  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: D**

**Formula:** `(kernel_h × kernel_w × in_channels + 1) × out_channels`

**Calculation:**
- `(3 × 3 × 32 + 1) × 64`
- `(288 + 1) × 64`
- `289 × 64 = 18,496`

The "+1" is the bias term per output channel.
</details>

---

#### Question 9
**What does MaxPooling2D(pool_size=(2,2)) do?**

A) Doubles the spatial dimensions  
B) Halves the spatial dimensions  
C) Doubles the number of channels  
D) Halves the number of channels  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

Max pooling with 2×2 window **halves spatial dimensions**:
- Input: 64×64×128 → Output: 32×32×128
- Takes maximum value in each 2×2 window
- Reduces spatial size by factor of 2
- Channels remain unchanged

**Benefits:**
- Translation invariance
- Reduces parameters
- Increases receptive field
</details>

---

#### Question 10
**In a CNN, early layers typically learn ___, while deeper layers learn ___.**

A) Objects; edges  
B) Edges; objects  
C) Colors; shapes  
D) Textures; colors  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

**Feature Hierarchy:**
- **Layer 1-2 (early):** Edges, corners, simple patterns
- **Layer 3-4 (middle):** Textures, patterns, combinations
- **Layer 5+ (deep):** Object parts, semantic features

**For Sentinel-2:**
- Early: Water/land boundaries, spectral edges
- Middle: Vegetation textures, urban patterns
- Deep: Forest stands, agricultural fields
</details>

---

#### Question 11
**What is the purpose of padding='same' in a convolutional layer?**

A) To reduce overfitting  
B) To increase accuracy  
C) To preserve spatial dimensions  
D) To add dropout  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**padding='same'** adds zeros around borders to **preserve dimensions**:
- Input: 64×64 → Conv 3×3 with 'same' → Output: 64×64
- Without padding ('valid'): 64×64 → 62×62

**Why it matters:**
- Prevents information loss at borders
- Makes architecture design easier
- Standard practice in modern CNNs
</details>

---

#### Question 12
**Which operation provides translation invariance in CNNs?**

A) Convolution  
B) Batch normalization  
C) Pooling  
D) Dropout  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**Pooling** provides translation invariance:
- Detects features regardless of exact position
- Small shifts in input don't drastically change output
- Example: Forest detected whether at pixel (10,10) or (12,11)

**Convolution** provides some invariance through weight sharing, but pooling is the primary mechanism.
</details>

---

### Part C: CNN Architectures (Questions 13-17)

#### Question 13
**Which CNN architecture introduced skip connections?**

A) AlexNet  
B) VGG  
C) ResNet  
D) LeNet  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**ResNet (2015)** introduced skip connections (residual connections):
- Enables very deep networks (50-152 layers)
- Solves vanishing gradient problem
- Formula: `F(x) + x` (add input to output)

**Impact:** Revolutionized deep learning, enabling practical 100+ layer networks.
</details>

---

#### Question 14
**VGG16 uses how many convolutional and fully connected layers in total?**

A) 11  
B) 13  
C) 16  
D) 19  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**VGG16** has 16 layers with trainable parameters:
- 13 convolutional layers (all 3×3)
- 3 fully connected layers
- Total: 16 layers (hence the name)

**Architecture philosophy:** Simple, uniform structure using only 3×3 filters.
</details>

---

#### Question 15
**U-Net architecture is specifically designed for:**

A) Image classification  
B) Object detection  
C) Semantic segmentation  
D) Style transfer  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**U-Net** is designed for **semantic segmentation** (pixel-level classification):

**Structure:**
- Encoder (contracting path): Captures context
- Decoder (expanding path): Enables precise localization
- Skip connections: Combines low-level and high-level features

**EO Applications:**
- Building footprint extraction
- Forest boundary mapping
- Agricultural field delineation
- Flood extent mapping
</details>

---

#### Question 16
**Compared to VGG16, ResNet50 has:**

A) More parameters and higher accuracy  
B) Fewer parameters and higher accuracy  
C) More parameters and lower accuracy  
D) Fewer parameters and lower accuracy  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

**Comparison:**
- VGG16: ~138M parameters, ~92% ImageNet accuracy
- ResNet50: ~25M parameters, ~95% ImageNet accuracy

ResNet achieves better accuracy with fewer parameters due to:
- Skip connections
- Deeper architecture (50 vs 16 layers)
- More efficient design
</details>

---

#### Question 17
**Which architecture would you choose for classifying 64×64 Sentinel-2 patches into land cover classes?**

A) U-Net  
B) YOLO  
C) ResNet50  
D) Faster R-CNN  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**ResNet50** is ideal for scene classification:
- Input: Fixed-size images
- Output: Single class per image
- Strong feature extraction
- Transfer learning available

**Why not others:**
- U-Net: For segmentation (pixel-level)
- YOLO/Faster R-CNN: For object detection (bounding boxes)
</details>

---

### Part D: EO Applications & Practical Considerations (Questions 18-20)

#### Question 18
**For Sentinel-2 images with 10 bands, which approach is best for using pre-trained ImageNet models (trained on 3-channel RGB)?**

A) Discard Sentinel-2, use only RGB  
B) Use 1×1 convolution to convert 10 channels to 3  
C) Train from scratch with 10 channels  
D) Pad with zeros to make 10 channels  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

**Best practice:** Use 1×1 convolution adapter:

```python
# Adapter layer
Conv2D(3, (1,1), input_shape=(224, 224, 10))
# Then pre-trained ResNet
base_model = ResNet50(weights='imagenet')
```

This allows transfer learning while using all 10 bands.

**Why not A:** Loses valuable NIR and SWIR information  
**Why not C:** Requires more data and time  
**Why not D:** Doesn't properly adapt channels  
</details>

---

#### Question 19
**You have 500 labeled Sentinel-2 patches for training a land cover classifier. What should you do?**

A) Build a deep CNN with 50+ layers  
B) Use transfer learning from a pre-trained model  
C) Skip CNNs, use Random Forest instead  
D) Generate synthetic data only  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: B**

With limited data (500 samples), **transfer learning** is best:
- Leverage pre-trained weights (ImageNet, EuroSAT)
- Fine-tune on your specific data
- Much better than training from scratch

**Alternatives:**
- C is also reasonable but CNNs can achieve higher accuracy
- Combine with data augmentation to increase effective dataset size

**Not A:** Deep networks need 1000s of samples to train from scratch
</details>

---

#### Question 20
**Which data augmentation technique is most appropriate for Sentinel-2 land cover classification?**

A) Random horizontal flips only  
B) Color jittering (changing RGB values)  
C) 90° rotations and flips  
D) Perspective transforms  

<details>
<summary><b>Answer & Explanation</b></summary>

**Correct Answer: C**

**90° rotations and flips** are ideal for EO because:
- Land cover patterns have no preferred orientation
- Forests look the same from any angle
- Creates 8× more training data (4 rotations × 2 flips)

**Why not others:**
- A: Too limited
- B: Dangerous - changes spectral signatures
- D: Not realistic for nadir satellite imagery

**Safe augmentations for EO:**
- Rotation (90°, 180°, 270°)
- Horizontal/vertical flips
- Brightness adjustment (moderate)
- Small translations
</details>

---

## Answer Key

| Q# | Answer | Topic |
|----|--------|-------|
| 1  | B | Activation functions |
| 2  | C | ReLU activation |
| 3  | B | Skip connections |
| 4  | B | Forward propagation |
| 5  | B | Learning rate |
| 6  | C | Adam optimizer |
| 7  | B | Convolution purpose |
| 8  | D | Parameter calculation |
| 9  | B | Max pooling |
| 10 | B | Feature hierarchy |
| 11 | C | Padding |
| 12 | C | Translation invariance |
| 13 | C | ResNet |
| 14 | C | VGG16 |
| 15 | C | U-Net segmentation |
| 16 | B | ResNet vs VGG |
| 17 | C | Scene classification |
| 18 | B | Transfer learning adapter |
| 19 | B | Limited data strategy |
| 20 | C | EO data augmentation |

---

## Scoring Guide

**18-20 correct (90-100%):** Excellent! You're ready for Session 4.

**14-17 correct (70-85%):** Good understanding. Review missed topics.

**10-13 correct (50-65%):** Some gaps. Reread course materials and retake quiz.

**<10 correct (<50%):** Revisit Session 3 materials thoroughly before proceeding.

---

## Topics to Review Based on Scores

**If you missed questions 1-6:** Review neural network fundamentals
- [Interactive Notebook Part 1-3](../notebooks/session3_theory_interactive.ipynb)
- [Course Page Section B](../../../course_site/day2/sessions/session3.qmd#part-b)

**If you missed questions 7-12:** Review convolution operations
- [Interactive Notebook Part 4](../notebooks/session3_theory_interactive.ipynb)
- [Course Page Section C](../../../course_site/day2/sessions/session3.qmd#part-c)

**If you missed questions 13-17:** Review CNN architectures
- [Interactive Notebook Part 5](../notebooks/session3_theory_interactive.ipynb)
- [Architecture Guide](CNN_ARCHITECTURE_GUIDE.md)

**If you missed questions 18-20:** Review EO-specific applications
- [Course Page Section E-F](../../../course_site/day2/sessions/session3.qmd#part-e)
- [Architecture Guide Section 3](CNN_ARCHITECTURE_GUIDE.md#designing-for-eo)

---

## Additional Practice

**Want more questions?** Try these exercises:

1. **Calculate parameters** for a full CNN architecture
2. **Design a CNN** for a specific Philippine use case
3. **Compare architectures** for different tasks (classification vs segmentation)
4. **Plan data augmentation** strategy for limited datasets

---

## Next Steps

✅ **Passed the quiz?** Congratulations! You're ready for Session 4.

📚 **Need review?** That's normal! Deep learning has many concepts. Take your time.

🎯 **Session 4 Preview:** You'll implement everything you learned:
- Build CNNs with TensorFlow/Keras
- Train on real Palawan data
- Apply transfer learning
- Compare with Random Forest

---

**Quiz Version:** 1.0  
**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 3
