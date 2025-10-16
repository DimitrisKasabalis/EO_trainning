# DAY 2 Course Content Development Plan
## Advanced AI/ML for Earth Observation – Classification & CNNs

**Document Version:** 1.0
**Date Created:** 2025-10-14
**Course:** CopPhil Advanced Training Program
**Target Audience:** Intermediate learners with Python and basic ML knowledge from DAY 1

---

## Table of Contents

1. [Overall Course Structure](#overall-course-structure)
2. [Session 1: Supervised Classification with Random Forest](#session-1-supervised-classification-with-random-forest)
3. [Session 2: Land Cover Classification Lab (Palawan)](#session-2-land-cover-classification-lab-palawan)
4. [Session 3: Introduction to Deep Learning and CNNs](#session-3-introduction-to-deep-learning-and-cnns)
5. [Session 4: CNN Hands-on Lab](#session-4-cnn-hands-on-lab)
6. [Overall Learning Objectives](#overall-learning-objectives)
7. [Implementation Timeline](#implementation-timeline)
8. [Technical Requirements](#technical-requirements)

---

## Overall Course Structure

**Duration:** Full day (approximately 6-8 hours)
**Focus:** Transition from traditional machine learning (Random Forest) to deep learning (CNNs)
**Philippine Context:** Palawan land cover classification and NRM applications

### Session Overview

| Session | Title | Duration | Type | Key Technologies |
|---------|-------|----------|------|------------------|
| 1 | Supervised Classification with Random Forest | 3 hours | Theory + Hands-on | GEE, Sentinel-2, scikit-learn |
| 2 | Land Cover Classification Lab (Palawan) | 1.5-2 hours | Hands-on | GEE, Sentinel-2, Random Forest |
| 3 | Introduction to Deep Learning and CNNs | 2-2.5 hours | Theory + Interactive | TensorFlow/PyTorch basics |
| 4 | CNN Hands-on Lab | 2.5-3 hours | Hands-on | TensorFlow, PyTorch, EuroSAT |

### Pedagogical Flow

```
Session 1: RF Theory → Session 2: RF Practice (Palawan)
                ↓
Session 3: CNN Theory → Session 4: CNN Practice (EuroSAT/Palawan)
```

---

## Session 1: Supervised Classification with Random Forest

**Duration:** 3 hours (1.5h theory + 1.5h hands-on)
**Learning Objective:** Understand and implement Random Forest classification for Earth Observation data using Google Earth Engine

### Content Components to Create

#### 1. Presentation Materials (45 minutes)

**Topics to Cover:**
- Introduction to supervised classification
- Decision trees fundamentals (visual diagrams)
- Random Forest ensemble method
  - Bootstrap aggregation (bagging)
  - Feature randomness
  - Voting mechanism
- Feature importance and variable selection
- Accuracy assessment and confusion matrices
- Google Earth Engine platform capabilities
- Philippine NRM applications (forestry, agriculture, urban planning)

**Deliverables:**
- [ ] Reveal.js presentation slides
- [ ] Visual diagrams for decision tree splitting
- [ ] Random Forest ensemble animation/diagram
- [ ] Philippine case study examples
- [ ] PDF export for offline use

#### 2. Jupyter Notebook - Theory Section (45 minutes)

**Topics to Cover:**
- Interactive explanations with visualizations
- Decision tree splitting examples using simple datasets
- Random Forest voting mechanism with animations
- Feature importance plots and interpretation
- Confusion matrix interpretation exercises
- Interactive quizzes/checkpoints

**Deliverables:**
- [ ] Interactive Jupyter notebook with markdown explanations
- [ ] Code cells demonstrating RF concepts with simple data
- [ ] Visualization functions for feature importance
- [ ] Exercise cells with TODO markers for students

#### 3. Jupyter Notebook - Hands-on Lab (1.5 hours)

**Workflow Structure:**

**A. Setup (10 minutes)**
- Import libraries (geemap, ee, scikit-learn, matplotlib)
- Authenticate Google Earth Engine
- Define Palawan study area geometry
- Set date range for imagery

**B. Data Acquisition (20 minutes)**
- Load Sentinel-2 imagery collection
- Apply cloud masking (QA60 band)
- Filter by date and cloud cover
- Compute spectral indices:
  - NDVI (Normalized Difference Vegetation Index)
  - NDWI (Normalized Difference Water Index)
  - NDBI (Normalized Difference Built-up Index)
  - EVI (Enhanced Vegetation Index)
- Create median composite image
- Visualize RGB and false color composites

**C. Training Data (20 minutes)**
- Load training polygons for land cover classes:
  - Forest
  - Agriculture
  - Water
  - Urban
  - Bare soil
- Visualize training areas on map
- Sample spectral values from training areas
- Prepare feature matrix (bands + indices)
- Inspect training data statistics

**D. Model Training (20 minutes)**
- Configure Random Forest classifier in GEE
- Set hyperparameters:
  - Number of trees (e.g., 100)
  - Variables per split
  - Min leaf population
- Train model on Sentinel-2 features
- Inspect feature importance

**E. Classification (15 minutes)**
- Apply trained model to study area
- Generate land cover classification map
- Create appropriate color palette for visualization
- Display results on interactive map
- Calculate area statistics by class

**F. Validation (25 minutes)**
- Split data into train/test sets (80/20)
- Apply model to validation data
- Calculate accuracy metrics:
  - Overall accuracy
  - Producer's accuracy (recall)
  - User's accuracy (precision)
  - Kappa coefficient
- Generate and visualize confusion matrix
- Analyze misclassification patterns
- Visualize feature importance ranking

**G. Exercises (20 minutes)**
- Modify number of trees and observe effects on accuracy
- Add/remove spectral indices and compare results
- Classify a different Philippine region
- Export classification results to Google Drive
- Generate accuracy report

**Deliverables:**
- [ ] Complete Jupyter notebook with all code sections
- [ ] Sample training data (GeoJSON files) for Palawan
- [ ] Helper functions for visualization
- [ ] Student version with exercises (TODO markers)
- [ ] Instructor solution notebook

#### 4. Supporting Materials

**To Create:**
- [ ] Quick reference guide for GEE Random Forest functions
- [ ] Land cover classification scheme for Philippines (LULC classes)
- [ ] Sample training data shapefiles/GeoJSON for Palawan
- [ ] Best practices checklist for classification
- [ ] Troubleshooting guide for common errors
- [ ] Code snippets library

---

## Session 2: Land Cover Classification Lab (Palawan)

**Duration:** 1.5-2 hours (Pure hands-on application)
**Learning Objective:** Apply advanced Random Forest techniques to real-world Philippine NRM scenarios

### Content Components to Create

#### 1. Jupyter Notebook - Extended Hands-on Lab

**A. Advanced Feature Engineering (30 minutes)**

**Topics:**
- Texture features using GLCM (Gray-Level Co-occurrence Matrix)
  - Contrast, correlation, entropy
- Temporal features
  - Seasonal composites (dry season, wet season)
  - Multi-temporal indices
  - Change metrics
- Topographic features
  - DEM-derived slope
  - Aspect
  - Elevation
- Radar features (optional)
  - Sentinel-1 backscatter

**Code Sections:**
- Load SRTM DEM for Palawan
- Calculate terrain metrics
- Compute GLCM texture features
- Create dry/wet season composites
- Stack all features into comprehensive feature set

**B. Palawan-Specific Case Study (45 minutes)**

**Study Area:** Palawan Biosphere Reserve
**Focus:** Protected area monitoring and deforestation detection

**Land Cover Classes:**
1. Primary forest
2. Secondary forest
3. Mangroves
4. Agricultural land (rice, coconut)
5. Grassland/scrubland
6. Water bodies
7. Urban/built-up
8. Bare soil/mining areas

**Workflow:**
- Load pre-prepared training datasets for Palawan
  - High-quality reference polygons
  - Balanced sample sizes
- Process multi-temporal Sentinel-2 data
  - Dry season composite (January-April)
  - Wet season composite (June-September)
- Calculate comprehensive feature stack (15-20 features)
- Train optimized Random Forest model
  - Hyperparameter tuning results applied
  - Cross-validation implemented
- Generate high-resolution land cover map
- Validate against independent reference data
- Calculate area statistics by land cover class
- Export results for GIS analysis

**C. Model Optimization (30 minutes)**

**Topics:**
- Hyperparameter tuning
  - Grid search for optimal parameters
  - Number of trees vs. accuracy
  - Max depth impact
  - Min samples split
- Cross-validation implementation
  - K-fold CV strategy
  - Spatial CV considerations
- Class balancing techniques
  - SMOTE for minority classes
  - Class weights adjustment
- Handling mixed pixels
  - Sub-pixel classification approaches
  - Fuzzy classification
- Edge detection and refinement
  - Post-processing filters
  - Boundary smoothing

**D. NRM Applications (15 minutes)**

**Practical Workflows:**

1. **Deforestation Detection**
   - Compare 2020 vs. 2024 classifications
   - Identify forest loss areas
   - Calculate deforestation rates
   - Map hotspots

2. **Change Detection Analysis**
   - Multi-temporal comparison
   - Transition matrices
   - Agricultural expansion patterns
   - Urban growth monitoring

3. **Protected Area Monitoring**
   - Classification within PA boundaries
   - Encroachment detection
   - Habitat quality assessment
   - Compliance reporting

4. **Agricultural Analysis**
   - Crop type mapping
   - Agricultural land expansion
   - Conversion patterns

5. **Stakeholder Reporting**
   - Generate summary statistics
   - Create maps for reports
   - Export data for decision-makers
   - Visualization for non-technical audiences

**Deliverables:**
- [ ] Extended hands-on Jupyter notebook
- [ ] Pre-processed Sentinel-2 datasets for Palawan
- [ ] Training/validation polygons (GeoJSON format)
- [ ] Reference land cover maps (GeoTIFF)
- [ ] Code templates for common NRM operations
- [ ] Hyperparameter tuning results documentation
- [ ] Troubleshooting guide for advanced features
- [ ] Sample output maps and reports

---

## Session 3: Introduction to Deep Learning and CNNs

**Duration:** 2-2.5 hours (Theory-focused with interactive examples)
**Learning Objective:** Understand deep learning fundamentals and CNN architecture for Earth Observation applications

### Content Components to Create

#### 1. Presentation Materials (60 minutes)

**Part A: From Traditional ML to Deep Learning (15 minutes)**

**Topics:**
- Recap of Random Forest approach from Session 1-2
- Limitations of Random Forest for image data
  - Manual feature engineering required
  - Limited spatial context
  - Performance ceiling
- Why CNNs for Earth Observation?
  - Automatic feature learning
  - Spatial hierarchy
  - State-of-the-art performance
- Success stories in EO applications
  - Scene classification benchmarks
  - Semantic segmentation results
  - Object detection examples
  - Philippine case studies

**Part B: Neural Network Fundamentals (20 minutes)**

**Topics:**
- Biological inspiration (brief)
- Artificial neurons
  - Inputs, weights, bias
  - Weighted sum
  - Activation functions (ReLU, Sigmoid, Tanh)
- Network architecture
  - Input layer, hidden layers, output layer
  - Fully connected networks
- Forward propagation
  - Step-by-step calculation example
- Loss functions
  - MSE for regression
  - Cross-entropy for classification
- Backpropagation intuition
  - Gradient descent
  - Chain rule (simplified)
- Optimizers
  - SGD, Momentum
  - Adam (recommended)
- Training concepts
  - Epochs, batches
  - Learning rate
  - Overfitting vs. underfitting
  - Regularization (dropout, L2)

**Visual Elements:**
- Interactive neural network visualizations
- Activation function plots
- Gradient descent animation
- Training curve examples

**Part C: Convolutional Neural Networks (25 minutes)**

**Topics:**
- **Convolutional Layers**
  - Convolution operation explained
  - Kernels/filters (3×3, 5×5)
  - Feature maps generation
  - Stride and padding
  - Receptive fields
  - Visualization of learned filters
  - Why convolution for images?

- **Pooling Layers**
  - Max pooling
  - Average pooling
  - Spatial dimension reduction
  - Translation invariance

- **CNN Architecture Components**
  - Typical structure: Conv → ReLU → Pool → Conv → ReLU → Pool → FC
  - Feature hierarchy (edges → textures → objects)
  - Early layers vs. deep layers

- **Spatial Hierarchies**
  - Low-level features (edges, corners)
  - Mid-level features (textures, patterns)
  - High-level features (semantic objects)
  - Visualization examples on Sentinel imagery

**Deliverables:**
- [ ] Reveal.js presentation slides (60 slides)
- [ ] Interactive neural network visualizations (TensorFlow Playground links)
- [ ] CNN animation/diagrams
- [ ] Filter visualization examples
- [ ] Architecture diagrams (VGG, ResNet)

#### 2. Jupyter Notebook - Interactive Theory (45 minutes)

**Part A: Hands-on Neural Network Basics (20 minutes)**

**Activities:**
- Build a simple perceptron from scratch (NumPy)
  - Define neuron class
  - Forward pass implementation
  - Sigmoid activation
- Visualize activation functions
  - Plot ReLU, Sigmoid, Tanh
  - Compare behaviors
- Build 2-layer neural network for tabular data
  - Use simple classification dataset
  - Train with gradient descent
  - Observe training dynamics
  - Plot loss curve
- Interactive exploration
  - Change learning rate
  - Observe convergence

**Part B: Convolution Operations (15 minutes)**

**Activities:**
- Load Sentinel-2 image patch
- Apply convolution filters manually
  - Edge detection (Sobel)
  - Blur (Gaussian)
  - Sharpening
- Visualize filter outputs
- Understand kernel operations
  - Sliding window
  - Element-wise multiplication
  - Summation
- Max pooling demonstrations
  - 2×2 pooling
  - Dimension reduction
  - Feature preservation

**Part C: Classic CNN Architectures (10 minutes)**

**Content:**
- **LeNet** (1998)
  - First successful CNN
  - Architecture walkthrough
  - MNIST application

- **VGG** (2014)
  - Simple, repeated blocks
  - 3×3 convolutions
  - Deep networks (16, 19 layers)

- **ResNet** (2015)
  - Skip connections
  - Residual learning
  - Very deep networks (50, 101, 152 layers)

- **Architecture Evolution Timeline**
  - Progress visualization
  - ImageNet performance improvements

**Deliverables:**
- [ ] Interactive theory Jupyter notebook
- [ ] NumPy neural network implementation
- [ ] Convolution visualization code
- [ ] Sample Sentinel-2 image patches
- [ ] Architecture comparison charts

#### 3. Presentation Materials (continued) (30 minutes)

**Part D: CNNs for Earth Observation Tasks (20 minutes)**

**Task Categories:**

1. **Scene Classification**
   - Classify entire image patch
   - Land use categories
   - EuroSAT benchmark
   - UC Merced dataset

2. **Semantic Segmentation**
   - Pixel-level classification
   - U-Net architecture
   - Land cover mapping
   - Building footprint extraction

3. **Object Detection**
   - Bounding boxes
   - YOLO, Faster R-CNN
   - Ship detection
   - Tree detection

4. **Change Detection**
   - Multi-temporal analysis
   - Siamese networks
   - Building damage assessment
   - Deforestation monitoring

5. **Super-Resolution**
   - Enhance spatial resolution
   - ESRGAN for EO
   - Sentinel-2 to 1m

6. **Cloud Masking**
   - Automated cloud detection
   - Shadow removal
   - Quality assessment

**Philippine EO Applications:**
- Typhoon damage assessment
- Rice paddy monitoring
- Illegal logging detection
- Coral reef mapping
- Urban expansion tracking

**Part E: Practical Considerations (10 minutes)**

**Topics:**
- **Data Preparation**
  - Image patches vs. full scenes
  - Patch size selection (64×64, 128×128, 256×256)
  - Overlapping patches
  - Label preparation

- **Data Augmentation**
  - Rotation, flipping
  - Brightness, contrast adjustments
  - Elastic deformations
  - Why augmentation matters (limited labeled data)

- **Transfer Learning**
  - Pre-trained models (ImageNet)
  - Fine-tuning strategies
  - Benefits for EO (limited labeled data)
  - Domain adaptation

- **Computational Requirements**
  - GPU vs. CPU
  - Training time estimates
  - Memory considerations
  - Cloud computing options (Colab, AWS, Azure)

- **Framework Comparison**
  - **TensorFlow/Keras**: Easy to learn, high-level API
  - **PyTorch**: Research-friendly, dynamic graphs
  - Choice depends on use case

**Deliverables:**
- [ ] Presentation slides (30 slides)
- [ ] Philippine EO application examples
- [ ] Framework comparison chart
- [ ] Computational requirements guide

#### 4. Interactive Quiz/Discussion (15 minutes)

**Quiz Questions:**
- What are the advantages of CNNs over Random Forest for image classification?
- Explain what a convolutional filter does.
- What is the purpose of pooling layers?
- When would you use transfer learning?
- What's the difference between scene classification and semantic segmentation?

**Scenario Exercises:**
- Design a CNN for classifying 10 land cover classes
- Choose appropriate architecture for building detection
- Estimate training time for given dataset

**Deliverables:**
- [ ] Quiz questions with answers
- [ ] Discussion prompts
- [ ] Scenario cards

---

## Session 4: CNN Hands-on Lab

**Duration:** 2.5-3 hours (Intensive hands-on coding)
**Learning Objective:** Build, train, and evaluate CNN models for Earth Observation image classification using TensorFlow/Keras and PyTorch

### Content Components to Create

#### 1. Jupyter Notebook - TensorFlow/Keras Track (Primary) (90 minutes)

**Part A: Environment Setup (10 minutes)**

**Setup Tasks:**
- Import TensorFlow, Keras
- Import EO libraries (rasterio, GDAL if needed)
- Import standard libraries (NumPy, matplotlib, scikit-learn)
- Check TensorFlow version
- Check GPU availability and configuration
  ```python
  print(tf.config.list_physical_devices('GPU'))
  ```
- Load helper functions (visualization, metrics)
- Set random seeds for reproducibility

**Dataset Download:**
- EuroSAT RGB dataset (~90MB)
- Automated download function
- Verify data integrity
- Extract to working directory

**Part B: Data Preparation (20 minutes)**

**EuroSAT Dataset:**
- 10 land use classes:
  1. Annual Crop
  2. Forest
  3. Herbaceous Vegetation
  4. Highway
  5. Industrial
  6. Pasture
  7. Permanent Crop
  8. Residential
  9. River
  10. Sea/Lake

**Data Loading:**
- Load dataset structure
- Explore directory organization
- Count samples per class
- Verify class balance

**Visualization:**
- Display sample images from each class
- Create class distribution bar chart
- Visualize RGB band statistics

**Data Splitting:**
- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified split to maintain class balance
- Create file path lists

**Preprocessing:**
- Normalize pixel values to [0, 1]
- Resize if needed (standardize to 64×64)
- Convert to TensorFlow tensors

**Data Augmentation Pipeline:**
- RandomFlip (horizontal, vertical)
- RandomRotation (±30°)
- RandomZoom (±10%)
- RandomContrast
- Demonstrate augmentation effects visually

**Data Generators:**
- Create `tf.data.Dataset` pipelines
- Set batch size (32 or 64)
- Shuffle training data
- Prefetch for performance
- Cache dataset in memory

**Part C: Build Basic CNN (25 minutes)**

**Architecture Design:**

```python
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2,2)),

    # Block 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    # Block 3
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    # Fully connected
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])
```

**Model Compilation:**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam(learning_rate=0.001)`
- Metrics: `['accuracy']`

**Model Summary:**
- Display architecture
- Count parameters
- Visualize with `plot_model()`

**Exercises:**
- Explain each layer's purpose
- Calculate output dimensions at each layer
- Estimate memory requirements

**Part D: Training and Evaluation (25 minutes)**

**Training Setup:**
- Set number of epochs (20-30)
- Configure callbacks:
  - EarlyStopping (monitor val_loss, patience=5)
  - ModelCheckpoint (save best model)
  - ReduceLROnPlateau (reduce LR on plateau)

**Training Execution:**
```python
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=[early_stopping, checkpoint]
)
```

**Training Visualization:**
- Plot training vs. validation loss
- Plot training vs. validation accuracy
- Identify overfitting/underfitting
- Analyze learning curves

**Test Set Evaluation:**
- Load best model checkpoint
- Evaluate on test set
- Calculate final accuracy

**Detailed Analysis:**
- Generate confusion matrix
  - Visualize as heatmap
  - Identify commonly confused classes
- Per-class accuracy (recall)
- Per-class precision
- F1-scores for each class
- Overall metrics summary

**Error Analysis:**
- Identify misclassified samples
- Visualize worst predictions
- Display predicted vs. true labels
- Analyze patterns in errors
  - Which classes are confused?
  - Why might the model fail?

**Part E: Model Experimentation (10 minutes)**

**Experiments to Try:**

1. **Architecture Modifications:**
   - Add/remove convolutional layers
   - Change number of filters
   - Try different kernel sizes
   - Modify dropout rate

2. **Hyperparameter Tuning:**
   - Learning rate (0.01, 0.001, 0.0001)
   - Batch size (16, 32, 64, 128)
   - Number of epochs

3. **Optimizer Comparison:**
   - SGD vs. Adam vs. RMSprop
   - Compare convergence speed

4. **Regularization:**
   - L2 regularization
   - Dropout variations
   - Batch normalization

**Exercise:**
- Students modify one aspect
- Train and compare results
- Document findings

**Deliverables:**
- [ ] Complete TensorFlow/Keras Jupyter notebook
- [ ] EuroSAT dataset download scripts
- [ ] Visualization helper functions
- [ ] Student version with exercises
- [ ] Instructor solution notebook
- [ ] Pre-trained model weights

#### 2. Jupyter Notebook - PyTorch Track (Alternative/Advanced) (90 minutes)

**Part A: PyTorch Fundamentals (15 minutes)**

**Core Concepts:**
- **Tensors:** PyTorch equivalent of NumPy arrays
  - Creation, operations
  - GPU transfer (.cuda(), .to(device))
- **Autograd:** Automatic differentiation
  - Requires_grad
  - Backward pass
- **DataLoader:** Efficient data loading
  - Batch loading
  - Shuffling
  - Multi-threading
- **Dataset Class:** Custom data handling
  - `__len__` method
  - `__getitem__` method
- **nn.Module:** Building models
  - Define `__init__` and `forward`
  - Layer composition

**Basic Examples:**
- Tensor operations
- Simple gradient computation
- Custom dataset implementation

**Part B: EuroSAT Classification with PyTorch (35 minutes)**

**Custom Dataset Class:**
```python
class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

**Data Transforms:**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**CNN Architecture:**
```python
class EuroSATCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EuroSATCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Training Loop:**
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc
```

**Validation Loop:**
```python
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc
```

**Full Training Process:**
- Initialize model, optimizer, loss function
- Train for multiple epochs
- Track train and validation metrics
- Save best model
- Plot learning curves
- Evaluate on test set

**Part C: Transfer Learning (20 minutes)**

**Pre-trained Models:**
- Load ResNet18 from torchvision.models
- Understand ImageNet pre-training
- Benefits for EO applications

**Model Modification:**
```python
import torchvision.models as models

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)
```

**Fine-tuning Strategies:**
1. **Feature Extraction:** Freeze all layers except final
2. **Fine-tuning:** Unfreeze later layers after initial training
3. **Full Training:** Train all layers with lower learning rate

**Comparison:**
- Train ResNet18 with transfer learning
- Compare with from-scratch CNN
- Observe faster convergence
- Better accuracy with less data

**Part D: TorchGeo Library (20 minutes)**

**Introduction to TorchGeo:**
- Specialized library for geospatial data
- Built on top of PyTorch
- Handles multi-spectral imagery
- Includes EO-specific datasets

**Installation:**
```bash
pip install torchgeo
```

**Working with EO Datasets:**
```python
from torchgeo.datasets import EuroSAT
from torchgeo.datamodules import EuroSATDataModule

# Easy dataset loading
dataset = EuroSAT(root='data', download=True)

# DataModule for training
datamodule = EuroSATDataModule(
    root_dir='data',
    batch_size=64,
    num_workers=4
)
```

**Multi-spectral Data Handling:**
- Load all 13 Sentinel-2 bands (not just RGB)
- Handle different band combinations
- Spectral normalization techniques
- Multi-spectral augmentations

**Sentinel-2 Workflow:**
```python
from torchgeo.datasets import Sentinel2

# Load Sentinel-2 dataset
sentinel = Sentinel2(
    root='path/to/data',
    bands=['B2', 'B3', 'B4', 'B8']  # Select specific bands
)
```

**Built-in Transformations:**
- Spectral indices calculation
- Cloud masking
- Temporal compositing
- Spatial augmentations

**Example Project:**
- Load Palawan Sentinel-2 tiles using TorchGeo
- Apply multi-spectral CNN
- Compare RGB vs. multi-spectral performance

**Deliverables:**
- [ ] Complete PyTorch Jupyter notebook
- [ ] Custom Dataset class examples
- [ ] Training loop implementations
- [ ] Transfer learning examples
- [ ] TorchGeo integration guide
- [ ] Student version with exercises
- [ ] Instructor solution notebook

#### 3. Jupyter Notebook - Semantic Segmentation Extension (Optional Advanced) (45 minutes)

**Target Audience:** Advanced students or as extension material

**Part A: Palawan Sentinel-2 Patches (15 minutes)**

**Dataset Preparation:**
- Pre-prepared Palawan image patches (256×256)
- Corresponding pixel-level land cover masks
- Multiple bands (RGB + NIR or full multi-spectral)
- 5-7 land cover classes

**Data Loading:**
```python
# Load image and mask pairs
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img = rasterio.open(os.path.join(image_dir, filename))
        mask = rasterio.open(os.path.join(mask_dir, filename))
        images.append(img.read())
        masks.append(mask.read())
    return np.array(images), np.array(masks)
```

**Visualization:**
- Display multi-band imagery
- Overlay land cover masks
- Show class distribution
- Visualize training samples

**Data Augmentation for Segmentation:**
- Synchronized transforms for image-mask pairs
- Rotation, flipping (apply to both)
- Elastic deformations
- Brightness/contrast (image only)

**Part B: U-Net Architecture (20 minutes)**

**U-Net Overview:**
- Encoder-decoder structure
- Skip connections between encoder/decoder
- Symmetric architecture
- Designed for biomedical image segmentation
- Adapted for EO applications

**Architecture Diagram:**
- Contracting path (encoder)
- Expanding path (decoder)
- Skip connections
- Bottleneck

**Implementation in TensorFlow/Keras:**
```python
def unet_model(input_shape=(256, 256, 4), num_classes=5):
    inputs = Input(input_shape)

    # Encoder (contracting path)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder (expanding path)
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])  # Skip connection
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(c7)

    return Model(inputs=[inputs], outputs=[outputs])
```

**Architecture Visualization:**
- Plot model structure
- Understand skip connections
- Calculate parameters

**Part C: Training Segmentation Model (10 minutes)**

**Loss Functions:**
- Binary cross-entropy (2 classes)
- Categorical cross-entropy (multi-class)
- Dice loss (handles class imbalance)
- Combined loss (BCE + Dice)

**Metrics:**
- IoU (Intersection over Union)
- Dice coefficient
- Per-class accuracy
- Pixel accuracy

**Training Process:**
```python
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', iou_metric, dice_coefficient]
)

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stopping, checkpoint]
)
```

**Visualization:**
- Display training curves
- Visualize predictions vs. ground truth
- Show segmentation masks
- Color-coded class predictions
- Overlay on original imagery

**Evaluation:**
- Calculate IoU per class
- Generate confusion matrix (pixel-level)
- Identify well-segmented vs. poorly segmented classes
- Analyze boundary accuracy

**Deliverables:**
- [ ] Semantic segmentation Jupyter notebook
- [ ] Pre-prepared Palawan image patches
- [ ] Corresponding land cover masks
- [ ] U-Net implementation (TensorFlow and PyTorch versions)
- [ ] Custom loss functions (Dice, IoU)
- [ ] Visualization utilities
- [ ] Student version with exercises
- [ ] Instructor solution notebook

#### 4. Supporting Materials

**To Create:**
- [ ] Pre-downloaded EuroSAT dataset (~90MB RGB version)
- [ ] Pre-prepared Palawan Sentinel-2 patches (256×256, multi-band)
- [ ] Land cover masks for Palawan patches
- [ ] CNN architecture templates (both frameworks)
- [ ] Common error solutions guide
  - CUDA out of memory
  - Dimension mismatch errors
  - NaN loss issues
  - Poor convergence
- [ ] Performance optimization tips
  - Batch size selection
  - Learning rate tuning
  - Data pipeline optimization
  - Mixed precision training
- [ ] Model export and deployment guide
  - SavedModel format (TensorFlow)
  - TorchScript (PyTorch)
  - ONNX export
  - Inference optimization
- [ ] Comparison chart: TensorFlow vs. PyTorch
- [ ] GPU setup guide (local and cloud)
- [ ] Colab notebook versions (free GPU access)

---

## Overall Learning Objectives

### By the End of DAY 2, Participants Will Be Able To:

#### 1. Machine Learning Foundations
- Explain the supervised classification workflow for Earth Observation data
- Implement Random Forest classification using Google Earth Engine
- Perform feature engineering with spectral indices and auxiliary data
- Conduct accuracy assessment using confusion matrices and statistics
- Interpret feature importance for model understanding
- Apply hyperparameter tuning for optimal performance

#### 2. Deep Learning Fundamentals
- Understand neural network basics including forward/backward propagation
- Explain convolutional operations and their benefits for image data
- Describe CNN architecture components (Conv, Pool, FC layers)
- Identify appropriate CNN architectures for different EO tasks
- Distinguish between scene classification and semantic segmentation
- Understand the role of activation functions, loss functions, and optimizers

#### 3. Practical Implementation Skills
- Build, compile, and train CNNs using TensorFlow/Keras
- Implement CNN models in PyTorch for research flexibility
- Apply transfer learning to Earth Observation classification problems
- Implement data augmentation strategies for limited labeled datasets
- Evaluate model performance using appropriate metrics
- Debug common training issues (overfitting, poor convergence)
- Deploy trained models to classify Philippine landscapes

#### 4. Philippine NRM Context
- Apply classification techniques to Palawan land cover mapping
- Monitor forest cover change and agricultural expansion
- Support decision-making for protected area management
- Generate area statistics and change detection reports
- Create actionable insights for environmental stakeholders
- Understand specific challenges in Philippine EO applications

#### 5. Advanced Concepts (Optional)
- Implement semantic segmentation with U-Net architecture
- Work with multi-spectral imagery beyond RGB
- Use specialized EO libraries (TorchGeo)
- Design custom CNN architectures for specific tasks

---

## Cross-Session Integration

### Learning Progression:

```
Session 1: RF Theory
    ↓ (Apply concepts)
Session 2: RF Practice on Palawan
    ↓ (Recognize limitations)
Session 3: CNN Theory
    ↓ (Implement solutions)
Session 4: CNN Practice on EuroSAT/Palawan
```

### Consistent Elements Across Sessions:

1. **Palawan Case Study:**
   - Session 1: Introduce Palawan study area
   - Session 2: Deep dive into Palawan classification
   - Session 4: Palawan segmentation (optional)

2. **Methodology Comparison:**
   - Explicitly compare RF vs. CNN performance
   - Discuss when to use each approach
   - Trade-offs: interpretability vs. accuracy

3. **Philippine Context:**
   - All examples use Philippine locations
   - Reference local agencies (PhilSA, NAMRIA, DENR)
   - Discuss real-world applications for Philippine NRM

4. **Hands-on Focus:**
   - Every session includes coding
   - Progressive complexity
   - Build on previous knowledge

5. **Assessment Strategy:**
   - Session 1-2: Focus on accuracy metrics
   - Session 3: Conceptual quizzes
   - Session 4: Model performance evaluation

---

## Implementation Timeline

### Estimated Development Time: 8 weeks

| Week | Tasks | Deliverables | Status |
|------|-------|-------------|--------|
| **Week 1** | Session 1 presentation materials | Slides (45 min), diagrams | ⬜ Not Started |
| **Week 2** | Session 1 Jupyter notebooks | Theory notebook, hands-on lab | ⬜ Not Started |
| | Session 1 supporting materials | Training data, quick reference | ⬜ Not Started |
| **Week 3** | Session 2 Jupyter notebook | Extended lab with Palawan case | ⬜ Not Started |
| | Session 2 datasets | Pre-processed imagery, polygons | ⬜ Not Started |
| **Week 4** | Session 3 presentation materials | Slides (90 min), visualizations | ⬜ Not Started |
| **Week 5** | Session 3 interactive notebook | Neural network demos, convolution | ⬜ Not Started |
| | Session 3 quiz materials | Questions, scenarios | ⬜ Not Started |
| **Week 6** | Session 4 TensorFlow notebook | EuroSAT classification lab | ⬜ Not Started |
| | Session 4 datasets | EuroSAT download, Palawan patches | ⬜ Not Started |
| **Week 7** | Session 4 PyTorch notebook | PyTorch track, TorchGeo examples | ⬜ Not Started |
| | Session 4 segmentation notebook | U-Net implementation (optional) | ⬜ Not Started |
| **Week 8** | Integration and testing | Test all notebooks, fix bugs | ⬜ Not Started |
| | Documentation | README, setup guides, troubleshooting | ⬜ Not Started |
| | Final review | Ensure consistency, quality check | ⬜ Not Started |

### Detailed Task Breakdown:

#### Session 1 Development (Weeks 1-2):
- [ ] Create presentation slides on Random Forest theory
- [ ] Design decision tree visualizations
- [ ] Build interactive RF voting mechanism demo
- [ ] Develop theory Jupyter notebook
- [ ] Create hands-on lab notebook with GEE
- [ ] Prepare Palawan training data (GeoJSON)
- [ ] Write quick reference guide
- [ ] Create student and instructor versions

#### Session 2 Development (Week 3):
- [ ] Develop extended hands-on notebook
- [ ] Prepare multi-temporal Sentinel-2 composites
- [ ] Create training/validation polygons
- [ ] Generate reference land cover maps
- [ ] Write hyperparameter tuning section
- [ ] Develop NRM application examples
- [ ] Create code templates
- [ ] Write troubleshooting guide

#### Session 3 Development (Weeks 4-5):
- [ ] Create deep learning fundamentals slides
- [ ] Design CNN architecture diagrams
- [ ] Build neural network animations
- [ ] Develop interactive theory notebook
- [ ] Create convolution visualization demos
- [ ] Write architecture comparison section
- [ ] Prepare EO applications slides
- [ ] Develop quiz questions and scenarios
- [ ] Create discussion prompts

#### Session 4 Development (Weeks 6-7):
- [ ] Develop TensorFlow/Keras notebook
- [ ] Download and prepare EuroSAT dataset
- [ ] Create data loading and preprocessing code
- [ ] Implement CNN architecture from scratch
- [ ] Write training and evaluation sections
- [ ] Develop PyTorch notebook
- [ ] Create transfer learning examples
- [ ] Write TorchGeo integration guide
- [ ] Develop U-Net segmentation notebook
- [ ] Prepare Palawan segmentation patches
- [ ] Create visualization utilities
- [ ] Write deployment guide
- [ ] Create Colab versions

#### Integration and Testing (Week 8):
- [ ] Test all notebooks end-to-end
- [ ] Verify datasets are accessible
- [ ] Check code compatibility (TF, PyTorch versions)
- [ ] Fix bugs and errors
- [ ] Ensure consistent styling
- [ ] Verify learning progression
- [ ] Test on different environments (local, Colab)
- [ ] Create master README
- [ ] Write setup instructions
- [ ] Finalize troubleshooting guide
- [ ] Peer review all materials

---

## Technical Requirements

### Software Dependencies:

#### Core Libraries:
```
Python 3.8+
TensorFlow 2.10+
Keras (bundled with TensorFlow)
PyTorch 1.12+
torchvision
```

#### Earth Observation Libraries:
```
earthengine-api
geemap
rasterio
GDAL
geopandas
```

#### Data Science Stack:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
```

#### Specialized EO:
```
torchgeo
sentinelsat (optional)
```

#### Utilities:
```
jupyter
jupyterlab
tqdm
pillow
```

### Hardware Recommendations:

#### Minimum (CPU-only):
- **Processor:** Dual-core, 2.5 GHz
- **RAM:** 8 GB
- **Storage:** 20 GB free space
- **Internet:** Stable connection for GEE
- **Note:** Sessions 1-2 work fine; Session 4 will be slow

#### Recommended (GPU):
- **Processor:** Quad-core, 3.0 GHz
- **RAM:** 16 GB
- **GPU:** NVIDIA GPU with 4+ GB VRAM
  - GTX 1050 Ti or better
  - RTX 2060 or better (optimal)
- **Storage:** 50 GB free space (SSD preferred)
- **Internet:** High-speed for dataset downloads

#### Cloud Alternative:
- **Google Colab** (Free tier):
  - Free GPU access (limited hours)
  - 12 GB RAM
  - No setup required
  - Ideal for Session 4
- **Google Colab Pro** ($10/month):
  - More GPU hours
  - Better GPUs
  - 25 GB RAM
  - Background execution

### Platform Compatibility:

| Platform | Session 1-2 (GEE) | Session 3 (Theory) | Session 4 (CNN) |
|----------|-------------------|-------------------|-----------------|
| Windows  | ✅ Full support   | ✅ Full support   | ✅ GPU with CUDA |
| macOS    | ✅ Full support   | ✅ Full support   | ⚠️ CPU only (M1/M2 experimental) |
| Linux    | ✅ Full support   | ✅ Full support   | ✅ GPU with CUDA |
| Colab    | ✅ Recommended    | ✅ Works well     | ✅ Free GPU |

### Data Storage Requirements:

| Dataset | Size | Purpose | Sessions |
|---------|------|---------|----------|
| Palawan Training Polygons | 5 MB | RF training data | 1, 2 |
| Palawan Sentinel-2 Composites | 500 MB | Classification | 1, 2 |
| EuroSAT RGB | 90 MB | CNN classification | 4 |
| EuroSAT Full (13 bands) | 2 GB | Advanced CNN | 4 (optional) |
| Palawan Segmentation Patches | 200 MB | U-Net training | 4 (optional) |
| Pre-trained Models | 100 MB | Transfer learning | 4 |
| **Total (Minimum)** | **~1 GB** | | |
| **Total (Full)** | **~3 GB** | | |

### Google Earth Engine Setup:

1. **Account Registration:**
   - Sign up at https://earthengine.google.com/
   - Use organizational or personal Google account
   - Approval typically takes 1-2 days

2. **Authentication:**
   ```python
   import ee
   ee.Authenticate()  # First time only
   ee.Initialize()
   ```

3. **API Access:**
   - Cloud project required
   - Free tier sufficient for course

### GPU Setup (Local):

#### NVIDIA GPU (Windows/Linux):
1. Install NVIDIA drivers
2. Install CUDA Toolkit 11.2+
3. Install cuDNN 8.1+
4. Install TensorFlow/PyTorch GPU versions
5. Verify GPU detection

#### Apple Silicon (M1/M2):
1. Use TensorFlow-metal plugin (experimental)
2. Limited PyTorch support
3. Recommend Colab for Session 4

### Environment Management:

**Recommended: Conda Environment**
```bash
# Create environment
conda create -n copphil_day2 python=3.9

# Activate environment
conda activate copphil_day2

# Install dependencies
pip install -r requirements.txt
```

**Alternative: Virtual Environment**
```bash
# Create venv
python -m venv copphil_env

# Activate
source copphil_env/bin/activate  # Linux/Mac
copphil_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Additional Notes

### Pedagogical Considerations:

1. **Mixed Proficiency Levels:**
   - Provide basic and advanced tracks in Session 4
   - Optional advanced topics (U-Net, TorchGeo)
   - Self-paced exercises

2. **Hands-on Focus:**
   - Minimum 60% coding time
   - Theory always followed by practice
   - Real-world datasets

3. **Assessment Strategy:**
   - Embedded quizzes in notebooks
   - Practical exercises with solutions
   - Capstone project suggestion

4. **Filipino Context:**
   - All examples use Philippine locations
   - Reference local agencies
   - Discuss relevant applications

### Common Challenges to Address:

1. **Computational Limitations:**
   - Provide Colab alternatives
   - Optimize code for CPU
   - Pre-process large datasets

2. **Internet Connectivity:**
   - Provide offline dataset downloads
   - Cache GEE imagery
   - Downloadable presentations

3. **Prior Knowledge Gaps:**
   - Quick Python refresher in Session 1
   - NumPy/Pandas reminders
   - Link to DAY 1 materials

4. **Debugging Support:**
   - Common error solutions
   - Troubleshooting checklist
   - Office hours recommendations

### Quality Assurance:

- [ ] All code tested on Windows, Mac, Linux
- [ ] All notebooks run end-to-end without errors
- [ ] Datasets accessible and downloadable
- [ ] Visualizations render correctly
- [ ] Timing estimates verified
- [ ] Peer review completed
- [ ] Student pilot testing

### Future Enhancements:

- Video walkthroughs for each session
- Additional Philippine case studies
- Advanced topics (GANs, transformers for EO)
- Model deployment tutorials
- Integration with GEE Code Editor
- Mobile-friendly materials

---

## Contact and Support

**Course Development Team:**
- TBD

**Technical Support:**
- GitHub Issues: [Repository link]
- Discussion Forum: [Link]
- Office Hours: [Schedule]

**Resources:**
- Course Website: [URL]
- Code Repository: [GitHub URL]
- Dataset Repository: [Cloud storage URL]
- Documentation: [Docs URL]

---

**Document Status:** ✅ Complete and Ready for Implementation

**Next Action:** Begin Week 1 development tasks (Session 1 presentation materials)
