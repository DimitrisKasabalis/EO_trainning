# Session 3: Introduction to Deep Learning and CNNs for Earth Observation

## Overview

**Duration:** 2.5 hours
**Type:** Theory + Interactive Demonstrations
**Difficulty:** Intermediate
**Prerequisites:** Sessions 1-2 complete (Random Forest classification)

This session bridges traditional machine learning and deep learning, introducing neural networks and Convolutional Neural Networks (CNNs) specifically for Earth Observation applications. Students transition from manual feature engineering (NDVI, GLCM) to automatic feature learning through deep neural networks.

## Learning Objectives

After completing this session, participants will be able to:

1. **Understand the ML → DL Transition**
   - Recognize when to use traditional ML vs deep learning
   - Explain the automatic feature learning paradigm
   - Identify computational and data requirements

2. **Master Neural Network Fundamentals**
   - Build simple perceptrons from scratch using NumPy
   - Implement and visualize activation functions (ReLU, sigmoid, softmax)
   - Understand forward and backward propagation conceptually

3. **Comprehend Convolutional Neural Networks**
   - Explain convolution operations on satellite imagery
   - Understand pooling, padding, and stride concepts
   - Compare CNN architectures (LeNet, VGG, ResNet, U-Net)

4. **Apply CNNs to EO Tasks**
   - Identify appropriate architectures for classification vs segmentation
   - Recognize object detection and change detection approaches
   - Connect CNN capabilities to Philippine EO applications

5. **Navigate Practical Considerations**
   - Address data-centric AI principles for EO
   - Understand transfer learning and data augmentation
   - Recognize computational constraints and optimization strategies

## Session Structure

### Part A: From Machine Learning to Deep Learning (15 min)
- Comparison: Random Forest vs CNNs
- When to use which approach
- Philippine EO context (PhilSA, DENR applications)

### Part B: Neural Network Fundamentals (25 min)
- The perceptron: building block of neural networks
- Activation functions (ReLU, sigmoid, tanh, softmax)
- Multi-layer networks and hierarchical learning
- Training process: gradient descent and backpropagation

### Part C: Convolutional Neural Networks (30 min)
- Why CNNs for images (parameter sharing, local connectivity)
- Convolution operations (filters, feature maps)
- CNN building blocks (Conv, Pool, FC, Dropout)
- Classic architectures (LeNet, VGG, ResNet, U-Net)

### Part D: CNNs for Earth Observation (25 min)
- Scene classification (land cover, cloud detection)
- Semantic segmentation (flood mapping, building extraction)
- Object detection (ships, buildings, trees)
- Change detection (deforestation, urban expansion)

### Part E: Practical Considerations (15 min)
- Data requirements and sources (PhilSA, NAMRIA, ASTI)
- Transfer learning strategies
- Data augmentation techniques
- Computational requirements and cloud options
- Model interpretability (CAM, saliency maps)

## Materials

### Jupyter Notebooks

1. **session3_theory_STUDENT.ipynb** (~50 cells, 45 min)
   - Part 1: Build perceptron from scratch
   - Part 2: Activation function visualizations
   - Part 3: Train 2-layer network on spectral data
   - Part 4: Learning rate exploration

2. **session3_cnn_operations_STUDENT.ipynb** (~45 cells, 55 min)
   - Part 1: Manual convolution on Sentinel-2
   - Part 2: Edge detection filters (Sobel, Gaussian)
   - Part 3: Max pooling demonstration
   - Part 4: Architecture comparison (LeNet, VGG, ResNet, U-Net)

### Documentation

1. **CNN_ARCHITECTURES.md**
   - Detailed explanations of classic CNN architectures
   - Parameter counts and computational requirements
   - EO application suitability for each architecture

2. **EO_APPLICATIONS.md**
   - Comprehensive guide to CNN applications in EO
   - Philippine-specific use cases with stakeholders
   - Data requirements and annotation strategies

## Prerequisites

### Knowledge
- Complete Sessions 1-2 (Random Forest classification)
- Understanding of classification metrics (accuracy, confusion matrix)
- Basic Python and NumPy familiarity
- Conceptual understanding of matrix operations

### Technical Setup
- Google Colab account (free tier sufficient for Session 3)
- GPU runtime enabled (Runtime → Change runtime type → GPU)
- Python libraries: NumPy, Matplotlib, SciPy (pre-installed in Colab)

## Key Concepts

### Automatic Feature Learning
CNNs learn optimal features directly from raw pixel data, eliminating manual feature engineering. Early layers learn edges, middle layers learn textures, and deep layers learn semantic patterns.

### Receptive Field
The region of the input image that influences a particular neuron's activation. Deeper networks have larger receptive fields, capturing more spatial context.

### Translation Invariance
CNNs can recognize patterns regardless of position in the image through parameter sharing and pooling operations.

### Gradient Descent
Optimization algorithm that iteratively adjusts weights to minimize prediction error by moving "downhill" on the error surface.

## Philippine EO Applications

### PhilSA Space+ Dashboard
- Automated cloud masking (U-Net, 95% accuracy)
- National land cover classification (ResNet-50, 10 classes)
- Disaster rapid mapping (Sentinel-1 + U-Net, 6-hour response)

### DENR Forest Monitoring
- Protected area surveillance (monthly Sentinel-2 analysis)
- Illegal logging detection (change detection CNN)
- Biodiversity hotspot mapping (fine-grained classification)

### LGU Applications
- Land use planning with ASTI SkAI-Pinas models
- Quarterly monitoring using fine-tuned CNNs
- Integration with local GIS systems

## Expected Outcomes

### Conceptual Understanding
- Explain why CNNs excel at spatial pattern recognition
- Sketch CNN architecture and label components
- Describe forward and backward propagation
- Identify appropriate architectures for different EO tasks

### Technical Skills
- Build perceptron from scratch in NumPy
- Implement activation functions
- Perform manual convolution on satellite imagery
- Visualize feature maps and pooling operations

### Practical Readiness
- Understand TensorFlow/Keras workflow for Session 4
- Anticipate data preparation challenges
- Recognize computational requirements
- Set realistic expectations for training time

## Assessment

### Formative (During Session)
- Self-check questions on key concepts
- Complete all TODO cells in notebooks
- Implement manual convolution from scratch
- Compare filter responses on different land cover types

### Summative (End of Session)
- 10-question knowledge check (multiple choice)
- Sketch appropriate architecture for given EO task
- Estimate data requirements for Philippine use case
- Demonstrate readiness for Session 4 hands-on implementation

## Resources

### Interactive Tools
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Visualize CNNs in browser
- [TensorFlow Playground](https://playground.tensorflow.org/) - Neural network sandbox
- [Distill.pub CNNs](https://distill.pub/2017/feature-visualization/) - Feature visualization

### Learning Materials
- [3Blue1Brown Neural Network Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Visual explanations
- [Stanford CS231n](http://cs231n.stanford.edu/) - Comprehensive CNN course
- [Andrew Ng Deep Learning](https://www.coursera.org/specializations/deep-learning) - Coursera specialization

### EO-Specific
- [EuroSAT Dataset](https://github.com/phelber/EuroSAT) - Sentinel-2 scene classification benchmark
- [TorchGeo](https://torchgeo.readthedocs.io/) - PyTorch for geospatial data
- [Awesome Satellite Imagery](https://github.com/robmarkcole/satellite-image-deep-learning) - Curated resources

## Troubleshooting

### Common Issues
- **Out of memory in Colab:** Use smaller image chips (128×128 instead of 256×256)
- **Slow manual convolutions:** Normal for NumPy implementation; Session 4 uses GPU-optimized libraries
- **Visualizations not displaying:** Add `%matplotlib inline` at notebook start
- **GPU not available:** Runtime → Change runtime type → GPU (free tier provides T4)

### Getting Help
- Instructor support during lab hours
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [CS231n Course Notes](http://cs231n.github.io/)
- PhilSA Data Support: data@philsa.gov.ph

## Next Steps

### Session 4 Preview
Hands-on implementation of CNNs using TensorFlow/Keras:
- Build and train ResNet classifier for Palawan land cover (8 classes)
- Implement U-Net for flood mapping in Central Luzon
- Apply transfer learning and data augmentation
- Export trained models for operational use

### Recommended Pre-Work
1. Review notebook exercises from Session 3
2. Read U-Net paper ([Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597))
3. Enable GPU in Colab and test TensorFlow installation
4. Familiarize with TensorFlow/Keras API basics

## License and Citation

**Course Materials:** CopPhil Advanced Training Program
**Developed by:** EO Training Team with Philippine EO community input
**Last Updated:** October 2025

**Citation:**
```
CopPhil Advanced Training Program (2025). Session 3: Introduction to Deep Learning
and CNNs for Earth Observation. Day 2, AI/ML for Philippine EO Professionals.
```

## Contact

**Training Coordinators:** training@copphil.org
**Technical Support:** support@copphil.org
**PhilSA Collaboration:** data@philsa.gov.ph
