---
trigger: always_on
---

You will follow the following stracture for the course for DAY 2: "Day 2: Machine Learning for Image Classification & Introduction to Deep Learning

Session 1: Supervised Classification with Random Forest for EO (1.5 hours)
○	Module: Theory of Decision Trees and Random Forest (RF) algorithm: This module will explain the principles behind decision trees and how the Random Forest (RF) algorithm builds an ensemble of such trees to improve classification accuracy and robustness. Key advantages in EO, such as its ability to handle high-dimensional data (many spectral bands or features) and its non-parametric nature (making no assumptions about data distribution), will be discussed. Its sensitivity to the quality and design of training samples will also be highlighted as a critical consideration.
○	Module: Feature selection and importance in RF: RF provides measures of variable importance (e.g., Gini importance), which can be used to understand which input features (e.g., spectral bands, vegetation indices) are most influential in the classification process and for feature selection to potentially improve model efficiency and accuracy.
○	Module: Training data preparation: Best practices for preparing training data for supervised classification will be covered, including strategies for sampling (e.g., stratified random sampling), defining clear and representative land cover classes, and ensuring an adequate number of high-quality samples.
○	Module: Model training, prediction, and accuracy assessment: The process of training an RF classifier, using it to predict classes for an entire image, and evaluating its performance using standard metrics like the confusion matrix, overall accuracy, producer's and user's accuracy, and the Kappa coefficient will be detailed.

Session 2: Hands-on: Land Cover Classification (NRM Focus) using Sentinel-2 & Random Forest (2.5 hours)
○	Case Study: Land Cover Classification in Palawan. This case study directly aligns with the Natural Resource Management (NRM) thematic area specified in the TOR. Palawan, with its rich biodiversity and pressures from development, serves as an excellent example for NRM applications.
○	Platform: Google Earth Engine (for data preparation and RF classification) and/or Python with Scikit-learn in Google Colab (for a more detailed local analysis or comparison).
○	Data: Sentinel-2 multispectral imagery for a selected area in Palawan. Optionally, Shuttle Radar Topography Mission (SRTM) DEM data can be incorporated as an additional feature to aid classification, as terrain can influence land cover types.
○	Workflow:
1.	Define an Area of Interest (AOI) in Palawan.
2.	Access and pre-process Sentinel-2 imagery within GEE (e.g., cloud masking, creating a median composite for a specific period).
3.	Collect training samples: Participants will learn to digitize polygons representing different land cover classes (e.g., forest, mangrove, agriculture, urban, water) directly in GEE, or use pre-existing shapefiles if available.
4.	Extract predictor variables for training points: This includes spectral band values from Sentinel-2 and derived indices like NDVI. If DEM is used, elevation and slope can also be added as features.
5.	Train a Random Forest classifier in GEE using ee.Classifier.smileRandomForest() or a similar function.
6.	Classify the Sentinel-2 image composite for the AOI.
7.	Perform accuracy assessment using a reserved set of validation samples.
8.	(Optional) Export the classified map and discuss how QGIS can be used for further cartographic refinement, map layout, and post-processing tasks.
○	Practical Tips/Pitfalls to Highlight: The importance of creating high-quality, representative training samples cannot be overstated. Issues such as mixed pixels (pixels containing multiple land cover types), the impact of atmospheric conditions (even with L2A data), and the careful selection of input features (bands and indices) will be discussed as common challenges.

Session 3: Introduction to Deep Learning: Neural Networks & CNNs (1.5 hours)
○	Module: Recap Neural Networks. Introduction to Deep Learning concepts: Building upon basic neural network ideas, this module will formally introduce Deep Learning as a subfield of ML characterized by networks with multiple layers (deep architectures).
○	Module: Convolutional Neural Networks (CNNs): The core architecture of CNNs will be explained, including the role and function of convolutional layers (applying filters to extract features like edges, textures), pooling layers (reducing dimensionality and creating invariance to small translations), and fully connected layers (for final classification or regression). The concept of learnable filters and hierarchical feature extraction (from simple to complex features) will be central. Common activation functions like ReLU will be introduced.
○	Module: How CNNs learn features for image analysis. Applications in EO: The power of CNNs in automatically learning relevant features from raw pixel data will be emphasized, contrasting with traditional methods that require manual feature engineering. Diverse applications in EO, such as image classification (assigning a single label to an image patch), object detection (locating objects), and semantic segmentation (pixel-wise classification), will be showcased.
○	Module: Introduction to TensorFlow and PyTorch for building CNNs: A brief overview of these two leading deep learning frameworks, highlighting their main components for defining and training CNN models, will be provided. The transition from Random Forest, a more intuitively understandable ensemble method, to CNNs allows participants to appreciate the significant leap in automated feature extraction offered by deep learning. The hands-on RF case study provides a concrete example of a successful EO application before delving into the more abstract workings of CNNs.

Session 4: Hands-on: Basic CNN for Image Classification with TensorFlow/Keras or PyTorch (2.5 hours)
○	Platform: Google Colab with GPU acceleration enabled.
○	Dataset: A small, pre-prepared EO image patch dataset will be used. Options include a subset of the EuroSAT dataset (derived from Sentinel-2, as used in several tutorials) or image patches extracted from the Palawan LULC case study data. The dataset will consist of image patches labeled with specific classes (e.g., 'forest', 'water', 'urban').
○	Workflow:
1.	Load and pre-process image patches: This includes normalizing pixel values (e.g., to a 0-1 range) and resizing patches to a uniform input size for the CNN.
2.	Define a simple CNN architecture: Participants will build a basic CNN, for example, with 2-3 convolutional layers, interspersed with pooling layers, and followed by one or more dense (fully connected) layers for classification.
3.	Compile the model: This involves specifying a loss function suitable for multi-class classification (e.g., categorical cross-entropy) and an optimizer (e.g., Adam).
4.	Train the model: The CNN will be trained on the training set of image patches.
5.	Evaluate the model: Performance will be assessed on a separate validation or test set of patches using metrics like accuracy.
6.	Visualize some predictions: A few example predictions on test images will be shown to qualitatively assess performance.
○	Focus: The primary goal of this session is to provide a foundational understanding of the code structure and workflow for building, training, and evaluating a CNN. The emphasis is on the process rather than achieving state-of-the-art accuracy. Reference tutorials like those found for EuroSAT classification can serve as a basis.
○	Tooling Choice Consideration: The choice of TensorFlow or PyTorch for this initial CNN exercise will influence subsequent deep learning sessions. While PyTorch's "Pythonic" nature is often favored in research for its ease of experimentation, TensorFlow with the high-level Keras API is also very accessible for beginners. Maintaining consistency with one framework for the more advanced U-Net and LSTM models later in the training could reduce the cognitive load on participants. Providing well-commented Colab notebooks for this exercise is a direct contribution to the materials for the Digital Space Campus 1, facilitating self-paced learning."