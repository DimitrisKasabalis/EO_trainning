---
trigger: always_on
---

You will follow the following stracture for the course for DAY 3: "Day 3: Advanced Deep Learning: Semantic Segmentation & Object Detection

Session 1: Semantic Segmentation with U-Net for EO (1.5 hours)
○	Module: Concept of semantic segmentation: This module will introduce semantic segmentation as a pixel-wise classification task, where each pixel in an image is assigned a class label (e.g., water, building, forest). This contrasts with image classification (one label per image) and object detection (bounding boxes around objects).
○	Module: U-Net Architecture: The U-Net architecture, highly popular for semantic segmentation in EO and medical imaging, will be detailed. This includes its characteristic encoder (contracting path) for feature extraction and context aggregation, decoder (expansive path) for precise localization and upsampling, and the crucial skip connections that fuse high-resolution features from the encoder with upsampled features in the decoder to preserve spatial detail. The role of the bottleneck layer will also be explained.
○	Module: Applications in EO: Various applications of U-Net in Earth Observation will be showcased, such as flood mapping from SAR/optical data, detailed land cover mapping, road network extraction, and building footprint delineation.
○	Module: Loss functions for segmentation: Common loss functions used for training segmentation models, such as pixel-wise cross-entropy, Dice loss, and Jaccard index (Intersection over Union - IoU), will be introduced, highlighting their suitability for handling class imbalance and focusing on boundary accuracy.

Session 2: Hands-on: Flood Mapping (DRR Focus) using Sentinel-1 SAR & U-Net (2.5 hours)
○	Case Study: Flood Mapping in Central Luzon (Pampanga River Basin), focusing on a specific past typhoon event like Ulysses (2020) or Karding (2022). 
○	Platform: Google Colab with GPU acceleration. The deep learning framework (PyTorch or TensorFlow) will be consistent with that used on Day 2.
○	Data: Pre-processed Sentinel-1 SAR image patches (e.g., 128x128 or 256x256 pixels) containing both VV and VH polarizations will be provided. Corresponding binary flood masks (1 for flood, 0 for non-flood) for these patches, derived from a chosen typhoon event, will also be supplied.
■	Data Preparation Pitfall to Highlight: The significant effort required for preparing analysis-ready SAR data (speckle filtering, radiometric calibration, geometric terrain correction) and generating accurate ground truth flood masks will be emphasized. For this hands-on session, providing pre-processed patches is essential to allow participants to focus on understanding and implementing the U-Net model itself.
○	Workflow:
1.	Load SAR image patches and their corresponding binary flood masks.
2.	Implement data augmentation techniques suitable for SAR data if time permits and deemed beneficial (e.g., rotation, flip).
3.	Define the U-Net model architecture using the chosen framework.
4.	Compile the model, selecting an appropriate loss function (e.g., Dice loss or a combination) and optimizer.
5.	Train the U-Net model on the training set of patches, monitoring performance on a validation set.
6.	Evaluate the trained model's performance on a test set using metrics like Intersection over Union (IoU), F1-score, precision, and recall.
7.	Visualize some of the model's predictions on test patches, overlaying the predicted flood extent on the SAR imagery.
○	Conceptual Hurdle for EO Users: A key challenge for participants might be grasping how the U-Net architecture, particularly its convolutional layers, processes SAR data (which is inherently different from optical data in terms of image characteristics) and how the skip connections contribute to the precise delineation of flood boundaries. Explaining the flow of information and the role of multi-resolution feature fusion will be important. The use of a highly relevant DRR case study like flood mapping in a major Philippine river basin, employing advanced deep learning, makes this session particularly impactful. The complexity is managed by providing pre-processed data, allowing focus on the AI/ML aspects.

Session 3: Object Detection Techniques for EO Imagery (1.5 hours)
○	Module: Concept of object detection: This module will clearly define object detection as the task of not only classifying objects within an image but also localizing them, typically by drawing bounding boxes around each detected instance.
○	Module: Overview of popular architectures: A high-level overview of common object detection architectures will be provided:
■	Two-stage detectors: These first propose regions of interest and then classify those regions (e.g., R-CNN, Fast R-CNN, Faster R-CNN).
■	Single-stage detectors: These perform localization and classification in a single pass, often being faster (e.g., YOLO - You Only Look Once, SSD - Single Shot MultiBox Detector).
■	Transformer-based detectors (e.g., DETR): Briefly introduced as an emerging and powerful approach, if appropriate for an advanced audience.
○	Module: Applications in EO: Numerous applications of object detection in Earth Observation will be discussed, such as detecting and counting ships, vehicles, aircraft, buildings, oil tanks, and other infrastructure elements from satellite or aerial imagery.
○	Module: Challenges in EO object detection: Specific challenges pertinent to EO will be highlighted, including the detection of small objects relative to image size, variations in object scale and orientation, complex backgrounds (e.g., urban clutter), atmospheric effects, and often, the limited availability of large, accurately labeled EO datasets for training object detectors.

Session 4: Hands-on: Feature/Object Detection from Sentinel Imagery (Urban Monitoring Focus) (2.5 hours)

○	Case Study: Informal Settlement Growth/Building Detection in Metro Manila (e.g., focusing on areas like Quezon City or along the Pasig River corridor). This aligns with urban monitoring aspects of DRR and NRM as per the TOR.
○	Platform: Google Colab with GPU. The deep learning framework will be consistent with previous sessions.
○	Data: Pre-prepared Sentinel-2 optical image patches of an urban area in Metro Manila. These patches will come with annotations, specifically bounding boxes delineating buildings or informal settlement clusters. While Sentinel-1 can also be used for detecting built-up change, Sentinel-2 offers richer spectral information for visual object characteristics, which might be more intuitive for an initial object detection exercise.
○	Workflow (simplified to be feasible within the time, focusing on practical application):
1.	Load image patches and their corresponding bounding box annotations (coordinates and class labels).
2.	Option A (Simpler, Recommended): Utilize a pre-trained object detection model available through TensorFlow Hub or PyTorch Hub (e.g., a lightweight version of SSD or YOLO). Fine-tune this pre-trained model on the provided settlement/building dataset. This approach leverages transfer learning and is more practical for a short training session.
3.	Option B (More Involved, if time and audience proficiency allow): Implement a very simplified version of a single-stage detector like YOLO or SSD. This would involve understanding the grid cell approach, anchor boxes (if used), and the prediction of bounding box coordinates, objectness score, and class probabilities.
4.	Train or fine-tune the model using the prepared dataset.
5.	Evaluate the model's performance using appropriate object detection metrics (e.g., mean Average Precision - mAP).
6.	Visualize the detected bounding boxes overlaid on test images to qualitatively assess performance.
○	Conceptual Hurdle for EO Users: A common point of confusion can be the distinction between image classification (one label per image/patch), semantic segmentation (pixel-wise labels), and object detection (bounding boxes around instances). Clearly explaining these differences with visual examples is crucial. For object detection specifically, concepts like anchor boxes (in YOLO/SSD) and non-max suppression (NMS) for refining predictions can be challenging and will be explained intuitively. The urban monitoring case study directly addresses issues relevant to the Philippines, such as unplanned urban growth and disaster vulnerability in dense settlements. By focusing on using pre-trained models or simplified architectures, the hands-on session remains achievable while introducing core object detection concepts."