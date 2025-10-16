# Session 4: Model Deployment Guide
**From Training to Production**

---

## Overview

This guide covers deploying trained CNN models for operational Earth Observation monitoring in the Philippines.

---

## Table of Contents

1. [Model Export Formats](#export-formats)
2. [Inference Optimization](#optimization)
3. [Deployment Strategies](#deployment)
4. [GEE Integration](#gee-integration)
5. [Scaling Considerations](#scaling)
6. [Monitoring & Maintenance](#monitoring)

---

## <a name="export-formats"></a>1. Model Export Formats

### SavedModel (TensorFlow)

**Best for:** Production deployment, TensorFlow Serving

```python
# Export trained model
model.save('models/eurosat_cnn_v1')

# Folder structure created:
# models/eurosat_cnn_v1/
#   ├── saved_model.pb
#   ├── variables/
#   │   ├── variables.data-00000-of-00001
#   │   └── variables.index
#   └── assets/

print("✓ SavedModel exported")
```

**Loading:**
```python
loaded_model = tf.keras.models.load_model('models/eurosat_cnn_v1')
predictions = loaded_model.predict(test_images)
```

**Advantages:**
- ✅ Complete model + weights + optimizer state
- ✅ Language-agnostic (C++, Java, Go)
- ✅ TensorFlow Serving compatible
- ✅ Includes preprocessing layers

**File size:** ~50-500 MB depending on model

---

### H5 Format (Keras)

**Best for:** Quick exports, Python-only deployment

```python
# Export
model.save('models/eurosat_cnn_v1.h5')

# Load
loaded_model = tf.keras.models.load_model('models/eurosat_cnn_v1.h5')
```

**Advantages:**
- ✅ Single file
- ✅ Smaller than SavedModel
- ✅ Easy to share

**Disadvantages:**
- ❌ Python/Keras only
- ❌ No TensorFlow Serving support

---

### TFLite (Mobile/Edge)

**Best for:** Mobile apps, edge devices, resource-constrained environments

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save
with open('models/eurosat_cnn_v1.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

**Inference:**
```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/eurosat_cnn_v1.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
```

**Advantages:**
- ✅ **10-50× smaller** (quantization)
- ✅ Faster inference on mobile
- ✅ Works on Android, iOS, Raspberry Pi
- ✅ No TensorFlow dependency needed

**File size:** ~5-50 MB (quantized)

---

### ONNX (Cross-framework)

**Best for:** Multi-framework deployment, interoperability

```python
import tf2onnx

# Convert TensorFlow to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "models/eurosat_cnn_v1.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print("✓ ONNX model exported")
```

**Advantages:**
- ✅ Works with PyTorch, TensorFlow, scikit-learn
- ✅ ONNX Runtime for fast inference
- ✅ Cloud provider support

---

## <a name="optimization"></a>2. Inference Optimization

### Quantization

**Reduce precision:** FP32 → INT8

```python
# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for calibration
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()
```

**Benefits:**
- 4× smaller model size
- 2-3× faster inference
- Minimal accuracy loss (<1%)

---

### Pruning

**Remove unnecessary weights:**

```python
import tensorflow_model_optimization as tfmot

# Define pruning schedule
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,  # Remove 50% of weights
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile and fine-tune
model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_for_pruning.fit(X_train, y_train, epochs=2)

# Strip pruning wrappers
final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

**Benefits:**
- 2-4× smaller models
- Faster inference
- Minimal accuracy drop

---

### Batch Processing

**Process multiple images at once:**

```python
# Instead of one-by-one:
# for img in images:
#     pred = model.predict(img)  # Slow!

# Batch processing:
batch_size = 32
predictions = model.predict(images, batch_size=batch_size)  # Much faster!
```

**Speedup:** 5-10× for GPU, 2-3× for CPU

---

## <a name="deployment"></a>3. Deployment Strategies

### Strategy 1: Local Python Service

**Use case:** Small-scale, development, prototyping

```python
# Flask API
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/eurosat_cnn_v1')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    file = request.files['image']
    img = Image.open(file).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    # Predict
    prediction = model.predict(img_array)
    class_id = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][class_id])
    
    return jsonify({
        'class': class_names[class_id],
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Pros:** Simple, fast iteration  
**Cons:** Not scalable, single-threaded

---

### Strategy 2: TensorFlow Serving (Docker)

**Use case:** Production, scalable, multi-model

```dockerfile
# Dockerfile
FROM tensorflow/serving:latest

# Copy model
COPY models/eurosat_cnn_v1 /models/eurosat_cnn/1

# Environment variables
ENV MODEL_NAME=eurosat_cnn

# Expose ports
EXPOSE 8501

# Start serving
CMD ["/usr/bin/tf_serving_entrypoint.sh"]
```

```bash
# Build and run
docker build -t eurosat_serving .
docker run -p 8501:8501 eurosat_serving
```

**Client request:**
```python
import requests
import json

# Prepare data
data = json.dumps({
    "signature_name": "serving_default",
    "instances": test_images.tolist()
})

# Send request
response = requests.post('http://localhost:8501/v1/models/eurosat_cnn:predict', data=data)
predictions = response.json()['predictions']
```

**Pros:** Scalable, production-ready, GPU support  
**Cons:** More complex setup

---

### Strategy 3: Cloud Deployment

#### Google Cloud AI Platform

```bash
# Export model to Google Cloud Storage
gsutil cp -r models/eurosat_cnn_v1 gs://your-bucket/models/

# Create model
gcloud ai-platform models create eurosat_cnn --regions=us-central1

# Create version
gcloud ai-platform versions create v1 \
    --model=eurosat_cnn \
    --origin=gs://your-bucket/models/eurosat_cnn_v1 \
    --runtime-version=2.11 \
    --framework=tensorflow \
    --python-version=3.8
```

**Prediction:**
```python
from googleapiclient import discovery

service = discovery.build('ml', 'v1')
name = 'projects/YOUR_PROJECT/models/eurosat_cnn/versions/v1'

response = service.projects().predict(
    name=name,
    body={'instances': test_images.tolist()}
).execute()

predictions = response['predictions']
```

---

#### AWS SageMaker

```python
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Define model
model = TensorFlowModel(
    model_data='s3://your-bucket/models/eurosat_cnn_v1.tar.gz',
    role='your-sagemaker-role',
    framework_version='2.11'
)

# Deploy
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Predict
predictions = predictor.predict(test_images)
```

---

## <a name="gee-integration"></a>4. GEE Integration

### Option 1: GEE Assets + External API

**Workflow:**
1. Export Sentinel-2 patches from GEE
2. Send to external API (TensorFlow Serving)
3. Return predictions to GEE

```javascript
// GEE Code Editor
var sentinel = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(aoi)
    .filterDate('2024-01-01', '2024-12-31')
    .median();

// Export patches
var patches = sentinel.toArray();
Export.image.toDrive({
    image: patches,
    description: 'sentinel2_patches',
    scale: 10,
    region: aoi
});
```

**Python (external):**
```python
# Load exported patches
patches = np.load('sentinel2_patches.npy')

# Predict
predictions = model.predict(patches)

# Upload results back to GEE as asset
```

---

### Option 2: EEIFIED Models (Experimental)

**Convert TensorFlow to GEE:**

```python
import ee
import tensorflow as tf

# Note: Limited to specific architectures
# Check GEE documentation for supported layers

# Export model in GEE-compatible format
# (Requires specific preprocessing)
```

**Current Limitations:**
- Limited layer support
- Small models only
- Experimental feature

---

## <a name="scaling"></a>5. Scaling Considerations

### Horizontal Scaling (Multiple Instances)

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eurosat-serving
spec:
  replicas: 3  # 3 instances
  selector:
    matchLabels:
      app: eurosat
  template:
    metadata:
      labels:
        app: eurosat
    spec:
      containers:
      - name: serving
        image: your-registry/eurosat_serving:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

---

### Load Balancing

```nginx
# Nginx configuration
upstream eurosat_backend {
    least_conn;
    server eurosat-1:8501;
    server eurosat-2:8501;
    server eurosat-3:8501;
}

server {
    listen 80;
    location /v1/models/eurosat_cnn:predict {
        proxy_pass http://eurosat_backend;
    }
}
```

---

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def predict_cached(image_hash):
    # Cached predictions for repeated images
    image = load_image_from_hash(image_hash)
    return model.predict(image)

# Usage
img_hash = hashlib.md5(image_bytes).hexdigest()
prediction = predict_cached(img_hash)
```

---

## <a name="monitoring"></a>6. Monitoring & Maintenance

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Prediction request received")
    
    try:
        prediction = model.predict(image)
        logger.info(f"Prediction successful: {prediction}")
        return jsonify({'prediction': prediction})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500
```

---

### Metrics Tracking

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_counter.inc()
    
    start_time = time.time()
    prediction = model.predict(image)
    duration = time.time() - start_time
    
    prediction_duration.observe(duration)
    
    return jsonify({'prediction': prediction})
```

---

### Model Versioning

```python
# models/
#   ├── eurosat_cnn_v1/  (2024-01-15)
#   ├── eurosat_cnn_v2/  (2024-03-20)
#   └── eurosat_cnn_v3/  (2024-06-10)

# Load specific version
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v3')
model = tf.keras.models.load_model(f'models/eurosat_cnn_{MODEL_VERSION}')

# A/B testing
import random
def get_model():
    if random.random() < 0.9:
        return model_v3  # 90% traffic
    else:
        return model_v4  # 10% traffic (canary)
```

---

### Retraining Triggers

**When to retrain:**

1. **Scheduled:** Monthly/quarterly with new data
2. **Performance drop:** Accuracy falls below threshold
3. **Distribution shift:** New imagery characteristics
4. **User feedback:** Systematic errors reported

```python
# Automated retraining trigger
current_accuracy = evaluate_on_recent_data()

if current_accuracy < ACCURACY_THRESHOLD:
    logger.warning("Accuracy dropped below threshold!")
    trigger_retraining_pipeline()
```

---

## Philippine Deployment Example

**Scenario:** Mangrove monitoring system for Palawan

### Architecture

```
[Sentinel-2 Imagery (GEE)]
         ↓
[Preprocessing Pipeline]
         ↓
[Export patches to Cloud Storage]
         ↓
[TensorFlow Serving (3 replicas)]
         ↓
[Predictions stored in PostgreSQL]
         ↓
[Web Dashboard for Visualstration]
         ↓
[Alerts to Conservation Teams]
```

### Implementation

```python
# Daily batch processing
def daily_mangrove_monitoring():
    # 1. Get new Sentinel-2 imagery from GEE
    patches = download_sentinel2_patches(date=today)
    
    # 2. Predict
    predictions = model.predict(patches)
    
    # 3. Calculate mangrove extent
    mangrove_area = calculate_area(predictions, class_id=MANGROVE_CLASS)
    
    # 4. Compare with historical
    change = mangrove_area - historical_area
    
    # 5. Alert if significant change
    if abs(change) > ALERT_THRESHOLD:
        send_alert_to_conservation_team(change, location=palawan)
    
    # 6. Store results
    store_in_database(predictions, mangrove_area, date=today)

# Run daily at 2 AM
schedule.every().day.at("02:00").do(daily_mangrove_monitoring)
```

---

## Checklist: Deployment Readiness

**Model:**
- [ ] Model trained and validated (>90% accuracy)
- [ ] Exported in appropriate format
- [ ] Optimized (quantized/pruned if needed)
- [ ] Versioned and documented

**Infrastructure:**
- [ ] Deployment environment chosen
- [ ] GPU/CPU resources allocated
- [ ] Scaling strategy defined
- [ ] Load balancer configured (if needed)

**Monitoring:**
- [ ] Logging implemented
- [ ] Metrics tracked (latency, accuracy)
- [ ] Alerts configured
- [ ] Dashboard created

**Testing:**
- [ ] Unit tests for preprocessing
- [ ] Integration tests for API
- [ ] Load testing completed
- [ ] Disaster recovery plan

**Documentation:**
- [ ] API documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide
- [ ] Contact information

---

**Last Updated:** October 2025  
**For:** CopPhil Advanced Training - Session 4
