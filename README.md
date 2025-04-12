# InsightFace Ray Serve API

A scalable REST API for face detection, attribute extraction, and comparison, powered by [InsightFace](https://github.com/deepinsight/insightface) and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html).

## Overview

This project provides a production-ready REST API for face analysis tasks, deployed using Ray Serve for automatic scaling. The API supports:

- Face detection and landmark extraction
- Face attribute analysis (gender, age, race)
- Face quality and mask detection
- Face comparison (similarity scoring)

The service accepts images via base64 encoding or URLs (including S3 storage) and returns detailed information about detected faces.

## Features

- **Scalable deployment**: Uses Ray Serve's autoscaling to handle variable loads
- **Multiple image input methods**: Support for base64-encoded images and URLs (HTTP/HTTPS/S3)
- **Comprehensive face analysis**: Detects and analyzes multiple faces in a single image
- **Face comparison**: Compares faces from two different images for similarity
- **Health check endpoint**: Monitors service status

## Installation

### Prerequisites

- Python 3.11+
- OpenCV
- CUDA support (recommended for production deployments)

### Setup with Conda

The recommended way to set up the environment is using Conda:

```bash
# Clone the repository
git clone https://github.com/BillDuke13/insightface-ray-serve-api.git
cd insightface-ray-serve-api

# Create and activate the conda environment
conda env create -f environment.yml
conda activate insightface-ray-serve-api
```

### Manual Setup

If you prefer not to use Conda, you can install the dependencies manually:

```bash
pip install ray[serve]==2.44.1 inspireface fastapi uvicorn opencv-python numpy pydantic requests boto3
```

## Usage

### Starting the API Server

To start the API server:

```bash
python -m ray.serve run src.api:app_instance
```

This will start the server on the default port (8000). You can customize the port and other Ray Serve configuration parameters as needed.

### API Endpoints

The API provides the following endpoints:

#### Face Detection and Attribute Extraction

```
POST /v1/models/insightface:detect
```

Detects faces in an image and extracts detailed attributes for each face.

#### Face Comparison

```
POST /v1/models/insightface:compare
```

Compares the primary faces from two images and returns a similarity score.

#### Health Check

```
GET /-/healthz
```

Returns the health status of the API.

### Example API Requests

#### Face Detection with Base64 Image

```python
import requests
import base64

# Read an image file and convert to base64
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# Make the API request
response = requests.post(
    "http://localhost:8000/v1/models/insightface:detect",
    json={"image_base64": image_base64}
)

# Process the response
faces = response.json()["faces"]
print(f"Detected {len(faces)} faces")
for i, face in enumerate(faces):
    print(f"Face {i+1}:")
    print(f"  - Confidence: {face['confidence']:.4f}")
    if face.get('gender'):
        print(f"  - Gender: {face['gender']}")
    if face.get('age_bracket'):
        print(f"  - Age bracket: {face['age_bracket']}")
```

#### Face Detection with Image URL

```python
import requests

# Make the API request with an image URL
response = requests.post(
    "http://localhost:8000/v1/models/insightface:detect",
    json={"image_url": "https://example.com/image.jpg"}
)

# Process the response
faces = response.json()["faces"]
```

#### Face Comparison

```python
import requests

# Compare faces from two image URLs
response = requests.post(
    "http://localhost:8000/v1/models/insightface:compare",
    json={
        "image1_url": "https://example.com/person1.jpg",
        "image2_url": "https://example.com/person2.jpg"
    }
)

# Get the similarity score
similarity = response.json()["similarity"]
print(f"Similarity score: {similarity:.4f}")
```

## API Reference

### Detection Response Format

The response from the detection endpoint includes the following information for each detected face:

```json
{
  "faces": [
    {
      "bounding_box": [x1, y1, x2, y2],
      "confidence": 0.99,
      "landmarks": [[x1, y1], [x2, y2], ...],
      "feature": [0.1, 0.2, ...],
      "roll": 2.5,
      "yaw": 1.3,
      "pitch": 0.7,
      "quality": 0.98,
      "mask_confidence": 0.01,
      "liveness_confidence": 0.96,
      "gender": "Female",
      "age_bracket": "20-29 years old",
      "race": "Asian"
    },
    ...
  ]
}
```

### Comparison Response Format

The response from the comparison endpoint includes:

```json
{
  "similarity": 0.85
}
```

The similarity score ranges from 0 to 1, where higher values indicate greater similarity between the faces.

## Performance Considerations

- The service automatically scales based on load, but initial requests may be slower due to model loading
- For production deployment, consider:
  - Using CUDA-enabled devices for improved performance
  - Adjusting the `min_replicas` parameter for consistent response times
  - Monitoring memory usage, especially with multiple parallel requests

## License

This project is licensed under the terms of the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
