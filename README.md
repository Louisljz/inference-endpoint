# BISINDO Sign Language Recognition API

A FastAPI-based REST API that performs video classification using deep learning models. The system processes video inputs through MediaPipe for pose estimation and uses a custom transformer model for action classification.

## Project Structure

```
.
├── app/
│   ├── preprocessing.py    # Video preprocessing and pose estimation
│   ├── model.py           # ML model definition and inference
│   └── main.py           # FastAPI application and endpoints
├── model/                 # Directory containing model files
│   ├── sign_transformer.keras  # Trained model weights
│   └── labels.json       # Class labels mapping
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Local Development

### Manual Setup
1. Clone the repository:
```cmd
git clone https://github.com/yourusername/video-classification-api
cd video-classification-api
```

2. Create and activate virtual environment:
```cmd
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```cmd
pip install -r requirements.txt
```

4. Start the development server:
```cmd
cd app
fastapi dev main.py
```

### Run using Docker
1. **Build the Docker image**:
```cmd
docker build -t sign-recog-api .
```

2. **Run the Docker container**:
```cmd
docker run -d -p 8000:8000 sign-recog-api
```

## API Endpoints

### GET /
Health check endpoint that returns API status

### POST /predict
Accepts video URL and returns classification results

**Request Body:**
```json
{
    "url": "https://drive.google.com/uc?id=1ZtIq7sxkrHuRB7HOdD3d7MDMOdwRFdPm&export=download"
}
```

**Response:**
```json
{
    "label": "buka",
    "confidence": 0.95
}
```

## Inference Pipeline

1. **Video Input Processing**
   - Downloads video from provided URL
   - Performs motion-based trimming
   - Samples 113 frames from the video

2. **Pose Estimation**
   - Uses MediaPipe Holistic model to extract:
     - 33 pose landmarks
     - 21 left hand landmarks
     - 21 right hand landmarks
   - Calculates angles between key points

3. **Model Prediction**
   - Processes landmarks and angles through transformer model
   - Returns predicted action label with confidence score
