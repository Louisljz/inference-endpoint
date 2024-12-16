from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from preprocessing import VideoProcessor
from model import VideoClassifier
import urllib.request
import tempfile
import os

app = FastAPI(title="Video Classification API")

# Initialize model
processor = VideoProcessor()
model = VideoClassifier()

class VideoRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Video Classification API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_video(video_request: VideoRequest):
    try:
        # Download video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            urllib.request.urlretrieve(video_request.url, tmp_file.name)
            tmp_file.close()
            
            # Preprocess video
            input_data = processor.process_video(tmp_file.name)
            
            # Get prediction
            prediction = model.predict(input_data)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return PredictionResponse(
                label=prediction["label"],
                confidence=round(prediction["confidence"], 2)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
