"""
FastAPI application for plant classification
"""

import io
import time
from typing import Optional, Dict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

from src.config import settings
from src.training.model import DandelionGrassClassifier


# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Dandelion vs Grass classification"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Global model variable
model = None
device = None
class_names = ["dandelion", "grass"]


def create_model(num_classes=2):
    """Create DandelionGrassClassifier model"""
    model = DandelionGrassClassifier(pretrained=False, num_classes=num_classes)
    return model


def load_model_from_checkpoint(model_path: str, device: torch.device):
    """Load model from checkpoint file"""
    model = create_model(num_classes=2)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load the state dict directly (no need to remove prefix since we're using the same class)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_image_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    prediction_time: float
    timestamp: str


@app.on_event("startup")
async def load_model_on_startup():
    """Load model when API starts"""
    global model, device
    
    logger.info("Loading model...")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    try:
        model_path = Path("models/best_model.pth")
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}, creating untrained model for demo")
            model = create_model(num_classes=2)
            model.to(device)
        else:
            model = load_model_from_checkpoint(str(model_path), device)
            logger.success(f"✅ Model loaded from {model_path}")
        
        model.eval()
        logger.success("✅ Model ready for predictions")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Create a dummy model to keep API running
        model = create_model(num_classes=2)
        model.to(device)
        model.eval()
        logger.warning("⚠️  Running with untrained model")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Classification API",
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    
    Args:
        file: Image file to classify
        
    Returns:
        Prediction response with class and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Preprocess image
        transform = get_image_transforms()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        prediction_time = time.time() - start_time
        
        logger.info(f"Prediction: {predicted_class} ({confidence_score:.4f}) in {prediction_time:.3f}s")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence_score,
            prediction_time=prediction_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "ResNet18",
        "classes": class_names,
        "image_size": 224,
        "device": str(device),
        "model_type": "ResNet18",
        "num_parameters": sum(p.numel() for p in model.parameters())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)