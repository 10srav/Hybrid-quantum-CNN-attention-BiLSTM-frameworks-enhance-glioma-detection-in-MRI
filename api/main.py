"""
FastAPI backend for Hybrid Quantum CNN Glioma Detection.
Provides REST API endpoints for model inference.
"""

import os
import sys
import time
import base64
from io import BytesIO
from pathlib import Path
from datetime import timedelta
from typing import Optional

import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from models.hybrid_qcnn import HybridQCNN, create_model
from data.preprocessing import preprocess_for_model
from utils.gradcam import generate_heatmap
from utils.helpers import load_checkpoint, get_device

from .schemas import (
    HealthResponse, PredictionResponse, BatchPredictionResponse,
    ModelInfo, Token, ErrorResponse
)
from .auth import (
    authenticate_user, create_access_token, get_current_user,
    get_current_active_user
)

config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Quantum CNN Glioma Detection API",
    description="Production API for glioma detection in MRI scans using Hybrid Quantum CNN with Attention and BiLSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, device
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(model_type="full", num_classes=2)
    model = model.to(device)
    model.eval()
    
    # Try to load pretrained weights
    checkpoint_path = config.api.model_path
    if checkpoint_path.exists():
        try:
            model, _, _ = load_checkpoint(model, str(checkpoint_path), device=device)
            print(f"Loaded model weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using randomly initialized model")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Using randomly initialized model (for demo purposes)")
    
    print("Model loaded and ready for inference!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global model
    model = None
    torch.cuda.empty_cache()


# ====================
# Health & Info
# ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Hybrid Quantum CNN Glioma Detection API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    num_params = sum(p.numel() for p in model.parameters())
    
    return ModelInfo(
        name="HybridQCNN",
        num_parameters=num_params
    )


# ====================
# Authentication
# ====================

@app.post("/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token."""
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=config.api.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.api.access_token_expire_minutes * 60
    )


@app.get("/users/me", tags=["Auth"])
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Get current user info."""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "is_active": current_user["is_active"]
    }


# ====================
# Prediction Endpoints
# ====================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(...),
    return_heatmap: bool = True,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Predict glioma from MRI image.
    
    - **file**: MRI image file (JPEG, PNG)
    - **return_heatmap**: Whether to return Grad-CAM heatmap
    
    Returns prediction with probability and optional attention heatmap.
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        input_tensor = preprocess_for_model(image_rgb)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dim
        
        # Inference
        with torch.no_grad():
            outputs, attention = model(input_tensor, return_attention=return_heatmap)
            probs = torch.softmax(outputs, dim=1)
        
        # Get predictions
        glioma_prob = probs[0, 1].item()
        non_glioma_prob = probs[0, 0].item()
        prediction = "glioma" if glioma_prob > 0.5 else "non_glioma"
        confidence = max(glioma_prob, non_glioma_prob)
        
        # Generate heatmap
        heatmap_base64 = None
        if return_heatmap:
            try:
                heatmap_result = generate_heatmap(
                    model=model,
                    input_tensor=input_tensor,
                    original_image=image_rgb,
                    target_class=1 if prediction == "glioma" else 0
                )
                heatmap_base64 = heatmap_result['base64']
            except Exception as e:
                print(f"Heatmap generation failed: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=prediction,
            glioma_probability=glioma_prob,
            non_glioma_probability=non_glioma_prob,
            confidence=confidence,
            heatmap_base64=heatmap_base64,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: list[UploadFile] = File(...),
    return_heatmaps: bool = False,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Batch prediction for multiple MRI images.
    Requires authentication.
    
    - **files**: List of MRI image files
    - **return_heatmaps**: Whether to return heatmaps (slower)
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 16:
        raise HTTPException(status_code=400, detail="Maximum 16 images per batch")
    
    start_time = time.time()
    predictions = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                predictions.append(PredictionResponse(
                    prediction="error",
                    glioma_probability=0.0,
                    non_glioma_probability=0.0,
                    confidence=0.0,
                    processing_time_ms=0.0
                ))
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_for_model(image_rgb)
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs, _ = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            glioma_prob = probs[0, 1].item()
            non_glioma_prob = probs[0, 0].item()
            
            predictions.append(PredictionResponse(
                prediction="glioma" if glioma_prob > 0.5 else "non_glioma",
                glioma_probability=glioma_prob,
                non_glioma_probability=non_glioma_prob,
                confidence=max(glioma_prob, non_glioma_prob),
                heatmap_base64=None,
                processing_time_ms=0.0
            ))
            
        except Exception as e:
            predictions.append(PredictionResponse(
                prediction="error",
                glioma_probability=0.0,
                non_glioma_probability=0.0,
                confidence=0.0,
                processing_time_ms=0.0
            ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time_ms=total_time
    )


@app.post("/predict/base64", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_base64(
    image_base64: str,
    return_heatmap: bool = True
):
    """
    Predict from base64-encoded image.
    
    - **image_base64**: Base64-encoded image string
    - **return_heatmap**: Whether to return heatmap
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Decode base64
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_for_model(image_rgb)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs, _ = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        glioma_prob = probs[0, 1].item()
        non_glioma_prob = probs[0, 0].item()
        prediction = "glioma" if glioma_prob > 0.5 else "non_glioma"
        
        heatmap_base64 = None
        if return_heatmap:
            try:
                heatmap_result = generate_heatmap(
                    model=model,
                    input_tensor=input_tensor,
                    original_image=image_rgb
                )
                heatmap_base64 = heatmap_result['base64']
            except:
                pass
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=prediction,
            glioma_probability=glioma_prob,
            non_glioma_probability=non_glioma_prob,
            confidence=max(glioma_prob, non_glioma_prob),
            heatmap_base64=heatmap_base64,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ====================
# Error Handlers
# ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True
    )
