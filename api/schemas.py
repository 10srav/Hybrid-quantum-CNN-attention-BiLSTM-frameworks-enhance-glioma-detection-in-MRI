"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionRequest(BaseModel):
    """Prediction request schema (for JSON input)."""
    image_base64: Optional[str] = None
    return_heatmap: bool = True
    target_class: Optional[int] = None


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: str
    glioma_probability: float = Field(..., ge=0, le=1)
    non_glioma_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    heatmap_base64: Optional[str] = None
    processing_time_ms: float
    model_version: str = "1.0.0"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    images_base64: List[str]
    return_heatmaps: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information schema."""
    name: str = "HybridQCNN"
    version: str = "1.0.0"
    architecture: str = "Quantum CNN + Attention + BiLSTM"
    num_parameters: int
    num_classes: int = 2
    input_shape: str = "(B, 16, 3, 128, 128)"
    classes: List[str] = ["non_glioma", "glioma"]


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    exp: Optional[datetime] = None


class UserCreate(BaseModel):
    """User creation schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response schema."""
    id: int
    username: str
    email: str
    is_active: bool = True
    created_at: datetime


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionLog(BaseModel):
    """Prediction logging schema."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    prediction: str
    probability: float
    processing_time_ms: float
    image_hash: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
