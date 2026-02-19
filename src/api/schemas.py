"""
API request/response schemas
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class PredictionItem(BaseModel):
    """Single prediction item (alias 'class' for JSON, 'class' is reserved in Python)."""
    model_config = ConfigDict(populate_by_name=True)
    class_name: str = Field(..., alias="class", description="Predicted class name")
    class_index: int = Field(..., description="Class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class PredictionResponse(BaseModel):
    """Prediction response"""
    predictions: List[PredictionItem] = Field(..., description="Top-k predictions")
    top_prediction: PredictionItem = Field(..., description="Top prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    results: List[PredictionResponse] = Field(..., description="Predictions for each image")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Model name")
    num_classes: int = Field(..., description="Number of classes")
    input_size: int = Field(..., description="Input image size")
