from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class StatusResponse(BaseModel):
    status: str = Field(default="running", example="running")

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    tensorflow_version: str = Field(..., example="2.16.1")
    model_loaded: bool = Field(..., example=True)

class PredictionResponse(BaseModel):
    prediction: str = Field(..., example="Dog")
    confidence: float = Field(..., example=98.42)
    probabilities: Dict[str, float] = Field(
        ..., 
        example={"Cat": 0.0158, "Dog": 0.9842}
    )

class BatchPredictionItem(BaseModel):
    filename: str = Field(..., example="pet.jpg")
    status: str = Field(..., example="success")
    prediction: Optional[str] = Field(default=None, example="Dog")
    confidence: Optional[float] = Field(default=None, example=98.42)
    probabilities: Optional[Dict[str, float]] = Field(default=None, example={"Cat": 0.0158, "Dog": 0.9842})
    error: Optional[str] = Field(default=None)

class BatchPredictionResponse(BaseModel):
    total: int = Field(..., example=2)
    results: List[BatchPredictionItem]

class ErrorDetail(BaseModel):
    error: str = Field(..., example="Invalid image format")
    details: Optional[str] = Field(default=None)
