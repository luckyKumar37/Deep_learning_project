import os
from typing import List
from pydantic import BaseModel

class Settings(BaseModel):
    # Application Config
    APP_TITLE: str = "Cats vs Dogs Classification API"
    APP_VERSION: str = "1.0.0"
    
    # Server Config
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Config
    MODEL_PATH: str = os.getenv("MODEL_PATH", "backend/model.keras")
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "256"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "256"))
    CLASS_NAMES: List[str] = ["Cat", "Dog"]
    
    # Activation mode: "sigmoid" (Dense 1) or "softmax" (Dense 2)
    MODEL_ACTIVATION: str = os.getenv("MODEL_ACTIVATION", "sigmoid")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    NORMALIZATION_FACTOR: float = float(os.getenv("NORMALIZATION_FACTOR", "255.0"))
    
    # File Validation
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

settings = Settings()
