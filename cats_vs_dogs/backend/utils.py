import io
import os
import logging
from typing import Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, status, UploadFile
from backend.config import settings

logger = logging.getLogger("cats_vs_dogs.utils")

def validate_uploaded_file(file: UploadFile, max_size_mb: int = settings.MAX_FILE_SIZE_MB) -> None:
    """
    Validates uploaded file extensions and basic size checks.
    Raises HTTPException with appropriate status codes on failure.
    """
    if not file or not file.filename:
        logger.warning("Empty file uploaded")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename is missing."
        )
        
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"Unsupported file extension: '{ext}' for file '{file.filename}'")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """
    Converts raw image bytes to a PIL Image instance.
    Validates corrupted/unreadable images.
    """
    if not image_bytes or len(image_bytes) == 0:
        logger.warning("Empty image bytes received")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )
        
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify image integrity
        # Re-open after verify as verify() alters file pointer state
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        logger.error(f"Failed to parse image file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid or corrupted image file."
        )

def preprocess_image(
    image: Image.Image, 
    target_size: Tuple[int, int] = (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT),
    normalization_factor: float = settings.NORMALIZATION_FACTOR
) -> np.ndarray:
    """
    Preprocesses PIL image for Keras model prediction:
    1. Converts image to RGB.
    2. Resizes to model's expected target dimensions (e.g. 256x256).
    3. Converts to float32 NumPy array.
    4. Normalizes pixel values (e.g. / 255.0).
    5. Adds batch dimension (1, H, W, C).
    """
    # Convert to RGB mode (handles PNG RGBA, grayscale, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize image
    image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to NumPy array
    img_array = np.asarray(image_resized, dtype=np.float32)
    
    # Normalize pixel values
    if normalization_factor > 0:
        img_array = img_array / normalization_factor
        
    # Add batch dimension
    batch_array = np.expand_dims(img_array, axis=0)
    
    logger.debug(f"Preprocessed image shape: {batch_array.shape}, min: {batch_array.min()}, max: {batch_array.max()}")
    return batch_array
