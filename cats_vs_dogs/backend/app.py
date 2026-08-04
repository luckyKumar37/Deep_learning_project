import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.schemas import (
    StatusResponse,
    HealthResponse,
    PredictionResponse,
    BatchPredictionResponse,
    BatchPredictionItem,
    ErrorDetail
)
from backend.predictor import predictor
from backend.utils import (
    validate_uploaded_file,
    bytes_to_image,
    preprocess_image
)

# Configure Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cats_vs_dogs.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager:
    Loads Keras model once at startup and cleans up on shutdown.
    """
    logger.info("Initializing FastAPI application...")
    logger.info(f"TensorFlow version: {tf.__version__}")
    try:
        predictor.load()
    except Exception as e:
        logger.critical(f"Critical failure loading TensorFlow model: {e}")
    yield
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="FastAPI Backend service for classifying uploaded images as Cat or Dog using a trained Keras CNN model.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for flexible frontend connection
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    HTTP Request logging middleware recording method, path, timing and status code.
    """
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration_ms}ms"
    )
    return response

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler to format HTTP error responses cleanly.
    """
    logger.warning(f"HTTPException [{exc.status_code}] on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all unhandled exception handler.
    """
    logger.error(f"Unhandled Exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )

@app.get(
    "/",
    response_model=StatusResponse,
    summary="Root Status Endpoint",
    description="Returns simple running status of the API."
)
async def root():
    return StatusResponse(status="running")

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check Endpoint",
    description="Returns status of backend, TensorFlow version, and whether the model is loaded."
)
async def health_check():
    is_healthy = predictor.is_loaded
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        tensorflow_version=tf.__version__,
        model_loaded=is_healthy
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify Image as Cat or Dog",
    description="Accepts a single uploaded image file (JPG, JPEG, PNG) and returns prediction class, confidence, and class probabilities."
)
async def predict_image(file: UploadFile = File(...)):
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available or failed to load."
        )

    # 1. Validate file metadata
    validate_uploaded_file(file)

    # 2. Read raw bytes
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload file stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content."
        )

    # 3. Parse PIL image & preprocess
    pil_image = bytes_to_image(contents)
    processed_array = preprocess_image(pil_image)

    # 4. Perform Inference
    try:
        predicted_class, confidence, probabilities = predictor.predict(processed_array)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed."
        )

    return PredictionResponse(
        prediction=predicted_class,
        confidence=confidence,
        probabilities=probabilities
    )

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify Multiple Images in Batch",
    description="Accepts multiple image files and returns list of prediction results for each file."
)
async def predict_batch(files: List[UploadFile] = File(...)):
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available or failed to load."
        )

    results: List[BatchPredictionItem] = []

    for file in files:
        filename = file.filename or "unknown"
        try:
            validate_uploaded_file(file)
            contents = await file.read()
            pil_img = bytes_to_image(contents)
            img_array = preprocess_image(pil_img)
            pred_class, conf, probs = predictor.predict(img_array)

            results.append(BatchPredictionItem(
                filename=filename,
                status="success",
                prediction=pred_class,
                confidence=conf,
                probabilities=probs
            ))
        except HTTPException as http_ex:
            results.append(BatchPredictionItem(
                filename=filename,
                status="error",
                error=str(http_ex.detail)
            ))
        except Exception as ex:
            results.append(BatchPredictionItem(
                filename=filename,
                status="error",
                error=f"Processing failed: {str(ex)}"
            ))

    return BatchPredictionResponse(
        total=len(results),
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
