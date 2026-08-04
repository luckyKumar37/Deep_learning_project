# Cats vs Dogs Backend (FastAPI Service)

FastAPI REST service for classifying uploaded images as **Cat** or **Dog** using a pre-trained Keras model (`model.keras`).

## Features

- **Lifespan Model Loading:** Loads Keras model once at application startup.
- **Preprocessing Pipeline:** Converts images to RGB, resizes to target dimensions `(256, 256)`, normalizes pixels, and batches NumPy arrays.
- **Flexible Activation:** Supports both `sigmoid` (Dense 1) and `softmax` (Dense 2) model activations.
- **Robust Exception Handling:** Validates format (`.jpg`, `.jpeg`, `.png`), size, and file corruption with structured JSON error responses.
- **Automated API Docs:** Interactive OpenAPI docs available at `/docs` and `/redoc`.
- **Request Logging:** Logs request timestamps, execution latency, predictions, and errors.

## Directory Structure

```text
backend/
├── app.py          # FastAPI application & route endpoints
├── config.py       # Pydantic configuration & settings
├── model.keras     # Trained TensorFlow/Keras model
├── predictor.py    # Model loading singleton & inference module
├── utils.py        # Image validation & preprocessing functions
├── schemas.py      # Pydantic request/response schemas
├── requirements.txt# Backend Python dependencies
├── .env.example    # Template environment file
└── README.md       # Backend documentation
```

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Start API Server
From the project root directory:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns `{"status": "running"}` |
| `GET` | `/health` | Health metrics, TensorFlow version, and model load status |
| `POST` | `/predict` | Predicts class for a single uploaded image |
| `POST` | `/predict/batch` | Predicts classes for multiple uploaded images |
