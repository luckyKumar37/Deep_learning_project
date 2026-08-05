# Cats vs Dogs Image Classification Web Application

A clean, modular, and production-quality full-stack machine learning application for classifying images as **Cat** or **Dog** using a pre-trained Keras model.

![Architecture Diagram](https://img.shields.io/badge/Architecture-Decoupled_Microservice-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI_0.100+-009688)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit_1.25+-FF4B4B)
![Model](https://img.shields.io/badge/Model-TensorFlow_Keras-FF6F00)

---

## 🌐 Live Deployments

- **🎨 Streamlit Web UI:** [https://dogvscats-hjh2rlqna6btj2xz9bzjph.streamlit.app/](https://dogvscats-hjh2rlqna6btj2xz9bzjph.streamlit.app/)
- **⚡ FastAPI REST API (Railway):** [https://deeplearningproject-production.up.railway.app/docs](https://deeplearningproject-production.up.railway.app/docs)

---

## 🏛️ System Architecture

The application enforces strict separation of concerns between client and server:

```text
                                 HTTP REST API
 ┌──────────────────────┐  ------------------------>  ┌──────────────────────┐
 │                      │                             │                      │
 │  Streamlit Frontend  │  <------------------------  │   FastAPI Backend    │
 │   (Port: 8501)       │     JSON Predictions        │     (Port: 8000)     │
 │                      │                             │                      │
 └──────────────────────┘                             └──────────┬───────────┘
   - No TensorFlow dependencies                                  │ Loads once
   - Upload & Preview Images                                 ┌───┴──────────┐
   - Displays Metrics & Graphs                             │ model.keras  │
                                                             └──────────────┘
```

1. **Frontend (Streamlit):** Serves as an interactive client. It **never imports TensorFlow or accesses `model.keras` directly**. All inference requests are dispatched via HTTP `POST` calls to the backend.
2. **Backend (FastAPI):** Responsible for loading the TensorFlow model once at startup, image validation, preprocessing (Pillow -> RGB -> 256x256 -> NumPy batch -> normalization), inference execution, and structured JSON responses.

---

## 📁 Folder Structure

```text
cats-vs-dogs/
│
├── backend/
│   ├── app.py          # FastAPI application routes & middleware
│   ├── model.keras     # Trained TensorFlow/Keras model
│   ├── config.py       # Pydantic environment configuration
│   ├── utils.py        # Image validation & preprocessing pipeline
│   ├── predictor.py    # Model singleton manager (Sigmoid & Softmax support)
│   ├── schemas.py      # Pydantic request & response schemas
│   ├── requirements.txt# Backend dependencies
│   ├── .env.example    # Example environment variables
│   └── README.md       # Backend documentation
│
├── frontend/
│   ├── app.py          # Streamlit UI dashboard
│   ├── config.py       # Frontend client configuration
│   ├── requirements.txt# Frontend dependencies
│   └── README.md       # Frontend documentation
│
├── README.md           # Master project documentation
└── .gitignore          # Git exclusion rules
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.9+ installed
- Virtual environment (optional but recommended)

### 1. Install Dependencies

Install backend requirements:
```bash
pip install -r backend/requirements.txt
```

Install frontend requirements:
```bash
pip install -r frontend/requirements.txt
```

---

### 2. Run the Application

#### Step A: Start the FastAPI Backend Server
Run the FastAPI backend on port `8000`:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- API Root: `http://localhost:8000/`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- ReDoc API Documentation: `http://localhost:8000/redoc`

#### Step B: Start the Streamlit Frontend Client
In a separate terminal window, launch the Streamlit frontend:
```bash
streamlit run frontend/app.py
```
- Web UI Dashboard: `http://localhost:8501`

---

## 🌐 API Specification

### `GET /`
Returns status of the API server.
```json
{
    "status": "running"
}
```

### `GET /health`
Returns backend health status, TensorFlow version, and model load status.
```json
{
    "status": "healthy",
    "tensorflow_version": "2.16.1",
    "model_loaded": true
}
```

### `POST /predict`
Classifies a single uploaded image (`JPG`, `JPEG`, `PNG`).

**Response Example:**
```json
{
    "prediction": "Dog",
    "confidence": 98.42,
    "probabilities": {
        "Cat": 0.0158,
        "Dog": 0.9842
    }
}
```

### `POST /predict/batch`
Classifies multiple uploaded images in batch.

---

## ⚙️ Model Activation & Preprocessing

- **Target Image Size:** `256 x 256` RGB.
- **Normalization:** Pixel values divided by `255.0` (`[0.0, 1.0]` range).
- **Activation Support:** The `predictor.py` module supports both single-output `sigmoid` (`Dense(1)`) and multi-output `softmax` (`Dense(2)`), configurable via `MODEL_ACTIVATION` setting in `config.py` or `.env`.

---

## ✨ Features & Highlights

- **Decoupled Architecture:** Strict separation between Streamlit and FastAPI.
- **Portfolio-Grade UI:** Glassmorphism card layouts, custom status badges, progress bars, and responsive spacing.
- **Session Prediction History:** Log and review predictions during your session.
- **Batch Processing:** Process multiple images in a single call.
- **JSON Result Export:** Download prediction metrics as structured JSON files.
- **Comprehensive Error Handling:** Validates empty files, corrupted images, unsupported extensions, backend offline states, and timeouts gracefully.
