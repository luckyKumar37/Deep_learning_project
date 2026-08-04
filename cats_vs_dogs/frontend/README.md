# Cats vs Dogs Frontend (Streamlit Client)

Streamlit Web Interface for the Cats vs Dogs Image Classification Application.

## Architecture & Design Principles

- **No Model Dependencies:** The frontend strictly communicates with the FastAPI backend over HTTP REST APIs. It never imports `tensorflow` or loads `.keras` files directly.
- **Dynamic Configuration:** Configurable backend API URL adjustable via sidebar, environment variable `BACKEND_URL`, or `config.py`.
- **Live Health Monitoring:** Ping status check against backend `/health` endpoint with visual status indicator.
- **Rich Interactive UI:** Supports drag-and-drop image upload, single and batch classification, JSON result download, inference latency tracking, and session prediction history.

## Directory Structure

```text
frontend/
├── app.py          # Streamlit UI app entry point
├── config.py       # Frontend client configuration
├── requirements.txt# Frontend Python dependencies
└── README.md       # Frontend documentation
```

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r frontend/requirements.txt
```

### 2. Launch Streamlit Application
From the project root:
```bash
streamlit run frontend/app.py
```
By default, Streamlit will open at `http://localhost:8501`.
