import os
import streamlit as st

def get_backend_url() -> str:
    # 1. Check Streamlit Cloud Secrets (st.secrets)
    try:
        if "BACKEND_URL" in st.secrets:
            return str(st.secrets["BACKEND_URL"]).strip()
    except Exception:
        pass
    # 2. Check Environment Variable
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url.strip()
    # 3. Default Localhost
    return "http://localhost:8000"

DEFAULT_BACKEND_URL = get_backend_url()
REQUEST_TIMEOUT = 15  # seconds
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_IMAGE_SIZE_MB = 10
