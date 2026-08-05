import io
import json
import time
import requests
import streamlit as st
from PIL import Image
try:
    from frontend.config import DEFAULT_BACKEND_URL, REQUEST_TIMEOUT, ALLOWED_EXTENSIONS
except ModuleNotFoundError:
    from config import DEFAULT_BACKEND_URL, REQUEST_TIMEOUT, ALLOWED_EXTENSIONS

# Set Page Config
st.set_page_config(
    page_title="Cats vs Dogs AI Classifier",
    page_icon="🐱🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Portfolio-Grade Aesthetic Styling
st.markdown("""
<style>
    /* Main Theme Adjustments */
    .main {
        background-color: #0e1117;
    }
    
    /* Card Containers */
    .custom-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Status Badge Styling */
    .status-online {
        display: inline-block;
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .status-offline {
        display: inline-block;
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid #e74c3c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Result Header Cards */
    .prediction-card-dog {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(30, 60, 114, 0.4);
    }
    .prediction-card-cat {
        background: linear-gradient(135deg, #d4418e 0%, #0652dd 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(212, 65, 142, 0.4);
    }
    
    .prediction-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .prediction-confidence {
        font-size: 1.4rem;
        font-weight: 600;
        opacity: 0.9;
    }
    
    /* Metrics box */
    .metric-container {
        display: flex;
        justify-content: space-around;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "history" not in st.session_state:
    st.session_state.history = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

# Helper function to check backend health
def check_backend_health(url: str):
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=4)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

# --- SIDEBAR ---
st.sidebar.markdown("## 🐾 App Navigation & Config")

# Backend URL Configuration
backend_url_input = st.sidebar.text_input(
    "Backend API Service URL",
    value=st.session_state.backend_url,
    help="FastAPI server endpoint URL"
)
st.session_state.backend_url = backend_url_input.strip()

# Check Backend Health Connection
is_healthy, health_data = check_backend_health(st.session_state.backend_url)

st.sidebar.markdown("### 🔌 System Health")
if is_healthy:
    st.sidebar.markdown('<div class="status-online">🟢 Backend Connected</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"**TensorFlow Version:** `{health_data.get('tensorflow_version', 'N/A')}`")
    st.sidebar.caption(f"**Model Loaded:** `{health_data.get('model_loaded', False)}`")
else:
    st.sidebar.markdown('<div class="status-offline">🔴 Backend Offline</div>', unsafe_allow_html=True)
    st.sidebar.error(f"Cannot reach FastAPI backend at `{st.session_state.backend_url}`.")
    st.sidebar.info("💡 Ensure the FastAPI backend is running: `uvicorn backend.app:app --port 8000`")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Model Details")
st.sidebar.write("**Architecture:** Convolutional Neural Network (CNN)")
st.sidebar.write("**Input Resolution:** 256 x 256 pixels (RGB)")
st.sidebar.write("**Target Classes:** Cat 🐱 vs Dog 🐶")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.caption(
    "Production-grade Deep Learning Web Application with FastAPI microservice backend "
    "and Streamlit client frontend."
)

# --- MAIN PAGE HEADER ---
st.title("🐱🐶 Cats vs Dogs Image Classification")
st.markdown(
    "Upload an image of a **Cat** or **Dog** to get instant prediction results, confidence scores, "
    "and probability distributions powered by our FastAPI deep learning backend."
)
st.markdown("---")

# Main Page Tabs
tab_single, tab_batch, tab_history = st.tabs([
    "📸 Single Prediction", 
    "📂 Batch Prediction", 
    "📜 Session History"
])

# ==========================================
# TAB 1: SINGLE IMAGE PREDICTION
# ==========================================
with tab_single:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("1. Select & Preview Image")
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, JPEG, PNG)...",
            type=ALLOWED_EXTENSIONS,
            key="single_file_uploader"
        )

        if uploaded_file is not None:
            try:
                # Open & preview PIL Image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                st.caption(f"Dimensions: {image.width}x{image.height} px | Size: {round(len(uploaded_file.getvalue()) / 1024, 2)} KB")
            except Exception as e:
                st.error("Error displaying image preview. The file might be corrupted.")
        else:
            st.info("👆 Upload an image above to begin classification.")

    with col_result:
        st.subheader("2. Prediction Analysis")
        
        if uploaded_file is not None:
            predict_btn = st.button("🚀 Classify Image", type="primary", use_container_width=True)

            if predict_btn:
                if not is_healthy:
                    st.error("⚠️ Backend API is offline. Please start the FastAPI backend server first.")
                else:
                    # Prepare file for request
                    file_bytes = uploaded_file.getvalue()
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "image/jpeg")}
                    
                    start_time = time.time()
                    with st.spinner("Analyzing image features via FastAPI backend..."):
                        try:
                            response = requests.post(
                                f"{st.session_state.backend_url.rstrip('/')}/predict",
                                files=files,
                                timeout=REQUEST_TIMEOUT
                            )
                            elapsed_ms = round((time.time() - start_time) * 1000, 2)
                            
                            if response.status_code == 200:
                                res = response.json()
                                pred_class = res["prediction"]
                                confidence = res["confidence"]
                                probs = res["probabilities"]

                                # Display Result Banner
                                if pred_class == "Dog":
                                    st.markdown(f"""
                                    <div class="prediction-card-dog">
                                        <div class="prediction-title">🐶 Dog</div>
                                        <div class="prediction-confidence">Confidence: {confidence:.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="prediction-card-cat">
                                        <div class="prediction-title">🐱 Cat</div>
                                        <div class="prediction-confidence">Confidence: {confidence:.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Confidence Bar
                                st.markdown("#### Confidence Score")
                                st.progress(confidence / 100.0)

                                # Probabilities Breakdown
                                st.markdown("#### Probability Breakdown")
                                col_cat, col_dog = st.columns(2)
                                with col_cat:
                                    cat_prob = probs.get("Cat", 0.0)
                                    st.metric(label="Cat Probability", value=f"{cat_prob * 100:.2f}%")
                                    st.progress(cat_prob)
                                with col_dog:
                                    dog_prob = probs.get("Dog", 0.0)
                                    st.metric(label="Dog Probability", value=f"{dog_prob * 100:.2f}%")
                                    st.progress(dog_prob)

                                # Metric timing badge
                                st.caption(f"⚡ Inference + Network round-trip: `{elapsed_ms} ms`")

                                # Add record to history
                                st.session_state.history.append({
                                    "timestamp": time.strftime("%H:%M:%S"),
                                    "filename": uploaded_file.name,
                                    "prediction": pred_class,
                                    "confidence": confidence,
                                    "probabilities": probs,
                                    "latency_ms": elapsed_ms
                                })

                                # JSON Download Option
                                st.markdown("---")
                                json_data = json.dumps(res, indent=2)
                                st.download_button(
                                    label="📥 Download JSON Result",
                                    data=json_data,
                                    file_name=f"prediction_{uploaded_file.name}.json",
                                    mime="application/json"
                                )

                            else:
                                try:
                                    err_detail = response.json().get("detail", "Unknown server error")
                                except Exception:
                                    err_detail = response.text
                                st.error(f"Backend Error (HTTP {response.status_code}): {err_detail}")

                        except requests.exceptions.Timeout:
                            st.error(f"⏱️ Request timed out after {REQUEST_TIMEOUT} seconds.")
                        except requests.exceptions.ConnectionError:
                            st.error("🔌 Could not connect to FastAPI server. Please check your backend URL and status.")
                        except Exception as ex:
                            st.error(f"An unexpected error occurred: {str(ex)}")

        else:
            st.info("Upload an image on the left and click **Classify Image** to view predictions.")

# ==========================================
# TAB 2: BATCH PREDICTION
# ==========================================
with tab_batch:
    st.subheader("📂 Batch Image Classification")
    st.write("Upload multiple cat and dog images to perform batch classification in a single request.")

    batch_files = st.file_uploader(
        "Upload multiple images...",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        key="batch_file_uploader"
    )

    if batch_files:
        st.write(f"Selected **{len(batch_files)}** files.")
        if st.button("🚀 Process Batch", type="primary"):
            if not is_healthy:
                st.error("⚠️ Backend API is offline.")
            else:
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type or "image/jpeg")) for f in batch_files
                ]
                with st.spinner(f"Running batch prediction for {len(batch_files)} images..."):
                    try:
                        res = requests.post(
                            f"{st.session_state.backend_url.rstrip('/')}/predict/batch",
                            files=files_payload,
                            timeout=REQUEST_TIMEOUT * 2
                        )
                        if res.status_code == 200:
                            data = res.json()
                            results = data.get("results", [])

                            st.success(f"Batch completed! Processed {len(results)} images.")
                            
                            # Build table data
                            table_rows = []
                            for r in results:
                                if r["status"] == "success":
                                    table_rows.append({
                                        "Filename": r["filename"],
                                        "Status": "✅ Success",
                                        "Prediction": r["prediction"],
                                        "Confidence (%)": f"{r['confidence']:.2f}%",
                                        "Cat Prob": f"{r['probabilities']['Cat'] * 100:.2f}%",
                                        "Dog Prob": f"{r['probabilities']['Dog'] * 100:.2f}%"
                                    })
                                else:
                                    table_rows.append({
                                        "Filename": r["filename"],
                                        "Status": "❌ Error",
                                        "Prediction": "N/A",
                                        "Confidence (%)": "N/A",
                                        "Cat Prob": "N/A",
                                        "Dog Prob": "N/A"
                                    })

                            st.dataframe(table_rows, use_container_width=True)

                            # Batch JSON download
                            st.download_button(
                                label="📥 Download Batch Results JSON",
                                data=json.dumps(data, indent=2),
                                file_name="batch_predictions.json",
                                mime="application/json"
                            )
                        else:
                            st.error(f"Batch request failed (HTTP {res.status_code}): {res.text}")
                    except Exception as err:
                        st.error(f"Batch processing error: {err}")

# ==========================================
# TAB 3: SESSION HISTORY
# ==========================================
with tab_history:
    st.subheader("📜 Session Prediction History")
    if st.session_state.history:
        col_hist_head, col_hist_clear = st.columns([3, 1])
        with col_hist_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No prediction history recorded in this session yet.")
