import time
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import tensorflow as tf
from backend.config import settings

logger = logging.getLogger("cats_vs_dogs.predictor")

class ModelPredictor:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[tf.keras.Model] = None
        self._is_loaded: bool = False

    def load(self) -> None:
        """
        Loads TensorFlow model into memory once during API startup.
        Checks candidate paths and automatically resolves Git LFS pointer files on cloud hosts.
        """
        if self._is_loaded and self.model is not None:
            logger.info("Model is already loaded.")
            return

        import os
        import requests

        target_paths = [
            self.model_path,
            os.path.join(os.path.dirname(__file__), "model.keras"),
            os.path.join(os.path.dirname(__file__), "..", "model.keras"),
            os.path.join(os.path.dirname(__file__), "..", "backend", "model.keras")
        ]

        resolved_path = None
        for p in target_paths:
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                resolved_path = p
                break

        if not resolved_path:
            # Check if an existing path is a Git LFS pointer file (< 1000 bytes)
            lfs_pointer_path = None
            for p in target_paths:
                if os.path.exists(p):
                    lfs_pointer_path = p
                    break
            
            resolved_path = lfs_pointer_path or os.path.join(os.path.dirname(__file__), "model.keras")
            
            # Download full model file if pointer or missing on cloud host
            lfs_url = "https://media.githubusercontent.com/media/luckyKumar37/Deep_learning_project/main/cats_vs_dogs/backend/model.keras"
            logger.warning(f"Git LFS pointer file or missing model detected at '{resolved_path}'. Downloading full 81.9MB model.keras from GitHub LFS...")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
                resp = requests.get(lfs_url, stream=True, timeout=120)
                if resp.status_code == 200:
                    with open(resolved_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    logger.info(f"Model successfully downloaded ({os.path.getsize(resolved_path)} bytes) to '{resolved_path}'.")
                else:
                    logger.error(f"Failed to download model from GitHub LFS. Status code: {resp.status_code}")
            except Exception as dl_err:
                logger.error(f"Exception during Git LFS model download: {dl_err}")

        logger.info(f"Loading Keras model from resolved path: {resolved_path} (size: {os.path.getsize(resolved_path) if os.path.exists(resolved_path) else 'N/A'} bytes)")
        start_time = time.time()
        try:
            self.model = tf.keras.models.load_model(resolved_path)
            self._is_loaded = True
            elapsed = time.time() - start_time
            logger.info(f"Model successfully loaded in {elapsed:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load model from {resolved_path}: {str(e)}")
            raise e

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def predict(self, input_array: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on preprocessed NumPy batch array.
        Returns:
            - predicted_class (str): "Cat" or "Dog"
            - confidence (float): percentage between 0 and 100
            - probabilities (Dict[str, float]): probability per class sum up to 1.0
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load() before predicting.")

        logger.debug(f"Running inference for shape: {input_array.shape}")
        
        # Raw model output shape (1, 1) or (1, 2)
        raw_preds = self.model.predict(input_array, verbose=0)
        
        activation_mode = settings.MODEL_ACTIVATION.lower()
        class_names = settings.CLASS_NAMES # Default ["Cat", "Dog"]

        if activation_mode == "softmax" or raw_preds.shape[-1] == 2:
            # Softmax output shape (1, 2)
            probs = raw_preds[0] # array of shape (2,)
            cat_prob = float(probs[0])
            dog_prob = float(probs[1])
            
            top_idx = int(np.argmax(probs))
            predicted_class = class_names[top_idx] if top_idx < len(class_names) else f"Class_{top_idx}"
            confidence = round(float(probs[top_idx]) * 100.0, 2)
            
            probabilities = {
                "Cat": round(cat_prob, 4),
                "Dog": round(dog_prob, 4)
            }
            
        else:
            # Sigmoid output shape (1, 1) or single scalar
            sig_val = float(raw_preds[0][0])
            
            dog_prob = sig_val
            cat_prob = 1.0 - sig_val
            
            if dog_prob > settings.CONFIDENCE_THRESHOLD:
                predicted_class = "Dog"
                confidence = round(dog_prob * 100.0, 2)
            else:
                predicted_class = "Cat"
                confidence = round(cat_prob * 100.0, 2)
                
            probabilities = {
                "Cat": round(cat_prob, 4),
                "Dog": round(dog_prob, 4)
            }

        logger.info(
            f"Prediction completed: predicted_class='{predicted_class}', "
            f"confidence={confidence}%, probabilities={probabilities}"
        )
        return predicted_class, confidence, probabilities

# Global predictor instance
predictor = ModelPredictor()
