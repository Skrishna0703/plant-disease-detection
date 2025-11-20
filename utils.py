"""
Utility functions for plant disease detection system.
Handles image preprocessing, model loading, and prediction.
"""

import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from pathlib import Path


def preprocess_image(image_input, img_size=(224, 224)):
    """
    Preprocess image for model prediction.
    
    Args:
        image_input: PIL Image or file path
        img_size: Target image size (default: 224x224 for MobileNetV2)
    
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Load image if path is provided
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        
        # Resize to model input size
        img = img.resize(img_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] range
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


@st.cache_resource
def load_model_cached(model_path="models/plant_disease.h5"):
    """
    Load pre-trained model with caching for performance.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras/TensorFlow model
    """
    try:
        if not Path(model_path).exists():
            st.error(f"Model not found at {model_path}")
            st.info("Please train the model first using: python model_training.py")
            return None
        
        model = tf.keras.models.load_model(model_path)
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def predict_top_k(model, image_array, labels, k=3):
    """
    Get top-k predictions from the model.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        labels: Dictionary mapping class indices to label names
        k: Number of top predictions to return (default: 3)
    
    Returns:
        List of tuples: (disease_name, confidence, remedy)
    """
    try:
        if model is None or image_array is None:
            return None
        
        # Get predictions
        predictions = model.predict(image_array, verbose=0)
        
        # Get top-k indices and confidences
        top_k_indices = np.argsort(predictions[0])[-k:][::-1]
        top_k_confidences = predictions[0][top_k_indices]
        
        results = []
        for idx, confidence in zip(top_k_indices, top_k_confidences):
            # Convert to class label
            class_name = list(labels.keys())[idx] if idx < len(labels) else f"Class_{idx}"
            
            # Get disease info
            if class_name in labels:
                disease_info = labels[class_name]
                display_name = disease_info.get("display", class_name)
                remedy = disease_info.get("remedy", "No information available.")
            else:
                display_name = class_name.replace('_', ' ')
                remedy = "No information available."
            
            results.append({
                "class": class_name,
                "display": display_name,
                "confidence": float(confidence) * 100,
                "remedy": remedy
            })
        
        return results
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


def load_label_map(label_path="labels.json"):
    """
    Load label mapping from JSON file.
    
    Args:
        label_path: Path to labels.json file
    
    Returns:
        Dictionary with class labels and remedies
    """
    try:
        if not Path(label_path).exists():
            st.warning(f"Labels file not found at {label_path}")
            return {}
        
        with open(label_path, 'r') as f:
            labels = json.load(f)
        
        return labels
    
    except json.JSONDecodeError:
        st.error("Error reading labels.json - invalid JSON format")
        return {}
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return {}


def load_training_history(history_path="models/history.json"):
    """
    Load training history for displaying model metrics.
    
    Args:
        history_path: Path to history.json file
    
    Returns:
        Dictionary with training history
    """
    try:
        if not Path(history_path).exists():
            return None
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        return history
    
    except Exception as e:
        st.warning(f"Could not load training history: {str(e)}")
        return None


def validate_image_file(uploaded_file):
    """
    Validate uploaded image file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        True if valid, False otherwise
    """
    if uploaded_file is None:
        return False
    
    valid_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in valid_extensions:
        st.error(f"Invalid file type. Accepted formats: {', '.join(valid_extensions)}")
        return False
    
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        st.error("File size exceeds 10MB limit")
        return False
    
    return True


def get_dataset_info(train_dir="data/train", val_dir="data/val"):
    """
    Get information about the dataset.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
    
    Returns:
        Dictionary with dataset statistics
    """
    try:
        train_path = Path(train_dir)
        val_path = Path(val_dir)
        
        train_classes = len([d for d in train_path.iterdir() if d.is_dir()])
        val_classes = len([d for d in val_path.iterdir() if d.is_dir()])
        
        train_samples = sum(1 for _ in train_path.rglob('*') if _.is_file())
        val_samples = sum(1 for _ in val_path.rglob('*') if _.is_file())
        
        return {
            "train_classes": train_classes,
            "val_classes": val_classes,
            "train_samples": train_samples,
            "val_samples": val_samples
        }
    
    except Exception as e:
        st.warning(f"Could not load dataset info: {str(e)}")
        return None
