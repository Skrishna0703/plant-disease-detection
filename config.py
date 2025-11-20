"""
Configuration file for Plant Disease Detection System
"""

# Model Configuration
MODEL_CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "zoom_range": 0.15,
    "shear_range": 0.15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": "nearest",
}

# File Paths
PATHS = {
    "model": "models/plant_disease.h5",
    "history": "models/history.json",
    "labels": "labels.json",
    "demo_image": "assets/demo/sample_leaf.jpg",
    "train_data": "data/train",
    "val_data": "data/val",
}

# Model Callbacks Configuration
CALLBACKS_CONFIG = {
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True,
    },
    "model_checkpoint": {
        "monitor": "val_accuracy",
        "save_best_only": True,
    },
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Plant Disease Detection",
    "page_icon": "🍃",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Prediction Configuration
PREDICTION_CONFIG = {
    "top_k": 3,
    "confidence_threshold": 0.3,  # Minimum confidence to display
}

# File Upload Configuration
FILE_CONFIG = {
    "allowed_extensions": ["jpg", "jpeg", "png", "bmp", "gif"],
    "max_file_size_mb": 10,
    "required_image_size": (224, 224),
}
