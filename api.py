"""
FastAPI backend for plant disease detection
Serves predictions to React frontend
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
from pathlib import Path

from utils import (
    preprocess_image,
    load_model_cached,
    predict_top_k,
    load_label_map,
    load_training_history,
)

app = FastAPI(title="Plant Disease Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache model and labels on startup
model = None
labels = None
history = None


@app.on_event("startup")
async def startup_event():
    """Load model and labels on startup."""
    global model, labels, history
    model = load_model_cached("models/plant_disease.h5")
    labels = load_label_map("labels.json")
    history = load_training_history("models/history.json")


@app.get("/api/model-info")
async def get_model_info():
    """Get model training information."""
    if not history:
        return {"error": "Model history not found"}
    
    return {
        "train_acc": f"{history['accuracy'][-1] * 100:.1f}",
        "val_acc": f"{history['val_accuracy'][-1] * 100:.1f}",
        "epochs": len(history['accuracy']),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Make prediction on uploaded image."""
    if not model or not labels:
        return JSONResponse(
            status_code=500,
            content={"error": "Model or labels not found"}
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        preprocessed = preprocess_image(image)
        
        if preprocessed is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Image preprocessing failed"}
            )
        
        # Get predictions
        results = predict_top_k(model, preprocessed, labels, k=3)
        
        if not results:
            return JSONResponse(
                status_code=500,
                content={"error": "Prediction failed"}
            )
        
        return {
            "main_result": {
                "display": results[0]["display"],
                "confidence": results[0]["confidence"],
                "remedy": results[0]["remedy"],
            },
            "top_predictions": [
                {
                    "display": r["display"],
                    "confidence": r["confidence"],
                }
                for r in results
            ],
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction error: {str(e)}"}
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
