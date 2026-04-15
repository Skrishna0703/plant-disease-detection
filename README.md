# Plant Disease Detection System

A machine learning-powered web application that detects and identifies plant diseases from image inputs. This system uses deep learning models to classify crop diseases and provides predictions with confidence scores.

## Overview

The Plant Disease Detection System combines a backend API with a frontend interface to enable farmers and agricultural professionals to quickly identify plant diseases. The system supports multiple crop types and disease classifications.

## Features

- **Disease Classification**: Automated detection of plant diseases using CNN models
- **Multi-Crop Support**: Trained on multiple crop types including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, and Raspberry
- **Web API**: RESTful API for disease prediction and model management
- **Interactive Frontend**: User-friendly web interface for uploading and analyzing plant images
- **Real-time Predictions**: Fast inference with confidence scores
- **Model Management**: Tools for model training, validation, and evaluation

## Project Structure

```
plant-disease-detection/
├── app.py                    # Flask application entry point
├── api.py                    # API routes and endpoints
├── config.py                 # Configuration settings
├── model_training.py         # Model training pipeline
├── utils.py                  # Utility functions
├── setup.py                  # Project setup configuration
├── requirements.txt          # Python dependencies
├── labels.json              # Disease labels mapping
├── QUICK_REFERENCE.txt      # Quick start guide
├── models/
│   ├── plant_disease.h5     # Pre-trained model
│   └── history.json         # Training history
├── data/
│   ├── train/               # Training dataset (organized by crop/disease)
│   ├── val/                 # Validation dataset
│   ├── train_split/         # Split training data
│   └── val_split/           # Split validation data
├── frontend/                # React/TypeScript frontend
│   ├── src/
│   ├── public/
│   └── package.json
├── assets/                  # Media and resources
│   └── demo/
└── api.py                   # API endpoints
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+ (for frontend)
- pip (Python package manager)
- npm or yarn (for frontend dependencies)

### Backend Setup

1. **Navigate to the project directory:**
   ```bash
   cd plant-disease-detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application:**
   - Review and update `config.py` as needed
   - Ensure the model file (`models/plant_disease.h5`) is present

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

## Usage

### Running the Backend API

```bash
python app.py
```

The API will start on `http://localhost:5000` (or configured port).

### Running the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000` (or configured port).

### API Endpoints

#### Predict Disease
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: FormData with image file
- **Response**: 
  ```json
  {
    "disease": "Disease Name",
    "confidence": 0.95,
    "crop": "Crop Type"
  }
  ```

#### Get Model Status
- **URL**: `/status`
- **Method**: `GET`
- **Response**: Model availability and version information

## Dataset

The training dataset includes:
- **Crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry
- **Classes**: Healthy and disease-affected samples for each crop
- **Organization**: Structured by crop type and disease classification
- **Location**: `data/train/` and `data/val/` directories

### Supported Diseases

See `labels.json` for the complete list of supported crop types and disease classifications.

## Model Training

To train or retrain the model:

```bash
python model_training.py
```

This will:
- Load and preprocess training data from `data/train_split/`
- Train the CNN model on the dataset
- Validate against `data/val_split/`
- Save the trained model to `models/plant_disease.h5`
- Store training history in `models/history.json`

### Training Configuration

Edit `config.py` to modify:
- Learning rate
- Batch size
- Number of epochs
- Image dimensions
- Model architecture parameters

## Configuration

See `config.py` for configuration options:
- Model paths
- API port and host
- Dataset paths
- Training parameters
- Inference settings

## Requirements

Key dependencies:
- TensorFlow/Keras - Deep learning framework
- Flask - Web framework for API
- Pillow - Image processing
- NumPy - Numerical computing
- React/TypeScript - Frontend framework
- Axios - HTTP client

See `requirements.txt` for complete list.

## Quick Reference

For quick setup and testing, see `QUICK_REFERENCE.txt`.

## Performance Metrics

Training metrics and historical performance data are stored in `models/history.json`.

## Troubleshooting

### Model Loading Issues
- Verify `models/plant_disease.h5` exists
- Check TensorFlow/Keras versions compatibility
- Ensure sufficient disk space and memory

### API Connection Issues
- Verify backend is running on correct port
- Check CORS settings if frontend can't reach API
- Review API logs for detailed error messages

### Frontend Issues
- Clear browser cache
- Ensure Node.js dependencies are installed
- Check frontend configuration for API URL

## Contributing

Contributions are welcome! To contribute:
1. Create a new branch for features/fixes
2. Test changes thoroughly
3. Submit a pull request with detailed description

## License

[Specify your license here]

## Support

For issues or questions:
- Check `QUICK_REFERENCE.txt` for common solutions
- Review log files for error details
- Contact the development team

## Future Enhancements

- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Advanced visualization and heatmaps
- [ ] Batch processing capability
- [ ] Model versioning and A/B testing
- [ ] Cloud deployment ready

---

**Last Updated**: April 2026
