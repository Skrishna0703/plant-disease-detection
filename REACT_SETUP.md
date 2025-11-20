# Plant Disease Detection - React + Streamlit UI

Beautiful modern UI for plant disease detection using React frontend and Python backend.

## Setup

### 1. Backend API (Python)

Install dependencies:
```bash
pip install fastapi uvicorn python-multipart pillow tensorflow numpy
```

### 2. Frontend (React)

Navigate to the frontend directory and install:
```bash
cd frontend
npm install
```

## Running the Application

### Terminal 1: Start the Python Backend API
```bash
python api.py
```
The API will run on `http://localhost:8000`

### Terminal 2: Start the React Frontend
```bash
cd frontend
npm start
```
The app will open at `http://localhost:3000`

### Terminal 3 (Optional): Keep Streamlit running
```bash
streamlit run app.py
```

## Architecture

- **Frontend**: React with Axios for API calls
- **Backend**: FastAPI with CORS support
- **ML Model**: TensorFlow/Keras with preprocessing pipeline
- **Styling**: Custom CSS with modern green theme

## Features

✅ Modern responsive UI with card-based layout
✅ Real-time image upload and preview
✅ AI-powered disease detection
✅ Top 3 predictions display
✅ Treatment recommendations
✅ Model performance metrics in sidebar
✅ Recent detection history
✅ Mobile-friendly responsive design
✅ Beautiful animations and transitions

## File Structure

```
plant-disease-detection/
├── api.py                    # FastAPI backend
├── app.py                    # Original Streamlit app
├── model_training.py         # Model training script
├── utils.py                  # Utility functions
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Modern styling
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
└── models/
    └── plant_disease.h5     # Trained model
```

## First Time Setup

1. **Ensure model is trained**:
   ```bash
   python model_training.py
   ```

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   cd frontend
   npm install
   ```

3. **Start the stack**:
   - Terminal 1: `python api.py`
   - Terminal 2: `cd frontend && npm start`
   - Open browser: `http://localhost:3000`

## Troubleshooting

- **Port 8000 already in use**: Change `api.py` port
- **Port 3000 already in use**: Set `PORT=3001 npm start`
- **CORS errors**: Check FastAPI middleware in `api.py`
- **Model not found**: Ensure `models/plant_disease.h5` exists

## Tech Stack

- **Frontend**: React 18, Axios
- **Backend**: FastAPI, Uvicorn
- **ML**: TensorFlow, Keras, NumPy, PIL
- **Styling**: Custom CSS with CSS Grid/Flexbox
- **Deployment**: Ready for Docker/production

Enjoy your beautiful plant disease detection app! 🍃
