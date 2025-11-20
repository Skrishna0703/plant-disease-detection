# Plant Disease Detection - Quick Start Guide

## ⚡ 5-Minute Quick Start

### 1. Install (1 minute)
```bash
pip install -r requirements.txt
```

### 2. Run the App (30 seconds)
```bash
streamlit run app.py
```

### 3. Test with Demo
- Click "Try Demo Image" in the app
- Get instant prediction!

That's it! 🎉

---

## 🎓 Full Training (if you have the dataset)

### 1. Get Dataset
- Download PlantVillage from Kaggle
- Extract to `data/train` and `data/val`

### 2. Train Model (varies by hardware)
```bash
python model_training.py
```

### 3. Run App
```bash
streamlit run app.py
```

---

## 📱 Usage

1. Open: `http://localhost:8501`
2. Upload leaf image (JPG/PNG)
3. Get prediction with confidence
4. View treatment recommendation
5. See top-3 alternative predictions

---

## 🔧 Common Commands

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Install dependencies |
| `python model_training.py` | Train the model |
| `streamlit run app.py` | Run web app |
| `python -m venv venv` | Create virtual environment |

---

## ✅ What Works Out of the Box

- ✅ Streamlit UI with file upload
- ✅ Image preprocessing
- ✅ Model loading and caching
- ✅ Prediction display
- ✅ Top-3 confidence chart
- ✅ Treatment recommendations
- ✅ Demo image button

## ⚙️ What Requires Setup

- ❌ Model file (train with `python model_training.py`)
- ❌ Dataset (download PlantVillage)

---

## 🆘 Quick Troubleshooting

**"Model not found"** → Train with `python model_training.py`

**"No data classes"** → Add folders to `data/train` and `data/val`

**"Slow inference"** → Normal on CPU; GPU recommended

**App won't start** → Check all dependencies installed

---

## 💡 Pro Tips

1. **Start Simple**: Test with demo image first
2. **Quality Matters**: Use clear, well-lit leaf images
3. **GPU Boost**: Install `tensorflow-gpu` for 10x faster training
4. **Organize Data**: Strict folder structure is critical

---

🌾 **Ready to detect plant diseases? Run the app now!**

```bash
streamlit run app.py
```
