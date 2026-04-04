import streamlit as st
from PIL import Image
from utils import preprocess_image, load_model_cached, predict_top_k, load_label_map, load_training_history
import numpy as np
import json
import os

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌾", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    * {margin: 0; padding: 0; box-sizing: border-box;}
    body {background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);}
    .header-title {color: #00AA22; text-align: center; font-size: 42px; font-weight: 700; letter-spacing: 2px; margin: 30px 0 20px 0;}
    .info-box {border-radius: 15px; padding: 25px; text-align: center; font-weight: 600; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,206,209,0.2); transition: transform 0.3s; display: flex; flex-direction: column; justify-content: center;}
    .info-box:hover {transform: translateY(-5px);}
    .info-box h3 {font-size: 18px; margin-bottom: 8px;}
    .info-box p {font-size: 13px; opacity: 0.9;}
    .cyan-box {background: linear-gradient(135deg, #00CED1 0%, #00B8A8 100%); color: white;}
    .green-box {background: linear-gradient(135deg, #00AA22 0%, #008C1A 100%); color: white;}
    .purple-box {background: linear-gradient(135deg, #9B59B6 0%, #7D3C98 100%); color: white;}
    .orange-box {background: linear-gradient(135deg, #FF9500 0%, #E67E22 100%); color: white; font-size: 15px; padding: 20px;}
    .upload-section {border: 2px dashed #00CED1; border-radius: 15px; padding: 30px; background: linear-gradient(135deg, rgba(0,206,209,0.05) 0%, rgba(0,206,209,0.02) 100%); margin: 20px 0;}
    .prediction-box {background: linear-gradient(135deg, rgba(0,206,209,0.1) 0%, rgba(0,206,209,0.05) 100%); border: 2px solid #00CED1; border-radius: 12px; padding: 25px; margin: 15px 0;}
    .feature-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0;}
    .feature-card {border-radius: 12px; padding: 20px; color: white; font-weight: 600; font-size: 14px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);}
    .card-red {background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);}
    .card-cyan {background: linear-gradient(135deg, #4ECDC4 0%, #6FE7DD 100%);}
    .card-yellow {background: linear-gradient(135deg, #FFE66D 0%, #FFF5A1 100%); color: #333;}
    .card-purple {background: linear-gradient(135deg, #9B59B6 0%, #C39BD3 100%);}
    </style>
""", unsafe_allow_html=True)

# Header with title
st.markdown("<div class='header-title'>🌾 Plant & Crop Intelligence</div>", unsafe_allow_html=True)

# Info boxes row
col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown("""
    <div class='info-box cyan-box'>
        <h3>🔍 Disease Detection</h3>
        <p>AI-powered plant disease detection from leaf images</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='info-box green-box'>
        <h3>38 Plant Types</h3>
        <p>Comprehensive coverage of diseases across multiple crops</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='info-box purple-box'>
        <h3>📊 System Accuracy</h3>
        <p>95.0%+ Accuracy | Real-time Processing</p>
    </div>
    """, unsafe_allow_html=True)

# Key Benefits
st.markdown("""
<div class='info-box orange-box'>
    <h3 style='margin-bottom: 5px;'>✨ Key Benefits</h3>
    <p>✓ Advanced AI Detection | ✓ Instant Results | ✓ Treatment Recommendations | ✓ Disease Prevention Tips</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Disease Detection Section
st.markdown("<h2 style='color: #00AA22; margin: 30px 0 20px 0;'>🔍 Upload Plant Image</h2>", unsafe_allow_html=True)
col_left, col_right = st.columns(2, gap="large")

with col_left:
    uploaded_file = st.file_uploader("Choose a plant leaf image", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed")
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

with col_right:
    st.markdown("<h3 style='color: #00AA22; margin-bottom: 20px;'>Prediction Results</h3>", unsafe_allow_html=True)
    if not uploaded_file:
        st.markdown("<div class='prediction-box' style='color: #00CED1; text-align: center;'>📤 Upload an image to analyze</div>", unsafe_allow_html=True)
    else:
        model = load_model_cached("models/plant_disease.h5")
        labels = load_label_map("labels.json")
        if model and labels:
            with st.spinner("🔄 Analyzing..."):
                pil_image = Image.open(uploaded_file)
                preprocessed = preprocess_image(pil_image)
                if preprocessed is not None:
                    results = predict_top_k(model, preprocessed, labels, k=3)
                    if results:
                        main = results[0]
                        st.markdown(f"<div class='prediction-box'><h4 style='color: #00CED1; margin-bottom: 10px;'>🦠 {main['display']}</h4></div>", unsafe_allow_html=True)
                        confidence = main['confidence']
                        st.progress(confidence / 100)
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.markdown(f"<p style='color: #00AA22; font-weight: bold; margin-top: 15px;'>💊 Treatment: {main['remedy']}</p>", unsafe_allow_html=True)

if uploaded_file:
    model = load_model_cached("models/plant_disease.h5")
    labels = load_label_map("labels.json")
    if model and labels:
        pil_image = Image.open(uploaded_file)
        preprocessed = preprocess_image(pil_image)
        if preprocessed is not None:
            results = predict_top_k(model, preprocessed, labels, k=3)
            if results:
                st.markdown("---")
                st.markdown("<h3 style='color: #00AA22;'>🎯 Top 3 Predictions</h3>", unsafe_allow_html=True)
                
                pred_cols = st.columns(3, gap="medium")
                colors = ["#FF6B6B", "#00CED1", "#FFD700"]
                
                for idx in range(min(3, len(results))):
                    with pred_cols[idx]:
                        color = colors[idx]
                        conf = results[idx]['confidence']
                        disease = results[idx]['display']
                        
                        st.markdown(f"""
                        <div style='border:3px solid {color};border-radius:12px;padding:20px;background-color:#0a0a2e;text-align:center;'>
                            <h4 style='color:{color};margin:0 0 8px 0;font-size:16px'>【{idx+1}】 {disease.upper()}</h4>
                            <p style='color:#888;margin:0 0 15px 0;font-size:12px'>Rank #{idx+1}</p>
                            <div style='width:100%;height:10px;background-color:#1a1a3e;border-radius:5px;overflow:hidden;margin-bottom:15px;'>
                                <div style='width:{conf}%;height:100%;background-color:{color};'></div>
                            </div>
                            <p style='color:{color};font-size:18px;font-weight:bold;margin:0'>{conf:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

st.markdown("---")

# Footer Section
history_data = load_training_history("models/history.json")
labels = load_label_map("labels.json")
num_classes = len(labels) if labels else 0

train_acc = round(history_data['accuracy'][-1] * 100, 2) if history_data and 'accuracy' in history_data else 0
val_acc = round(history_data['val_accuracy'][-1] * 100, 2) if history_data and 'val_accuracy' in history_data else 0

footer_html = f'<div style="border:3px solid #00CED1;border-radius:20px;padding:50px;background:linear-gradient(135deg,rgba(0,206,209,0.05) 0%,rgba(0,206,209,0.02) 100%);text-align:center;margin:30px 0;"><h2 style="color:#FFFFFF;margin:0 0 10px 0;font-size:28px;letter-spacing:2px;">🌾 ABOUT THIS APPLICATION</h2><p style="color:#FFFFFF;margin:10px 0 25px 0;font-size:16px;font-weight:600">Plant Disease Detection System</p><p style="color:#AAA;margin:0 0 30px 0;font-size:13px;line-height:1.8;">An intelligent AI-powered application that analyzes plant leaf images and detects diseases using advanced Deep Learning and Convolutional Neural Networks for maximum accuracy and treatment recommendations.</p><div style="border:2px solid #00CED1;border-radius:12px;padding:15px;margin:0 auto 25px;background-color:rgba(0,206,209,0.1)"><p style="color:#FFD700;font-size:14px;margin:0;font-weight:700">✨ {train_acc}% Training Accuracy | {val_acc}% Validation Accuracy | {num_classes} Disease Classes | Real-time Detection</p></div><p style="color:#AAA;margin:0 0 25px 0;font-size:12px;">🔧 Built with: TensorFlow • Keras • Python • Streamlit • scikit-learn • NumPy • Pandas</p><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 auto 25px"><div style="background:linear-gradient(135deg,#FF6B6B 0%,#FF8E8E 100%);border-radius:10px;padding:15px;color:white;font-weight:bold;font-size:12px;">🧠 CNN Model</div><div style="background:linear-gradient(135deg,#4ECDC4 0%,#6FE7DD 100%);border-radius:10px;padding:15px;color:white;font-weight:bold;font-size:12px;">📊 {num_classes} Classes</div><div style="background:linear-gradient(135deg,#FFE66D 0%,#FFF5A1 100%);border-radius:10px;padding:15px;color:#333;font-weight:bold;font-size:12px;">📸 224×224 Images</div><div style="background:linear-gradient(135deg,#9B59B6 0%,#C39BD3 100%);border-radius:10px;padding:15px;color:white;font-weight:bold;font-size:12px;">🎯 Accurate</div></div><p style="color:#666;font-size:11px;margin:0;">© 2026 Shrikrishnasutar.dev | All Rights Reserved</p></div>'

st.markdown(footer_html, unsafe_allow_html=True)

# About Section - fetch real model data
# st.markdown("---")

# history_data = load_training_history("models/history.json")
# labels = load_label_map("labels.json")
# num_classes = len(labels) if labels else 0

# train_acc = round(history_data['accuracy'][-1] * 100, 2) if history_data and 'accuracy' in history_data else 0
# val_acc = round(history_data['val_accuracy'][-1] * 100, 2) if history_data and 'val_accuracy' in history_data else 0
# train_loss = round(history_data['loss'][-1], 4) if history_data and 'loss' in history_data else 0
# val_loss = round(history_data['val_loss'][-1], 4) if history_data and 'val_loss' in history_data else 0

# footer_html = f'<div style="border:3px solid #00CED1;border-radius:20px;padding:50px 40px;background-color:#0a0a1a;text-align:center;margin:30px 0;width:95%"><h2 style="color:#FFFFFF;margin:0 0 10px 0;font-size:32px;letter-spacing:3px">🌾 ABOUT THIS APPLICATION</h2><p style="color:#FFFFFF;margin:10px 0 30px 0;font-size:18px;font-weight:600">Plant Disease Detection System</p><p style="color:#AAA;margin:0 0 40px 0;font-size:14px;line-height:1.8">An intelligent AI-powered application that analyzes plant leaf images and detects diseases using advanced Deep Learning and Convolutional Neural Networks for maximum accuracy and treatment recommendations.</p><div style="border:2px solid #00CED1;border-radius:12px;padding:20px;margin:0 auto 30px;background-color:#0a0a2e"><p style="color:#FFD700;font-size:16px;margin:0;font-weight:700">✨ {train_acc}% Training Accuracy | {val_acc}% Validation Accuracy | {num_classes} Disease Classes | Real-time Detection</p></div><p style="color:#AAA;margin:0 0 30px 0;font-size:13px">🔧 Built with: TensorFlow • Keras • Python • Streamlit • scikit-learn • NumPy • Pandas</p><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:0 auto 30px"><div style="background:linear-gradient(135deg,#FF6B6B 0%,#FF8E8E 100%);border-radius:10px;padding:15px;color:white;font-weight:bold;font-size:13px">🦠 CNN Model</div><div style="background:linear-gradient(135deg,#4ECDC4 0%,#6FE7DD 100%);border-radius:10px;padding:15px;color:white;font-weight:bold;font-size:13px">📊 38 Classes</div><div style="background:linear-gradient(135deg,#FFE66D 0%,#FFF5A1 100%);border-radius:10px;padding:15px;color:#333;font-weight:bold;font-size:13px">224×224 Images</div><div style="background:linear-gradient(135deg,#9B59B6 0%,#C39BD3 100%);border-radius:10px;padding:15px;color:white;font-weight:bold;font-size:13px">🎯 Accurate</div></div><p style="color:#666;font-size:11px;margin:20px 0 0 0">© 2026 Shrikrishnasutar.dev | All Rights Reserved</p></div>'

# st.markdown(footer_html, unsafe_allow_html=True)
