import streamlit as st
from PIL import Image
from utils import preprocess_image, load_model_cached, predict_top_k, load_label_map, load_training_history

st.set_page_config(page_title="Plant Disease Detection", page_icon="", layout="wide")

st.markdown("<h1 style='color: #16a34a; text-align: center;'> Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; text-align: center;'>AI-Powered Disease Detection</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='color: #16a34a;'> Model Info</h2>", unsafe_allow_html=True)
    history = load_training_history("models/history.json")
    if history:
        col1, col2 = st.columns(2)
        with col1:
            acc = history['accuracy'][-1] * 100
            st.metric("Train Acc", f"{acc:.1f}%")
        with col2:
            val_acc = history['val_accuracy'][-1] * 100
            st.metric("Val Acc", f"{val_acc:.1f}%")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #16a34a;'>📊 Dataset Info</h3>", unsafe_allow_html=True)
    st.write("**Classes:** 38 plant diseases")
    st.write("**Training samples:** ~1900 images")
    st.write("**Train/Val Split:** 80/20")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #16a34a;'>🧠 Model Architecture</h3>", unsafe_allow_html=True)
    st.write("**Type:** Convolutional Neural Network (CNN)")
    st.write("**Framework:** TensorFlow/Keras")
    st.write("**Input:** 224×224 RGB images")
    st.write("**Output:** 38 disease classes")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #16a34a;'>🎯 How to Use</h3>", unsafe_allow_html=True)
    st.write("""
    1. Upload a clear image of a plant leaf
    2. The model will analyze it
    3. View the predicted disease and confidence
    4. See treatment recommendations
    5. Check top 3 predictions
    """)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader(" Upload Image")
    uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed")
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Uploaded Image")

with col_right:
    st.subheader(" Results")
    if not uploaded_file:
        st.info("Upload an image to analyze")
    else:
        model = load_model_cached("models/plant_disease.h5")
        labels = load_label_map("labels.json")
        if model and labels:
            with st.spinner("Analyzing..."):
                pil_image = Image.open(uploaded_file)
                preprocessed = preprocess_image(pil_image)
                if preprocessed is not None:
                    results = predict_top_k(model, preprocessed, labels, k=3)
                    if results:
                        main = results[0]
                        st.write(f"**Disease:** {main['display']}")
                        st.write(f"**Confidence:** {main['confidence']:.1f}%")
                        st.progress(main['confidence'] / 100)
                        st.write(f"**Treatment:** {main['remedy']}")

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
                st.subheader(" Top 3 Predictions")
                cols = st.columns(3)
                for idx, col in enumerate(cols):
                    if idx < len(results):
                        with col:
                            st.write(f"**{idx+1}. {results[idx]['display']}**")
                            st.write(f"{results[idx]['confidence']:.1f}%")

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <h3 style='color: #16a34a;'>About This Application</h3>
    <p>🌾 <strong>Plant Disease Detection System</strong></p>
    <p>An AI-powered application using Deep Learning and Machine Learning to detect plant diseases from leaf images.</p>
    <br>
    <p><strong>Model Details:</strong></p>
    <p>CNN Architecture | 38 Disease Classes | Real-time Detection</p>
    <br>
    <p style='font-size: 0.9rem; color: #888;'>
        <strong>Created with Deep Learning & Machine Learning</strong><br>
        Powered by TensorFlow • Keras • Streamlit<br>
        Dataset: PlantVillage<br>
        <br>
        © 2025 Shrikrishnasutar.dev | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
