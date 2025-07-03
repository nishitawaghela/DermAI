import streamlit as st
import tensorflow as tf
from utils import preprocess_image, predict_image, generate_gradcam
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="DermAI", layout="centered")
st.title("🧠 DermAI - Skin Disease Classifier")
st.write("Upload a skin lesion image to predict the disease and visualize the Grad-CAM heatmap.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    model = tf.keras.models.load_model("best_model.h5")
    processed_image = preprocess_image(image)
    pred_class, pred_label = predict_image(model, processed_image)

    st.success(f"Prediction: **{pred_label}**")

    heatmap, superimposed_img = generate_gradcam(model, processed_image, pred_class)
    st.image(superimposed_img, caption="Grad-CAM", use_column_width=True)
