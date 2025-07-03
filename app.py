
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from utils import get_prediction_with_gradcam

st.set_page_config(page_title="DermAI", layout="centered")
st.title("🧴 DermAI: Skin Disease Classifier")

model = load_model("best_model.h5")
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence, gradcam_image = get_prediction_with_gradcam(img, model, class_labels)
        st.success(f"Predicted: **{label.upper()}** with {confidence*100:.2f}% confidence")
        st.image(gradcam_image, caption="Grad-CAM", use_column_width=True)
