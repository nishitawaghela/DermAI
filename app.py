import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("DermAI - Skin Disease Classifier")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = tf.keras.models.load_model("best_model.h5")
    image_array = np.array(image) / 255.0
    prediction = model.predict(np.expand_dims(image_array, axis=0))[0]

    classes = ["Acne", "Eczema", "Psoriasis", "Rosacea"]
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")

