import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Label mapping
class_indices = {
    0: 'Actinic keratoses (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic nevi (nv)',
    6: 'Vascular lesions (vasc)'
}

st.set_page_config(page_title="DermAI - Skin Disease Classifier", layout="centered")

st.title("DermAI: Skin Disease Classifier")
st.caption("Upload a skin lesion image and let AI classify the condition.")
uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_indices[predicted_class]
    confidence = float(np.max(predictions[0]))

    st.markdown(f"Prediction: `{predicted_label}`")
    st.markdown(f"Confidence: {confidence * 100:.2f}%")

    # Grad-CAM
    st.markdown("---")
    st.subheader("Visual Explanation with Grad-CAM")
    heatmap, class_id = make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv")
    cam_img = save_and_display_gradcam(image, heatmap)

    st.image(cam_img, caption="Grad-CAM", use_column_width=True)

