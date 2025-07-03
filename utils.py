import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import cv2
from PIL import Image

CLASS_NAMES = [
    "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions",
    "Dermatofibroma", "Melanocytic nevi", "Melanoma", "Vascular lesions"
]

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_image(model, processed_image):
    preds = model.predict(processed_image)
    pred_class = np.argmax(preds[0])
    pred_label = CLASS_NAMES[pred_class]
    return pred_class, pred_label

def generate_gradcam(model, img_array, pred_index):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-3).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    heatmap = (cam * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = img_array[0]
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(255 * img)
    img = cv2.resize(img, (224, 224))

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return heatmap, Image.fromarray(superimposed_img)
