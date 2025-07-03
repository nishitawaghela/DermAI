
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from PIL import Image

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_prediction_with_gradcam(img, model, class_labels):
    processed = preprocess_image(img)
    preds = model.predict(processed)
    class_index = np.argmax(preds[0])
    class_output = model.output[:, class_index]

    last_conv_layer = model.get_layer("top_conv")
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img_array = np.array(img.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap_img, 0.4, 0)

    return class_labels[class_index], preds[0][class_index], superimposed_img
