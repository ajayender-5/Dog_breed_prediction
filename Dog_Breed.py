import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Set page configuration to full width
st.set_page_config(page_title="Dog Breed Prediction", page_icon="🐶")

# Load the pretrained VGG16 model without the top layer for feature extraction
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load the trained ANN model
ann_model = tf.keras.models.load_model(r"C:\Users\madas\Data Science 255 - Batch\Deep_Learning\Tasks\ann_model.h5")

# Load the label encoder
with open(r"C:\Users\madas\Data Science 255 - Batch\Deep_Learning\Tasks\label_encoder.pkl", 'rb') as file:
    label_encoder = pickle.load(file)

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    return preprocess_input(img_array)

def extract_features(img, model):
    img_array = preprocess_image(img)
    features = model.predict(img_array)
    return features.flatten()

def predict_new_image(img, feature_extractor_model, ann_model, label_encoder):
    # Extract features from the new image
    features = extract_features(img, feature_extractor_model)
    features = np.expand_dims(features, axis=0)  # Adjust shape for prediction

    # Predict using the ANN model
    predictions = ann_model.predict(features)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(predicted_class_index)
    return predicted_label[0]

# Streamlit app
st.title(":rainbow[Dog Breed Prediction]")
st.write("Upload an image of a dog and get the predicted breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Predict the breed of the dog
    predicted_label = predict_new_image(image, model, ann_model, label_encoder)
    
    # Display prediction
    st.write(f"Predicted Breed: {predicted_label}")

    # Visualize features
    st.write("Feature Maps from Convolutional Layers:")
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    img_array = preprocess_image(image)
    activations = activation_model.predict(img_array)
    
    for layer_name, layer_activation in zip([layer.name for layer in model.layers if 'conv' in layer.name], activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // 16
        display_grid = np.zeros((size * n_cols, 16 * size))
        for col in range(n_cols):
            for row in range(16):
                channel_image = layer_activation[0, :, :, col * 16 + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1.0 / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
