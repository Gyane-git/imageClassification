import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os


# Define model file name and Google Drive file ID
model_path = "vgg16_insect_model.h5"
drive_file_id = "1c4TELIMh_AaunvT-oQnAov-OBxGJhHJT"
drive_url = f"https://drive.google.com/uc?id={drive_file_id}"

# Download if model doesn't exist locally
if not os.path.exists(model_path):
    gdown.download(drive_url, model_path, quiet=False)


# Load the trained model
model = tf.keras.models.load_model("vgg16_insect_model.h5")

# Define class names
class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Web App UI
st.title("🪲 Insect Classifier")
st.write("Upload an image or capture from webcam to classify the insect.")

option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            prediction = model.predict(preprocess_image(image))
            class_index = np.argmax(prediction)
            st.success(f"Predicted Class: **{class_names[class_index]}**")

elif option == "Use Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert('RGB')
        st.image(image, caption="Captured Image", use_column_width=True)
        if st.button("Predict"):
            prediction = model.predict(preprocess_image(image))
            class_index = np.argmax(prediction)
            st.success(f"Predicted Class: **{class_names[class_index]}**")
