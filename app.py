import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import gdown

# ----------------------------
# Download model if not exists
# ----------------------------
model_path = "vgg16_insect_model.h5"
drive_file_id = "1c4TELIMh_AaunvT-oQnAov-OBxGJhHJT"
drive_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    gdown.download(drive_url, model_path, quiet=False)

# ----------------------------
# Load model and define classes
# ----------------------------
model = tf.keras.models.load_model(model_path)

class_names = [
    'aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper',
    'mites', 'mosquito', 'sawfly', 'stem_borer'
]

# ----------------------------
# Insect impact dictionary
# ----------------------------
insect_impact = {
    "aphids": "Aphids suck sap from crops like wheat, potatoes, and vegetables. They can cause stunted growth and transmit plant viruses.",
    "armyworm": "Armyworms are destructive pests of maize and rice. They feed in groups and can destroy crops quickly.",
    "beetle": "Certain beetles feed on crop leaves and reduce photosynthesis, affecting yield in vegetables and grains.",
    "bollworm": "Bollworms damage cotton, tomato, and chickpea by boring into fruits and bolls, ruining the harvest.",
    "grasshopper": "Grasshoppers consume plant leaves and stems. In swarms, they can devastate entire agricultural fields.",
    "mites": "Mites feed on plant juices, causing yellowing and death in vegetables, beans, and ornamental plants.",
    "mosquito": "While not a crop pest, mosquitoes breed in water around farms and pose health risks to workers.",
    "sawfly": "Sawfly larvae feed on the leaves of cereal crops like wheat and barley, reducing growth and yield.",
    "stem_borer": "Stem borers attack crops like rice, sugarcane, and maize by boring into the stems and causing wilting."
}

# ----------------------------
# Preprocess uploaded image
# ----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ----------------------------
# Streamlit App Interface
# ----------------------------
st.title("🪲 Insect Classifier for Agriculture")
st.write("Upload an image or use webcam to classify the insect. Useful for farmers and agricultural advisors.")

option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

predicted_class = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            prediction = model.predict(preprocess_image(image))
            class_index = np.argmax(prediction)
            predicted_class = class_names[class_index]
            st.success(f"✅ Predicted Class: **{predicted_class}**")

elif option == "Use Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert('RGB')
        st.image(image, caption="Captured Image", use_column_width=True)
        if st.button("Predict"):
            prediction = model.predict(preprocess_image(image))
            class_index = np.argmax(prediction)
            predicted_class = class_names[class_index]
            st.success(f"✅ Predicted Class: **{predicted_class}**")

# ----------------------------
# Show Impact if Prediction Done
# ----------------------------
if predicted_class:
    if st.button("Show Agricultural Impact"):
        impact = insect_impact.get(predicted_class, "Impact information not available.")
        st.info(f"🧠 **Impact of {predicted_class.capitalize()}:** {impact}")
