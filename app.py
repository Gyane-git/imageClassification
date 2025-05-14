import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Page config
st.set_page_config(layout="wide", page_title="Insect Classifier for Agriculture", page_icon="🪲")

# Download model if not exists
model_path = "vgg16_insect_model.h5"
drive_file_id = "1c4TELIMh_AaunvT-oQnAov-OBxGJhHJT"
drive_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    gdown.download(drive_url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Class labels
class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']

# Insect impact dictionary
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

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Sidebar input selection
st.sidebar.title("🪲 Input Method")
input_option = st.sidebar.radio("Select Image Source", ["Upload Image", "Capture with Webcam"])

# Main header
st.title("🧠 Insect Classifier for Agriculture")
st.subheader("Upload or capture an image to classify insect pests and learn their impact on crops.")

# Layout
col1, col2 = st.columns([1, 2])
image = None

# Handle image input
with col1:
    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("📁 Upload an insect image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    elif input_option == "Capture with Webcam":
        captured = st.camera_input("📷 Capture from Webcam")
        if captured:
            image = Image.open(captured).convert('RGB')

with col2:
    if image:
        st.image(image, caption="Selected Image", use_column_width=True)

# Prediction
if image:
    if st.button("🔍 Predict Insect"):
        prediction = model.predict(preprocess_image(image))
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]
        st.session_state['predicted_class'] = predicted_class
        st.success(f"✅ Predicted Class: {predicted_class.capitalize()}")

# Retrieve prediction from session
predicted_class = st.session_state.get('predicted_class', None)

# Show impact info
if predicted_class:
    if st.button("🌿 Show Agricultural Impact"):
        impact = insect_impact.get(predicted_class, "Impact information not available.")
        st.markdown(f"### 🧠 Impact of {predicted_class.capitalize()}")
        st.markdown(f"{impact}")
