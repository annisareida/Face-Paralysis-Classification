import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_modelYFPMEIIBINERaug1000.h5')

model = load_model()

# Kelas berdasarkan skala eFace
class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 'Severe']

# -----------------------------
# Fungsi prediksi
# -----------------------------
def predict_image(img_pil):
    img = img_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    class_name = class_names[predicted_index]

    return class_name

# -----------------------------
# UI Streamlit
# -----------------------------
st.title("Face Paralysis Detection")
st.write("Upload an image or use your webcam to classify face according to **eFace Scale**: "
         "**Complete, Mild, Moderate, Near Normal, Normal, Severe**.")

# Pilihan sumber input
input_type = st.radio("Select input method:", ['Upload Image', 'Use Webcam'])

image_source = None
if input_type == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_source = Image.open(uploaded_file).convert('RGB')
elif input_type == 'Use Webcam':
    webcam_image = st.camera_input("Take a photo")
    if webcam_image:
        image_source = Image.open(webcam_image).convert('RGB')

# -----------------------------
# Tampilkan gambar & hasil prediksi
# -----------------------------
if image_source:
    st.image(image_source, caption="Input Image", use_container_width=True)

    if st.button("Predict"):
        label = predict_image(image_source)
        st.success(f"Predicted Class: **{label}**")
