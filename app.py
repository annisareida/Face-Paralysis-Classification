import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN  # untuk deteksi wajah

# -----------------------------
# Load model
# -----------------------------

def load_model():
    return tf.keras.models.load_model('vggface_model_20251028_144939.h5')

model = load_model()
detector = MTCNN()  # inisialisasi detektor wajah

# Kelas berdasarkan skala eFace
class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 'Severe']

# -----------------------------
# Fungsi deteksi dan crop wajah
# -----------------------------
def crop_face(image_pil, margin=40):
    """
    Deteksi wajah dengan MTCNN dan crop area wajah dengan margin.
    Jika tidak ditemukan wajah, mengembalikan gambar asli.
    """
    img_array = np.array(image_pil)
    detections = detector.detect_faces(img_array)

    if len(detections) == 0:
        st.warning("No face detected — using full image.")
        return image_pil

    # Ambil wajah pertama yang terdeteksi
    x, y, w, h = detections[0]['box']

    # Tambahkan margin agar crop tidak terlalu ketat
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_array.shape[1], x + w + margin)
    y2 = min(img_array.shape[0], y + h + margin)

    cropped_face = img_array[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped_face)
    return cropped_pil

# -----------------------------
# Normalisasi gambar
# -----------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

def preprocess_image(img_pil):
    img = img_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = normalization_layer(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Fungsi prediksi
# -----------------------------
def predict_image(img_pil):
    img_array = preprocess_image(img_pil)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    return class_names[predicted_index]
# -----------------------------
# UI Streamlit
# -----------------------------
st.title("Face Paralysis Detection (with Face Cropping)")
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
    st.image(image_source, caption="Original Input", use_container_width=True)

    if st.button("Predict"):
        # Crop wajah sebelum klasifikasi
        cropped_face = crop_face(image_source)

        st.image(cropped_face, caption="Detected Face", use_container_width=True)

        # Prediksi kelas wajah hasil crop
        label = predict_image(cropped_face)
        st.success(f"Predicted Class: **{label}**")
