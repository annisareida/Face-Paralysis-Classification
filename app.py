# PERBAIKAN 1: Tambahkan 'import os' dan paksa penggunaan CPU.
# Ini harus menjadi baris PERTAMA sebelum mengimpor TensorFlow.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN  # untuk deteksi wajah

# -----------------------------
# Load model & detector (DENGAN CACHING)
# -----------------------------

# PERBAIKAN 2: Gunakan @st.cache_resource agar model tidak di-load ulang
# setiap kali ada interaksi UI. Ini meningkatkan performa secara drastis.
@st.cache_resource
def load_model_cached():
    """Loads the Keras model from disk."""
    return tf.keras.models.load_model('vggface_model_20251028_144939.h5')

@st.cache_resource
def load_detector_cached():
    """Initializes the MTCNN face detector."""
    return MTCNN()

# Load model dan detector menggunakan fungsi caching
model = load_model_cached()
detector = load_detector_cached()

# Kelas berdasarkan skala eFace
class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 'Severe']

# -----------------------------
# Fungsi deteksi dan crop wajah
# -----------------------------
def crop_face(image_pil, margin=40):
    """
    Deteksi wajah dengan MTCNN dan crop area wajah dengan margin.
    
    PERBAIKAN 3: Jika tidak ditemukan wajah, kembalikan None.
    Ini akan mencegah aplikasi memprediksi gambar penuh (full image).
    """
    img_array = np.array(image_pil)
    detections = detector.detect_faces(img_array)

    if len(detections) == 0:
        st.warning("Wajah tidak terdeteksi. Tidak dapat melanjutkan klasifikasi.")
        return None  # <-- PERBAIKAN: Kembalikan None, bukan gambar asli

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
# (Tidak perlu diubah, ini sudah benar)
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
# (Tidak perlu diubah, ini sudah benar)
def predict_image(img_pil):
    img_array = preprocess_image(img_pil)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    return class_names[predicted_index]

# -----------------------------
# UI Streamlit
# -----------------------------
st.title("Klasifikasi Kelumpuhan Wajah (eFace Scale)")
st.write("Unggah gambar untuk mengklasifikasikan kelumpuhan wajah berdasarkan **Skala eFace**: "
         "**Complete, Mild, Moderate, Near Normal, Normal, Severe**.")

# -----------------------------
# Upload Gambar
# -----------------------------
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_source = Image.open(uploaded_file).convert('RGB')
    st.image(image_source, caption="Gambar Asli", use_container_width=True)

    if st.button("Mulai Prediksi"):
        # PERBAIKAN: Tambahkan spinner agar pengguna tahu aplikasi sedang bekerja
        with st.spinner("Mendeteksi wajah dan melakukan prediksi..."):
            
            # 1. Coba crop wajah
            cropped_face = crop_face(image_source)

            # PERBAIKAN 3 (Lanjutan):
            # Hanya lanjutkan jika crop_face() berhasil mengembalikan gambar (bukan None)
            if cropped_face:
                st.image(cropped_face, caption="Wajah Terdeteksi", use_container_width=True)

                # 2. Prediksi kelas wajah hasil crop
                label = predict_image(cropped_face)
                st.success(f"Hasil Prediksi: **{label}**")
            else:
                # Pesan ini akan muncul jika 'crop_face' mengembalikan None
                st.error("Proses dibatalkan karena tidak ada wajah yang terdeteksi.")
