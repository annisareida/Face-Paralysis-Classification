import os
# Paksa penggunaan CPU untuk menghindari error CUDA jika GPU tidak dikonfigurasi dengan benar
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN

# ==========================================
# 1. CLASS FACE DETECTOR
# Bertanggung jawab untuk deteksi dan crop wajah
# ==========================================
class FaceDetector:
    def __init__(self):
        self.detector = self._load_detector()

    # Menggunakan st.cache_resource agar MTCNN hanya dimuat sekali
    @staticmethod
    @st.cache_resource
    def _load_detector():
        return MTCNN()

    def crop_face(self, image_pil, margin=40):
        """
        Mendeteksi wajah dan mengembalikan gambar yang di-crop.
        Mengembalikan None jika wajah tidak ditemukan.
        """
        img_array = np.array(image_pil)
        detections = self.detector.detect_faces(img_array)

        if len(detections) == 0:
            return None

        # Ambil wajah pertama yang terdeteksi
        x, y, w, h = detections[0]['box']

        # Tambahkan margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_array.shape[1], x + w + margin)
        y2 = min(img_array.shape[0], y + h + margin)

        cropped_face = img_array[y1:y2, x1:x2]
        return Image.fromarray(cropped_face)

# ==========================================
# 2. CLASS FACE CLASSIFIER
# Bertanggung jawab untuk memuat model dan melakukan prediksi
# ==========================================
class FaceClassifier:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 'Severe']
        self.normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Menggunakan st.cache_resource agar Model Keras hanya dimuat sekali
    @staticmethod
    @st.cache_resource
    def _load_model(path):
        return tf.keras.models.load_model(path)

    def _preprocess_image(self, img_pil):
        """Mengubah ukuran dan normalisasi gambar untuk model."""
        img = img_pil.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = self.normalization_layer(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_pil):
        """Melakukan prediksi dan mengembalikan label kelas."""
        img_array = self._preprocess_image(img_pil)
        prediction = self.model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        return self.class_names[predicted_index]

# ==========================================
# 3. CLASS FACE PARALYSIS APP (Main UI)
# Bertanggung jawab menangani tampilan Streamlit
# ==========================================
class FaceParalysisApp:
    def __init__(self):
        # Inisialisasi komponen logika
        self.detector = FaceDetector()
        # Pastikan nama file model sesuai dengan yang ada di folder Anda
        self.classifier = FaceClassifier('vggface_model_20251028_144939.h5')

    def display_header(self):
        st.title("Klasifikasi Kelumpuhan Wajah (eFace Scale)")
        st.write("Unggah gambar untuk mengklasifikasikan kelumpuhan wajah berdasarkan **Skala eFace**.")

    def handle_upload(self):
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image_source = Image.open(uploaded_file).convert('RGB')
            st.image(image_source, caption="Gambar Asli", use_container_width=True)
            
            # Tampilkan tombol prediksi hanya jika gambar sudah ada
            if st.button("Mulai Prediksi"):
                self.process_image(image_source)

    def process_image(self, image_source):
        with st.spinner("Mendeteksi wajah dan melakukan prediksi..."):
            # 1. Deteksi & Crop
            cropped_face = self.detector.crop_face(image_source)

            if cropped_face:
                # Tampilkan wajah yang terdeteksi
                st.image(cropped_face, caption="Wajah Terdeteksi (Cropped)", use_container_width=True)
                
                # 2. Klasifikasi
                label = self.classifier.predict(cropped_face)
                self.show_success(label)
            else:
                self.show_error("Wajah tidak terdeteksi. Proses dibatalkan karena tidak ada wajah yang terdeteksi.")

    def show_success(self, label):
        st.success(f"Hasil Prediksi: **{label}**")

    def show_error(self, message):
        st.error(message)

    def run(self):
        """Method utama untuk menjalankan aplikasi"""
        self.display_header()
        self.handle_upload()

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Membuat instance aplikasi dan menjalankannya
    app = FaceParalysisApp()
    app.run()
