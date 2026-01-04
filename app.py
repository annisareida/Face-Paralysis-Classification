import os
import time
import zipfile
import io
import pandas as pd # Tambahkan pandas untuk tabel hasil
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN

# ==========================================
# 1. CLASS FACE DETECTOR
# ==========================================
class FaceDetector:
    def __init__(self):
        self.detector = self._load_detector()

    @staticmethod
    @st.cache_resource
    def _load_detector():
        return MTCNN()

    def crop_face(self, image_pil, margin=40):
        try:
            img_array = np.array(image_pil)
            detections = self.detector.detect_faces(img_array)

            if len(detections) == 0:
                return None

            x, y, w, h = detections[0]['box']
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_array.shape[1], x + w + margin)
            y2 = min(img_array.shape[0], y + h + margin)

            cropped_face = img_array[y1:y2, x1:x2]
            return Image.fromarray(cropped_face)
        except:
            return None

# ==========================================
# 2. CLASS FACE CLASSIFIER
# ==========================================
class FaceClassifier:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 'Severe']
        self.normalization_layer = tf.keras.layers.Rescaling(1./255)

    @staticmethod
    @st.cache_resource
    def _load_model(path):
        return tf.keras.models.load_model(path)

    def _preprocess_image(self, img_pil):
        img = img_pil.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = self.normalization_layer(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_pil):
        img_array = self._preprocess_image(img_pil)
        prediction = self.model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction[0])
        return self.class_names[predicted_index]

# ==========================================
# 3. CLASS FACE PARALYSIS APP (Batch Mode)
# ==========================================
class FaceParalysisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Sistem Deteksi Kelumpuhan Wajah (Batch)",
            page_icon="🏥",
            layout="wide"
        )
        
        self.detector = FaceDetector()
        # Ganti dengan path model Anda
        self.classifier = FaceClassifier('vggface_model_20251028_144939.h5')

    def display_sidebar(self):
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
            st.title("Panduan Batch Upload")
            st.info(
                """
                1. Siapkan file **ZIP** berisi kumpulan foto wajah (.jpg, .png).
                2. Unggah file ZIP tersebut.
                3. Klik tombol **'Mulai Prediksi Massal'**.
                4. Sistem akan menghitung jumlah prediksi per kategori.
                """
            )
            st.warning("⚠️ Pastikan foto di dalam ZIP memiliki pencahayaan yang cukup.")

    def display_header(self):
        st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Batch Classification: Kelumpuhan Wajah</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Uji coba banyak gambar sekaligus menggunakan <b>VGG-Face</b></p>", unsafe_allow_html=True)
        st.divider()

    def handle_upload(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_zip = st.file_uploader("Unggah File ZIP (Kumpulan Gambar)", type=["zip"])

        if uploaded_zip:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_btn = st.button("🚀 Mulai Prediksi Massal", type="primary", use_container_width=True)

            if analyze_btn:
                self.process_zip(uploaded_zip)

    def process_zip(self, uploaded_zip):
        # Inisialisasi hitungan
        results_count = {name: 0 for name in self.classifier.class_names}
        results_count["Wajah Tidak Terdeteksi"] = 0
        
        detail_list = [] # Untuk tabel detail per file
        
        try:
            with zipfile.ZipFile(uploaded_zip, "r") as z:
                # Ambil daftar file yang valid (gambar saja)
                file_list = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                total_files = len(file_list)
                
                if total_files == 0:
                    st.error("ZIP kosong atau tidak berisi file gambar yang didukung.")
                    return

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()

                for i, file_name in enumerate(file_list):
                    # Update progress
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"Memproses {i+1}/{total_files}: {file_name}")

                    # Baca gambar dari ZIP
                    img_data = z.read(file_name)
                    img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')

                    # Deteksi & Prediksi
                    cropped_face = self.detector.crop_face(img_pil)
                    
                    if cropped_face:
                        label = self.classifier.predict(cropped_face)
                        results_count[label] += 1
                        detail_list.append({"Nama File": file_name, "Hasil": label})
                    else:
                        results_count["Wajah Tidak Terdeteksi"] += 1
                        detail_list.append({"Nama File": file_name, "Hasil": "Gagal Deteksi"})

                end_time = time.time()
                total_duration = end_time - start_time
                
                progress_bar.empty()
                status_text.empty()
                
                # TAMPILKAN HASIL AKHIR
                self.show_batch_results(results_count, detail_list, total_files, total_duration)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengekstrak ZIP: {e}")

    def show_batch_results(self, results_count, detail_list, total_files, duration):
        st.success(f"✅ Pemrosesan Selesai dalam {duration:.2f} detik")
        
        # Grid Dashboard
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("Total Gambar", total_files)
        col_metric2.metric("Berhasil", total_files - results_count["Wajah Tidak Terdeteksi"])
        col_metric3.metric("Gagal Deteksi", results_count["Wajah Tidak Terdeteksi"])

        st.divider()

        # Layout Hasil: Grafik dan Tabel
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            st.subheader("📊 Grafik Distribusi Kelas")
            # Konversi dict ke DataFrame untuk charting
            df_chart = pd.DataFrame(list(results_count.items()), columns=['Kategori', 'Jumlah'])
            st.bar_chart(df_chart.set_index('Kategori'))

        with col_table:
            st.subheader("📋 Tabel Ringkasan")
            st.table(df_chart)

        # Expander untuk daftar detail semua file
        with st.expander("🔍 Lihat Detail Per File"):
            df_detail = pd.DataFrame(detail_list)
            st.dataframe(df_detail, use_container_width=True)

    def run(self):
        self.display_sidebar()
        self.display_header()
        self.handle_upload()

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = FaceParalysisApp()
    app.run()
