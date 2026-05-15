import os 
import time  # Tambahkan library time 
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
 
# ========================================== 
# 2. CLASS FACE CLASSIFIER 
# ========================================== 
class FaceClassifier: 
    def __init__(self, model_path): 
        self.model = self._load_model(model_path) 
        self.class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 
'Severe'] 
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
         
        # Simulasi waktu proses (opsional, hanya agar progress bar terlihat) 
        # time.sleep(0.5)  
         
        prediction = self.model.predict(img_array) 
        predicted_index = np.argmax(prediction[0]) 
        return self.class_names[predicted_index] 
 
# ========================================== 
# 3. CLASS FACE PARALYSIS APP (UI dipercantik) 
# ========================================== 
class FaceParalysisApp: 
    def __init__(self): 
        # Konfigurasi Halaman (Harus di baris pertama dalam init) 
        st.set_page_config( 
            page_title="Sistem Deteksi Kelumpuhan Wajah", 
 
            page_icon="         ", 
            layout="wide"  # Menggunakan layout lebar 
        ) 
         
        self.detector = FaceDetector() 
        # Pastikan path model benar 
        self.classifier = FaceClassifier('vggface_model_20251028_144939.h5') 
 
    def display_sidebar(self): 
        """Menampilkan instruksi di sidebar agar UI utama bersih""" 
        with st.sidebar: 
            st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", 
width=100) 
            st.title("Panduan Pengguna") 
            st.info( 
                """ 
                1. **Upload** gambar wajah pasien. 
                2. Pastikan wajah terlihat **jelas** dan **fokus**. 
                3. Klik tombol **'Predict'**. 
                4. Sistem akan mendeteksi wajah dan memberikan hasil klasifikasi. 
                """ 
            ) 
            st.warning("    Aplikasi ini adalah alat bantu diagnosis awal, bukan 
pengganti diagnosis dokter.") 
             
            st.markdown("---") 
            st.caption("Skripsi Teknik Informatika 2025") 
 
    def display_header(self): 
        st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Klasifikasi 
Tingkat Kelumpuhan Wajah</h1>", unsafe_allow_html=True) 
        st.markdown("<p style='text-align: center;'>Menggunakan Arsitektur 
<b>VGG-Face</b> dengan Skala <b>eFACE</b></p>", unsafe_allow_html=True) 
        st.divider() 
 
    def handle_upload(self): 
        # Container untuk upload di tengah 
        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2: 
            uploaded_file = st.file_uploader("Unggah Citra Wajah (JPG/PNG)", 
type=["jpg", "jpeg", "png", "bmp"]) 
 
 
 
        if uploaded_file: 
            image_source = Image.open(uploaded_file).convert('RGB') 
             
            # Tampilkan gambar asli di tengah sebelum diproses 
            col1, col2, col3 = st.columns([1, 2, 1]) 
            with col2: 
                st.image(image_source, caption="Preview Citra Asli", 
use_container_width=True) 
                 
                # Tombol besar di tengah 
                analyze_btn = st.button("    Predict", type="primary", 
use_container_width=True) 
 
            if analyze_btn: 
                self.process_image(image_source) 
 
    def process_image(self, image_source): 
        st.divider() 
         
        # Mulai menghitung waktu 
        start_time = time.time() 
         
        # Progress Bar untuk UX yang lebih baik 
        progress_text = "Sedang memproses citra..." 
        my_bar = st.progress(0, text=progress_text) 
 
        try: 
            # 1. Deteksi Wajah 
            my_bar.progress(30, text="Mendeteksi wajah...") 
            cropped_face = self.detector.crop_face(image_source) 
 
            if cropped_face: 
                # 2. Klasifikasi 
                my_bar.progress(70, text="Menganalisis tingkat keparahan...") 
                label = self.classifier.predict(cropped_face) 
                 
                # Selesai menghitung waktu 
                end_time = time.time() 
                inference_time = end_time - start_time 
                 
                my_bar.progress(100, text="Selesai!") 
                time.sleep(0.5) # Jeda dikit biar user liat 100% 
 
                my_bar.empty() # Hilangkan progress bar 
 
                # TAMPILAN HASIL (Layout 2 Kolom) 
                self.show_results(cropped_face, label, inference_time) 
             
            else: 
                my_bar.empty() 
                st.error("  Wajah tidak terdeteksi! Harap unggah foto dengan wajah 
yang jelas.") 
         
        except Exception as e: 
            my_bar.empty() 
            st.error(f"Terjadi kesalahan: {e}") 
 
    def show_results(self, cropped_face, label, inference_time): 
        """Menampilkan hasil dengan layout kolom yang rapi""" 
         
        # Mengubah rasio kolom agar gambar crop lebih kecil 
        # Sebelumnya [1, 1], sekarang [1, 2] (Kolom hasil lebih lebar, kolom gambar 
lebih sempit) 
        # Atau bisa pakai width=tertentu di st.image 
        col_img, col_result = st.columns([1, 2], gap="large") 
 
        # Kolom Kiri: Gambar Hasil Crop (Lebih Kecil) 
        with col_img: 
            st.subheader("Wajah Terdeteksi") 
            # Mengurangi width agar tidak memenuhi layar (misal 250px atau 300px) 
            st.image(cropped_face, width=250, caption="Region of Interest (ROI)") 
 
        # Kolom Kanan: Hasil Prediksi & Metrik 
        with col_result: 
            st.subheader("Hasil Diagnosis Sistem") 
             
            # Kotak hasil utama 
            st.success(f"### Kategori: **{label}**") 
             
            # Tampilkan Metrik Waktu Inferensi 
            st.metric(label="Waktu Inferensi", value=f"{inference_time:.4f} detik", 
delta_color="off") 
 
            st.caption(f"Sistem mengklasifikasikan citra ini sebagai **{label}** dalam 
waktu **{inference_time:.4f} detik**.") 
 
 
 
            # Expander untuk detail kelas (opsional, pemanis) 
            with st.expander("ℹ️ Tentang Kategori Ini", expanded=True): 
                if label == "Normal": 
                    st.write("**Normal:** Fungsi wajah normal sepenuhnya. Simetris saat 
istirahat dan bergerak.") 
                elif label == "Near Normal": 
                    st.write("**Near Normal:** Sedikit kelemahan terlihat hanya pada 
inspeksi dekat. Simetris saat istirahat, sedikit asimetris saat bergerak.") 
                elif label == "Mild": 
                    st.write("**Mild:** Disfungsi ringan. Asimetri terlihat saat 
pergerakan, namun masih bisa menutup mata dengan usaha minimal.") 
                elif label == "Moderate": 
                    st.write("**Moderate:** Disfungsi sedang. Asimetri jelas terlihat. 
Mata mungkin tidak menutup sempurna tanpa usaha. Pergerakan dahi berkurang.") 
                elif label == "Severe": 
                    st.write("**Severe:** Disfungsi berat. Asimetri sangat jelas. Tidak ada 
pergerakan dahi. Mata tidak bisa menutup sempurna. Mulut sedikit bergerak.") 
                elif label == "Complete": 
                    st.write("**Complete:** Kelumpuhan total. Tidak ada pergerakan otot 
wajah sama sekali. Wajah sangat asimetris.") 
                else: 
                    st.write(f"Tingkat kelumpuhan wajah kategori {label}. Disarankan 
konsultasi lebih lanjut dengan dokter.") 
 
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
