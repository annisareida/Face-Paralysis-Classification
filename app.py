import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import dlib
import cv2

# -----------------------------
# Load model
# -----------------------------
def load_model():
    return tf.keras.models.load_model('vggface_model_20251028_144939.h5')

model = load_model()

# Kelas berdasarkan skala eFace
class_names = ['Complete', 'Mild', 'Moderate', 'Near Normal', 'Normal', 'Severe']

# -----------------------------
# Inisialisasi detektor wajah dlib
# -----------------------------
face_detector = dlib.get_frontal_face_detector()

def crop_face_dlib(image_pil, margin=40):
    """
    Deteksi wajah dengan dlib dan crop area wajah.
    """
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        st.warning("⚠️ No face detected — using full image.")
        return image_pil

    # Ambil wajah pertama yang terdeteksi
    face = faces[0]
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

    # Tambahkan margin agar crop tidak terlalu ketat
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    y2 = min(img.shape[0], y2 + margin)

    cropped_face = img[y1:y2, x1:x2]
    return Image.fromarray(cropped_face)

# -----------------------------
# Normalisasi (pengganti preprocess_input)
# -----------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

def preprocess_image(img_pil):
    """
    Ubah gambar menjadi array, resize ke 224x224, dan normalisasi.
    """
    img = img_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = normalization_layer(img_array)  # Normalisasi ke [0,1]
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
st.title("Face Paralysis Detection (with dlib Face Cropping)")
st.write("Upload or capture an image, and classify the face according to the **eFace Scale**:")
st.markdown("**Classes:** Complete · Mild · Moderate · Near Normal · Normal · Severe")

# Pilihan input
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

# Proses prediksi
if image_source:
    st.image(image_source, caption="Original Image", use_container_width=True)

    if st.button("Predict"):
        cropped_face = crop_face_dlib(image_source)
        st.image(cropped_face, caption="Cropped Face", use_container_width=True)

        label = predict_image(cropped_face)
        st.success(f"Predicted Class: **{label}**")
