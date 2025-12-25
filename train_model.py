import streamlit as st
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="HybrideFace Autism Detector",
    page_icon="🧠",
    layout="wide"
)

# =======================
# CUSTOM CSS
# =======================
st.markdown("""
<style>
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.result-autism { color: #ff4b4b; font-size: 24px; font-weight: bold; }
.result-normal { color: #4bff8f; font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 HybrideFace Autism Detector")
st.caption("CNN • KNN • SVM")

st.warning(
    "⚠️ Sistem ini bukan alat diagnosis medis. "
    "Hasil prediksi hanya bersifat pendukung."
)

# =======================
# LOAD MODEL (FINAL AMAN)
# =======================
IMG_SIZE = 128

K.clear_session()

cnn = load_model("models/cnn_model.h5")
knn = pickle.load(open("models/knn.pkl", "rb"))
svm = pickle.load(open("models/svm.pkl", "rb"))

cm_cnn = np.load("models/cm_cnn.npy")
cm_knn = np.load("models/cm_knn.npy")
cm_svm = np.load("models/cm_svm.npy")

# 🔒 FEATURE EXTRACTOR (NAMA UNIK → TIDAK BISA KETIMPA)
cnn_feature_model = Model(
    inputs=cnn.input,
    outputs=cnn.layers[-2].output
)

# CEK KERAS MODEL (STOP JIKA SALAH)
if not hasattr(cnn_feature_model, "predict"):
    st.error("Feature model bukan Keras Model. Hentikan eksekusi.")
    st.stop()

# =======================
# UPLOAD IMAGE
# =======================
uploaded_file = st.file_uploader(
    "Upload Foto Wajah (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Gambar tidak valid.")
        st.stop()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    st.image(img, caption="Foto Wajah", width=300)

    img_input = np.expand_dims(img.astype("float32") / 255.0, axis=0)

    # =======================
    # PREDIKSI (FINAL)
    # =======================
    features = cnn_feature_model.predict(img_input)

    pred_cnn = int(np.argmax(cnn.predict(img_input), axis=1)[0])
    pred_knn = int(knn.predict(features)[0])
    pred_svm = int(svm.predict(features)[0])

    c1, c2, c3 = st.columns(3)

    def show(col, title, pred):
        label = "Autis" if pred == 1 else "Non-Autis"
        css = "result-autism" if pred == 1 else "result-normal"
        col.markdown(f"<div class='card'><h4>{title}</h4><p class='{css}'>{label}</p></div>",
                     unsafe_allow_html=True)

    show(c1, "CNN", pred_cnn)
    show(c2, "KNN", pred_knn)
    show(c3, "SVM", pred_svm)

# =======================
# CONFUSION MATRIX
# =======================
st.subheader("📊 Confusion Matrix")

def plot_cm(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Autis", "Autis"],
                yticklabels=["Non-Autis", "Autis"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

c1, c2, c3 = st.columns(3)
plot_cm(cm_cnn, "CNN")
plot_cm(cm_knn, "KNN")
plot_cm(cm_svm, "SVM")
