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
# UI STYLE (TAMPILAN SAJA)
# =======================
st.markdown("""
<style>
body { background-color: #0e1117; }
h1, h2, h3, h4 { color: #ffffff; }
.section {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
.card {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}
.autism {
    color: #ff4b4b;
    font-size: 26px;
    font-weight: bold;
}
.normal {
    color: #4bff8f;
    font-size: 26px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown("""
<h1 style='text-align:center;'>🧠 HybrideFace Autism Detector</h1>
<p style='text-align:center; color:#bfbfbf;'>
Deteksi Autisme Berbasis Citra Wajah<br>
<strong>CNN • KNN • SVM</strong>
</p>
<hr>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ Sistem ini bukan alat diagnosis medis. "
    "Hasil prediksi hanya bersifat pendukung keputusan."
)

# =======================
# LOAD MODEL (ASLI)
# =======================
IMG_SIZE = 128
K.clear_session()

cnn = load_model("models/cnn_model.h5", compile=False)
knn = pickle.load(open("models/knn.pkl", "rb"))
svm = pickle.load(open("models/svm.pkl", "rb"))

cm_cnn = np.load("models/cm_cnn.npy")
cm_knn = np.load("models/cm_knn.npy")
cm_svm = np.load("models/cm_svm.npy")

# =======================
# BUILD CNN (ASLI)
# =======================
cnn.build((None, IMG_SIZE, IMG_SIZE, 3))

# =======================
# FEATURE EXTRACTION (ASLI)
# =======================
def extract_features(x):
    for layer in cnn.layers[:-1]:
        x = layer(x)
    return x

# =======================
# SECTION: UPLOAD
# =======================
st.markdown("<div class='section'><h2>📤 Upload Foto Wajah</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Format yang didukung: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Gambar tidak valid.")
        st.stop()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    col_img, col_result = st.columns([1, 2])

    with col_img:
        st.image(img, caption="Foto Wajah", width=300)

    img_norm = img.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # =======================
    # FEATURE EXTRACTION (ASLI)
    # =======================
    features = extract_features(img_input).numpy()

    # =======================
    # PREDIKSI (ASLI)
    # =======================
    pred_cnn = int(np.argmax(cnn.predict(img_input), axis=1)[0])
    pred_knn = int(knn.predict(features)[0])
    pred_svm = int(svm.predict(features)[0])

    with col_result:
        st.markdown("<h3>🔍 Hasil Prediksi</h3>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        def show(col, title, pred):
            label = "Autis" if pred == 1 else "Non-Autis"
            css = "autism" if pred == 1 else "normal"
            col.markdown(
                f"""
                <div class="card">
                    <h4>{title}</h4>
                    <div class="{css}">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        show(r1, "CNN", pred_cnn)
        show(r2, "KNN", pred_knn)
        show(r3, "SVM", pred_svm)

st.markdown("</div>", unsafe_allow_html=True)

# =======================
# CONFUSION MATRIX (ASLI)
# =======================
st.markdown("<div class='section'><h2>📊 Confusion Matrix Evaluasi Model</h2>", unsafe_allow_html=True)

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(2.3, 2.3))  # ⬅️ LEBIH KECIL
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,              # ⬅️ hilangkan colorbar agar ringkas
        annot_kws={"size": 5},   # ⬅️ kecilkan angka
        xticklabels=["Non-Autis", "Autis"],
        yticklabels=["Non-Autis", "Autis"],
        ax=ax
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    st.pyplot(fig)


c1, c2, c3 = st.columns(3)
with c1:
    plot_cm(cm_cnn, "CNN")
with c2:
    plot_cm(cm_knn, "KNN")
with c3:
    plot_cm(cm_svm, "SVM")

st.markdown("</div>", unsafe_allow_html=True)

# =======================
# 📈 TABEL AKURASI (TAMBAHAN)
# =======================
st.markdown("<div class='section'><h2>📈 Tabel Akurasi Model</h2>", unsafe_allow_html=True)

def calc_accuracy(cm):
    return (cm[0,0] + cm[1,1]) / cm.sum()

acc_cnn = calc_accuracy(cm_cnn)
acc_knn = calc_accuracy(cm_knn)
acc_svm = calc_accuracy(cm_svm)

st.table({
    "Metode": ["CNN", "KNN", "SVM"],
    "Akurasi (%)": [
        f"{acc_cnn*100:.2f}",
        f"{acc_knn*100:.2f}",
        f"{acc_svm*100:.2f}"
    ]
})

st.markdown("</div>", unsafe_allow_html=True)

# =======================
# 📊 GRAFIK BAR AKURASI (TAMBAHAN)
# =======================
st.markdown("<div class='section'><h2>📊 Grafik Perbandingan Akurasi</h2>", unsafe_allow_html=True)

methods = ["CNN", "KNN", "SVM"]
accuracies = [acc_cnn*100, acc_knn*100, acc_svm*100]

fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.bar(methods, accuracies)

ax.set_ylim(0, 100)
ax.set_ylabel("Akurasi (%)")
ax.set_xlabel("Metode")
ax.set_title("Perbandingan Akurasi CNN, KNN, dan SVM")

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.2f}%", ha="center")

st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# =======================
# FOOTER
# =======================
st.markdown("""
<div class="footer">
© 2025 | HybrideFace Autism Detector <br>
CNN • KNN • SVM | Streamlit Web Application
</div>
""", unsafe_allow_html=True)
