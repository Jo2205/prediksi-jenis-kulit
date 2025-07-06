import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# --- Konstanta ---
CLASS_NAMES = ['Berjerawat', 'Berminyak', 'Dermatitis_Perioral', 'Kering', 'Normal', 'Penuaaan', 'Vitiligo']
IMAGE_SIZE = (224, 224)

# --- Fungsi Memuat Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_fine_tuned.h5")

model = load_model()

# --- Judul Aplikasi ---
st.title("🧑‍⚕️ Aplikasi Prediksi Jenis Kulit (Multi Gambar)")

# --- Upload Gambar Multiple ---
uploaded_files = st.file_uploader("Unggah beberapa gambar wajah (jpg/jpeg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# --- Input nama-nama (manual)
if uploaded_files:
    st.markdown("### ✏️ Masukkan nama untuk masing-masing gambar:")
    names = []
    for i, uploaded_file in enumerate(uploaded_files):
        name = st.text_input(f"Nama untuk gambar {i+1} ({uploaded_file.name})", value=f"Pasien {i+1}")
        names.append(name)

    if st.button("🔍 Jalankan Prediksi"):
        results = []

        for img_file, name in zip(uploaded_files, names):
            image = Image.open(img_file).convert("RGB")
            image_resized = image.resize(IMAGE_SIZE)
            img_array = np.array(image_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Prediksi
            predictions = model.predict(img_batch)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = predictions[predicted_idx] * 100

            # Tampilkan hasil per gambar
            st.markdown(f"#### 🖼️ {name} ({img_file.name})")
            st.image(image, caption="Gambar diunggah", use_column_width=True)
            st.success(f"Prediksi: **{predicted_class}** ({confidence:.2f}%)")

            # Simpan data ke list
            result = {
                "Nama": name,
                "Nama File": img_file.name,
                "Prediksi": predicted_class,
                "Probabilitas (%)": round(confidence, 2),
            }

            # Tambahkan semua kelas dan probabilitasnya
            for i, cls in enumerate(CLASS_NAMES):
                result[f"Prob_{cls} (%)"] = round(predictions[i] * 100, 2)

            results.append(result)

        # Buat dataframe hasil
        df_results = pd.DataFrame(results)

        # Simpan ke CSV dalam memory
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.markdown("### 💾 Unduh Hasil Prediksi")
        st.download_button("📥 Unduh CSV", data=csv_data, file_name="hasil_prediksi_kulit.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("### 📈 Contoh Grafik Probabilitas Gambar Terakhir")

        # Tampilkan grafik dari gambar terakhir
        fig, ax = plt.subplots(figsize=(10, 4))
        last_pred = predictions
        bars = ax.bar(CLASS_NAMES, last_pred * 100, color="skyblue")
        ax.set_ylabel("Probabilitas (%)")
        ax.set_title(f"Distribusi Kelas: {predicted_class}")
        ax.set_ylim([0, 100])
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        st.pyplot(fig)

