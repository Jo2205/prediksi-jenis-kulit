# Aplikasi Prediksi Jenis Kulit (Multi Gambar)
Aplikasi ini bertujuan untuk memprediksi jenis kulit wajah dari gambar yang diunggah, seperti berjerawat, berminyak, kering, dan lainnya. Aplikasi ini menggunakan model deep learning untuk menganalisis gambar dan memberikan prediksi jenis kulit beserta tingkat kepercayaannya.

# Teknologi yang Digunakan
* Streamlit: Untuk membuat antarmuka aplikasi web interaktif.
* TensorFlow: Untuk melakukan prediksi menggunakan model deep learning.
* Pandas: Untuk pengelolaan data dan hasil prediksi.
* Matplotlib: Untuk membuat visualisasi grafik probabilitas prediksi.

# Model yang Digunakan
Model yang digunakan adalah best_model_fine_tuned.h5, yang merupakan model deep learning yang telah dilatih dengan menggunakan dataset dari Kaggle. Model ini mampu memprediksi jenis kulit wajah dengan akurasi yang baik.
Akurasi Model: 89%

# Dataset
Dataset yang digunakan untuk melatih model ini berasal dari Kaggle. Dataset ini berisi gambar wajah dengan berbagai label jenis kulit, yang digunakan untuk melatih model deep learning.
Link ke dataset: https://www.kaggle.com/datasets/igecko/14032025-1315-klasifikasikulitwajah7kelas


# Instalasi dan Menjalankan
# 1. Clone Repository dan Masuk Folder

```bash
git clone https://github.com/Jo2205/prediksi-jenis-kulit.git
cd prediksi-jenis-kulit
```

# 2. Install Library
```bash
pip install -r requirements.txt
```

# 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

# Fitur Visualisasi
Aplikasi ini juga dilengkapi dengan fitur visualisasi berupa grafik probabilitas, yang menunjukkan seberapa besar kemungkinan masing-masing kelas (misalnya, "Normal", "Berjerawat", dll.). Grafik ini memudahkan pengguna untuk memahami distribusi probabilitas prediksi untuk gambar yang diunggah.
Contoh:
* Prediksi: Normal (80%)
* Probabilitas Kelas Lain:
* Berjerawat: 10%
* Berminyak: 5%
* Kering: 5%

Proyek ini saya kembangkan sebagai bagian dari portofolio data science.
