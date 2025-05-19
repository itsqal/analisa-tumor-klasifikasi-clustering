# Analisis Prediktif dan Segmentasi Data Kanker Payudara Menggunakan Logistic Regression dan K-Means Clustering | Kelompok 12

# Anggota Kelompok
- Anaking Faiqal Lufi | 1202223053
- Muhammad Ricko Arif Andrian | 1202220216
- Muhamad Fadhilah | 102022300190

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

## Deskripsi Proyek
Proyek ini menganalisis dataset kanker payudara untuk melakukan **klasifikasi** (menggunakan Logistic Regression) dan **klasterisasi** (menggunakan K-Means) dengan pendekatan machine learning. Proyek mencakup eksplorasi data, preprocessing, deteksi outlier, pemodelan, evaluasi, serta visualisasi hasil untuk memahami karakteristik data dan performa model.

## Fitur Utama
- Eksplorasi data dengan visualisasi distribusi fitur dan statistik deskriptif
- Deteksi dan visualisasi outlier menggunakan boxplot
- Preprocessing data: mapping label, splitting, dan scaling dengan RobustScaler
- Klasifikasi menggunakan Logistic Regression dengan hyperparameter tuning
- Evaluasi model dengan metrik akurasi, confusion matrix, ROC-AUC, dan classification report
- Klasterisasi dengan K-Means dan analisis karakteristik klaster
- Visualisasi hasil analisis, korelasi fitur, dan klaster
- Penyimpanan model terbaik untuk kebutuhan deployment
- Dashboard [On Progress]

## Prasyarat
- Python 3.7 atau lebih tinggi
- Library: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`
- Jupyter Notebook atau JupyterLab

## Instalasi
1. Clone repository ini:
   ```bash
   git clone https://github.com/[username]/[repository].git
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd [repository]
   ```
3. Instal dependensi:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib notebook
   ```

## Cara Menjalankan
1. Buka terminal dan masuk ke direktori proyek.
2. Jalankan Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Buka file `.ipynb` di browser dan jalankan setiap cell secara berurutan.
4. **Catatan**: Fitur dashboard berbasis Streamlit belum tersedia. Dokumentasi visual dashboard akan ditambahkan setelah pengembangan selesai. [On Progress]

## Struktur Proyek
- **clf_model_breast_cancer.pkl/**: Model yang disimpan untuk deployment
- **Analisis Prediktif dan Segmentasi Data Kanker Payudara.ipynb/**: File Jupyter Notebook dengan analisis lengkap
- **README.md**: Dokumentasi proyek

### Alur Analisis
1. **Eksplorasi Data**:
   - Pemeriksaan dimensi, tipe data, duplikasi, dan nilai hilang
   - Visualisasi distribusi fitur
2. **Deteksi Outlier**:
   - Analisis outlier menggunakan boxplot
3. **Korelasi Fitur**:
   - Matriks korelasi untuk memahami hubungan antar fitur
4. **Preprocessing**:
   - Mapping label target ke numerik
   - Pembagian data ke train dan test set
   - Scaling fitur dengan RobustScaler
5. **Klasifikasi**:
   - Pelatihan Logistic Regression
   - Evaluasi dengan akurasi, confusion matrix, ROC-AUC
   - Hyperparameter tuning dengan GridSearchCV
6. **Klasterisasi**:
   - Scaling fitur untuk K-Means
   - Penentuan jumlah klaster optimal (elbow method)
   - Visualisasi dan analisis karakteristik klaster
7. **Penyimpanan Model**:
   - Menyimpan model terbaik menggunakan `joblib`
