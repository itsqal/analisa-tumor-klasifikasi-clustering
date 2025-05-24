import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from dataset import X, y

rf_model = joblib.load("./clf_model_breast_cancer.pkl")
# confusion_matrix_image = "./Confusion Matrix Model.png"
# pca_image = "./K-means_PCA_breast_cancer_classification.png"

label_encoder = LabelEncoder()
label_encoder.fit(['B', 'M'])
y_encoded = label_encoder.transform(y.values.ravel())

# Prediksi akurasi model
y_pred = rf_model.predict(X)
accuracy = accuracy_score(y_encoded, y_pred)

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ("Performa Model", "Prediksi Model"))

# **Model Performance Page**
if page == "Performa Model":
    st.title('Model Klasifikasi Tumor Payudara - Performance')

    st.markdown("""
    ### Deskripsi Dataset dan Algoritma

    **Dataset:**  
    Dataset yang digunakan adalah dataset kanker payudara yang tersedia di UCI Machine Learning Repository. Dataset ini terdiri dari 30 fitur yang menggambarkan berbagai karakteristik kanker payudara, seperti ukuran massa tumor, bentuk, dan lainnya. Label kelas terbagi menjadi dua kategori, yaitu 'M' untuk malignant (kanker ganas) dan 'B' untuk benign (kanker jinak). Dataset ini digunakan untuk melakukan klasifikasi apakah tumor tersebut benign atau malignant berdasarkan fitur-fitur yang ada.

    **Algoritma yang Digunakan:**  
    Pada dashboard ini, digunakan algoritma **Random Forest**, yang merupakan ensemble method yang menggunakan banyak decision trees untuk melakukan prediksi. Random Forest membangun banyak pohon keputusan pada subset acak dari data dan memberikan hasil prediksi dengan cara mayoritas voting pada prediksi dari setiap pohon.

    **Metode yang Digunakan:**  
    Model **Random Forest** ini dilatih menggunakan data fitur dari dataset kanker payudara dan label kelasnya ('M' atau 'B'). Model ini kemudian diuji untuk mengukur akurasi prediksinya. Selain itu, untuk memahami hubungan antar fitur, dilakukan **Principal Component Analysis (PCA)** untuk mengurangi dimensi data dan memvisualisasikan hubungan antar data dalam dua dimensi.
    """)

    # Menampilkan akurasi model
    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    # # Confusion matrix
    # st.subheader("Confusion Matrix")
    # st.image(confusion_matrix_image, caption="Confusion Matrix")

    # # Hasil PCA plot
    # st.subheader("PCA Visualization")
    # st.image(pca_image, caption="PCA Clustering Visualization")

if page == "Prediksi Model":
    st.title('Model Klasifikasi Tumor Payudara - Prediksi')

    st.subheader("Masukkan Fitur untuk Prediksi")

    # Input fields for 30 features
    features = [
        "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1",
        "concave_points1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", 
        "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", "symmetry2", 
        "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", "smoothness3", 
        "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
    ]

    # Create input form for 30 features
    feature_values = []
    for feature in features:
        value = st.number_input(f"Masukkan nilai untuk {feature}", min_value=0.0, step=0.1)
        feature_values.append(value)

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([feature_values], columns=features)

    # Predict based on the input data
    if st.button("Prediksi"):
        prediction = rf_model.predict(input_data)
        predicted_class = label_encoder.inverse_transform(prediction)
        st.subheader(f"Hasil Prediksi: {predicted_class[0]}")