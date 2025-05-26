import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

st.set_page_config(
    page_title="Dasbor Analisis Tumor Payudara", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_placeholder_plot(title="Contoh Plot"): 
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title(title)
    ax.set_xlabel("Sumbu X")
    ax.set_ylabel("Sumbu Y")
    ax.grid(True)
    return fig

def main():
    model = joblib.load('clf_model_breast_cancer.pkl')

    with st.sidebar:
        st.title("📁 Kontrol Input")
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Upload file CSV:",
            type=["csv"],
            help="Upload file CSV dengan data observasi tumor."
        )
        st.markdown("---")
        st.info("Pastikan Anda memasukkan data dengan fitur yang tepat untuk memperoleh hasil prediksi terbaik!")

    st.title("Dashboard analisis tumor payudara")
    st.markdown("""
        Selamat Datang! Dashboard ini dapat memberikan prediksi berdasarkan data Anda dan menunjukkan performa model yang kami gunakan.
    """)

    tab1_predict, tab2_results = st.tabs([
        "📈 Prediksi dari data",
        "💡 Pemodelan & Clustering" 
    ])

    with tab1_predict:
        st.header("🔮 Analisis dan prediksi data")
        st.markdown("Upload file CSV melalui kontrol di samping layar Anda.")

        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' berhasil diunggah!")
            try:
                df_input = pd.read_csv(uploaded_file)
                df_input_scaled = RobustScaler().fit_transform(df_input)
                
                st.subheader("📋 Preview Data") 
                st.dataframe(df_input.head(), use_container_width=True)

                st.markdown("---")
                
                num_rows = len(df_input)

                preds = model.predict(df_input_scaled)
                preds_string = ['Jinak' if pred == 0 else 'Ganas' for pred in preds]
                pred_probs = model.predict_proba(df_input_scaled)

                results = []

                for i in range(num_rows):
                    results.append({
                        'Observasi_Ke': i + 1,
                        'Prediksi': preds_string[i],
                        'Probabilitas Ganas': f'{pred_probs[i][1]*100:.2f}%'
                    })

                results_df = pd.DataFrame(results)

                st.subheader("📊 Hasil Prediksi")
                st.dataframe(results_df, use_container_width=True)

                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_download = convert_df_to_csv(results_df)
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)",
                    data=csv_download,
                    file_name=f"prediksi_{uploaded_file.name}.csv",
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"⚠️ Gagal saat memproses file: {e}")
                st.warning("Pastikan format CSV Anda sudah benar.") 
        else:
            st.info("Menunggu unggahan file CSV...") 

    with tab2_results:
        st.header("🔬 Evaluasi Model")  
        st.markdown("Jelajahi performa model prediksi dan analisis clustering.") 
        st.markdown("---")

        st.subheader("🥇 Evaluasi Model Regresi Logistik (Statis)") 
        st.markdown("Metrik performa dari model yang sudah dilatih pada set data uji.") 

        st.metric(label="Akurasi Set Uji", value="99.12%")  
        st.caption("Akurasi: (TP+TN)/(TP+TN+FP+FN)")

        st.write("**Matriks Konfusi:**") 
        st.image("confusion_matrix.png", caption="Matriks Konfusi Model")
        st.caption("Menampilkan True Positives, False Positives, True Negatives, False Negatives.") 

        st.write("**Perbandingan ROC AUC:**") 
        st.image("ROC_AUC_curve.png", caption="Kurva ROC AUC")
        st.caption("Menampilkan Perbadingan ROC Curve dan Nilai AUC Model Awal dan Tuned.")

        st.subheader("📝 Laporan Klasifikasi (Statis)")  
        report_text = """
        **Performa Keseluruhan:**
        ----------------------------------
        Kelas        | Presisi | Recall  | F1-Score
        ----------------------------------
        Jinak        | 0.99    | 1.00    | 0.99
        ----------------------------------
        Ganas        | 1.00    | 0.98    | 0.99
        ----------------------------------
        """
        st.markdown(report_text)
        st.caption("Detail presisi, recall, dan F1-score per kelas.")
                
        st.markdown("---")
        
        with st.container(border=True):
            st.subheader("🧩 Analisis Clustering (Statis)")  
            st.markdown("Hasil clustering K-Means divisualisasikan dengan PCA pada fitur dari data observasi.")

            st.metric(label=f"Skor Silhouette", value="0.5462", help="Rentang dari -1 hingga 1. Semakin tinggi semakin baik.")  
            st.caption("Mengukur seberapa mirip suatu objek dengan clusternya sendiri dibandingkan dengan cluster lain.")  

            st.image("cluster.png", caption="Hasil Klasterisasi Data setelah PCA")
            st.caption("Menampilkan Hasil Klasterisasi Data Dengan Setelah PCA.")

    st.markdown("---")
    st.caption("© 2025 Kelompok 12 - Dasbor Analisis Tumor Payudara")  


if __name__ == '__main__':
    main()