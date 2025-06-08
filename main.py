import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from inferenceLLM import getInterpretation

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

    tab1_predict, tab2_clustering, tab3_cluster_interpretation = st.tabs([
        "📈 Prediksi dari data",
        "📃 Clustering",
        "💡 Interpretasi Cluster" 
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

                # Tambahkan metrik jumlah jinak, ganas, dan total observasi
                col1, col2, col3 = st.columns(3)
                total_jinak = (results_df['Prediksi'] == 'Jinak').sum()
                total_ganas = (results_df['Prediksi'] == 'Ganas').sum()
                total_obs = len(results_df)
                col1.metric("Total Jinak", total_jinak)
                col2.metric("Total Ganas", total_ganas)
                col3.metric("Total Observasi", total_obs)

                st.markdown("---")
                st.markdown("### pilih baris hasil prediksi untuk mendapatkan interpretasi analitis:")
                selected_row = st.selectbox(
                    "Pilih Observasi untuk Interpretasi:",
                    options=results_df.index + 1,
                    format_func=lambda x: f"Observasi ke-{x} ({results_df.loc[x-1, 'Prediksi']}, Prob Ganas: {results_df.loc[x-1, 'Probabilitas Ganas']})"
                )
                if selected_row:
                    idx = selected_row - 1
                    pred_val = 1 if results_df.loc[idx, 'Prediksi'] == 'Ganas' else 0
                    prob_ganas = float(results_df.loc[idx, 'Probabilitas Ganas'].replace('%','')) / 100
                    if st.button(f"Dapatkan Interpretasi untuk Observasi ke-{selected_row}"):
                        with st.spinner("Mengambil interpretasi dari LLM..."):
                            llm_result = getInterpretation(pred_val, prob_ganas)
                        st.success("Interpretasi:")
                        st.write(llm_result)

            except Exception as e:
                st.error(f"⚠️ Gagal saat memproses file: {e}")
                st.warning("Pastikan format CSV Anda sudah benar.") 
        else:
            st.info("Menunggu unggahan file CSV...") 

    with tab2_clustering:
        st.header("📃 Clustering Data Observasi")
        st.markdown("Lakukan segmentasi data observasi menggunakan K-Means Clustering.")
        if uploaded_file is not None:
            try:
                model_kmeans = joblib.load('kmeans_model.pkl')
                cluster_labels = model_kmeans.predict(df_input_scaled)
                results_df_cluster = results_df.copy()
                results_df_cluster['Cluster'] = cluster_labels
                st.subheader("📊 Hasil Prediksi + Cluster")
                st.dataframe(results_df_cluster, use_container_width=True)
                col1, col2 = st.columns(2)
                col1.metric("Total Cluster 0", (results_df_cluster['Cluster'] == 0).sum())
                col2.metric("Total Cluster 1", (results_df_cluster['Cluster'] == 1).sum())
                st.info("*Cluster 1 cenderung berisi observasi jinak, Cluster 0 cenderung berisi observasi ganas.")
            except Exception as e:
                st.error(f"Gagal melakukan clustering: {e}")
        else:
            st.info("Unggah file CSV terlebih dahulu untuk melakukan clustering.")

    with tab3_cluster_interpretation:
        st.header("💡 Interpretasi Cluster")
        st.markdown("""
        **Cluster 0**: 
        - Cenderung berisi observasi dengan label asli ganas.
        - Distribusi data lebih tersebar dan beragam untuk setiap fitur.
        - Menunjukkan variasi yang lebih tinggi pada data observasi sel tumor ganas.
        
        **Cluster 1**:
        - Cenderung berisi observasi dengan label asli jinak.
        - Distribusi data lebih seragam dan outlier lebih sedikit.
        - Karakteristik fitur lebih homogen.
        """)
        st.image("cluster.png", caption="Visualisasi Hasil Clustering (PCA)")
        st.caption("Visualisasi hasil clustering K-Means pada data observasi menggunakan PCA.")
        
    st.caption("© 2025 Kelompok 12 - Dasbor Analisis Tumor Payudara")  


if __name__ == '__main__':
    main()