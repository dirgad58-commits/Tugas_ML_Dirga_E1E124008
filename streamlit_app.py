# ============================================
# DIAMOND PRICE PREDICTION - STREAMLIT VERSION
# 3 ALGORITMA: KNN, Random Forest, XGBoost
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_models():
    """Load semua model dari folder models/"""
    try:
        knn = joblib.load('models/knn_model.pkl')
        rf = joblib.load('models/rf_model.pkl')
        xgb = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_cut = joblib.load('models/le_cut.pkl')
        le_color = joblib.load('models/le_color.pkl')
        le_clarity = joblib.load('models/le_clarity.pkl')
        
        return {
            'KNN': knn,
            'Random Forest': rf,
            'XGBoost': xgb,
            'scaler': scaler,
            'le_cut': le_cut,
            'le_color': le_color,
            'le_clarity': le_clarity
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Pastikan folder models/ berisi semua file .pkl")
        return None

# Load models
models = load_models()

# ============================================
# SIDEBAR - INPUT
# ============================================
st.sidebar.title("💎 Input Karakteristik Diamond")
st.sidebar.markdown("---")

# Input form di sidebar
with st.sidebar.form("prediction_form"):
    carat = st.number_input("Carat (Berat)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    cut = st.selectbox("Cut (Potongan)", 
                       ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                       index=4)
    
    color = st.selectbox("Color (Warna)", 
                        ["D", "E", "F", "G", "H", "I", "J"],
                        index=3)
    
    clarity = st.selectbox("Clarity (Kejernihan)", 
                          ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
                          index=4)
    
    col1, col2 = st.columns(2)
    with col1:
        depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5)
        x = st.number_input("X (mm)", min_value=1.0, max_value=10.0, value=3.95)
        y = st.number_input("Y (mm)", min_value=1.0, max_value=10.0, value=3.98)
    
    with col2:
        table = st.number_input("Table (%)", min_value=40.0, max_value=80.0, value=55.0)
        z = st.number_input("Z (mm)", min_value=1.0, max_value=10.0, value=2.43)
    
    algorithm = st.selectbox("Pilih Algoritma",
                            ["KNN", "Random Forest", "XGBoost"],
                            index=1)
    
    submitted = st.form_submit_button("🔮 Prediksi Harga", use_container_width=True)

# ============================================
# MAIN PAGE
# ============================================
st.title("💎 Diamond Price Prediction")
st.markdown("Prediksi harga diamond menggunakan 3 algoritma Machine Learning")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediksi", "📈 Perbandingan Model", "ℹ️ About"])

with tab1:
    if models is not None and submitted:
        try:
            # Encode categorical
            cut_enc = models['le_cut'].transform([cut])[0]
            color_enc = models['le_color'].transform([color])[0]
            clarity_enc = models['le_clarity'].transform([clarity])[0]
            
            # Buat dataframe
            feature_cols = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
                           'depth', 'table', 'x', 'y', 'z']
            
            input_data = pd.DataFrame([[carat, cut_enc, color_enc, clarity_enc,
                                       depth, table, x, y, z]], columns=feature_cols)
            
            # Prediksi
            if algorithm == 'KNN':
                input_scaled = models['scaler'].transform(input_data)
                prediction = models[algorithm].predict(input_scaled)[0]
            else:
                prediction = models[algorithm].predict(input_data)[0]
            
            # Tampilkan hasil
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Carat", f"{carat} ct")
            with col2:
                st.metric("Cut", cut)
            with col3:
                st.metric("Color", color)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clarity", clarity)
            with col2:
                st.metric("Depth", f"{depth}%")
            with col3:
                st.metric("Table", f"{table}%")
            
            # Hasil prediksi
            st.markdown("---")
            st.markdown("### 💰 Hasil Prediksi")
            
            # Buat gauge chart
            fig = go.Figure(go.Indicator(
                mode = "number+delta",
                value = prediction,
                number = {'prefix': "$", 'font': {'size': 60}},
                title = {'text': f"Predicted Price ({algorithm})"},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan detail
            st.success(f"💰 Harga prediksi: **${prediction:,.2f}**")
            st.info(f"🤖 Algoritma yang digunakan: **{algorithm}**")
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    elif models is None:
        st.warning("⚠️ Model tidak ditemukan. Pastikan folder models/ ada dan berisi file .pkl")
        st.code("""
        Struktur folder yang benar:
        📁 models/
        ├── knn_model.pkl
        ├── rf_model.pkl
        ├── xgb_model.pkl
        ├── scaler.pkl
        ├── le_cut.pkl
        ├── le_color.pkl
        └── le_clarity.pkl
        """)
    
    else:
        st.info("👈 Silakan isi form di sidebar dan klik 'Prediksi Harga'")

with tab2:
    st.markdown("### 📊 Perbandingan Performa Model")
    
    # Data metrics (gunakan dari file metrics.pkl jika ada)
    try:
        metrics = joblib.load('models/metrics.pkl')
        
        # Buat dataframe
        df_metrics = pd.DataFrame(metrics).T
        df_metrics = df_metrics.reset_index().rename(columns={'index': 'Model'})
        
        # Tampilkan dalam 3 kolom
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(df_metrics, x='Model', y='R2', 
                        title='R² Score (Semakin Tinggi Semakin Baik)',
                        color='Model', color_discrete_sequence=['#667eea', '#764ba2', '#84fab0'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_metrics, x='Model', y='RMSE',
                        title='RMSE ($) (Semakin Rendah Semakin Baik)',
                        color='Model', color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(df_metrics, x='Model', y='MAE',
                        title='MAE ($) (Semakin Rendah Semakin Baik)',
                        color='Model', color_discrete_sequence=['#feca57', '#ff9ff3', '#54a0ff'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabel detail
        st.markdown("### 📋 Detail Metrics")
        st.dataframe(df_metrics.style.format({
            'R2': '{:.4f}',
            'RMSE': '${:,.2f}',
            'MAE': '${:,.2f}',
            'MSE': '{:,.2f}'
        }), use_container_width=True)
        
    except:
        st.info("ℹ️ File metrics.pkl tidak ditemukan. Jalankan training terlebih dahulu.")
        
        # Data dummy untuk demo
        df_dummy = pd.DataFrame({
            'Model': ['KNN', 'Random Forest', 'XGBoost'],
            'R2': [0.92, 0.98, 0.97],
            'RMSE': [800, 450, 500],
            'MAE': [450, 250, 280],
            'MSE': [640000, 202500, 250000]
        })
        
        st.dataframe(df_dummy)

with tab3:
    st.markdown("""
    ### ℹ️ Tentang Aplikasi
    
    Aplikasi ini dibuat untuk memprediksi harga diamond menggunakan 3 algoritma machine learning:
    
    #### 🤖 Algoritma yang Digunakan:
    1. **K-Nearest Neighbors (KNN)**
       - Berbasis jarak ke tetangga terdekat
       - Parameter: n_neighbors=7, weights='distance'
    
    2. **Random Forest**
       - Ensemble of decision trees
       - Parameter: n_estimators=100, max_depth=20
    
    3. **XGBoost**
       - Gradient boosting algorithm
       - Parameter: n_estimators=100, max_depth=6, learning_rate=0.1
    
    #### 📊 Dataset
    - **Sumber**: diamond.csv
    - **Jumlah data**: 53,940 sampel
    - **Fitur**: 9 features (carat, cut, color, clarity, depth, table, x, y, z)
    - **Target**: price (harga dalam USD)
    
    #### 📈 Metrik Evaluasi
    - **R² Score**: Seberapa baik model menjelaskan data (0-1)
    - **RMSE**: Root Mean Square Error
    - **MAE**: Mean Absolute Error
    
    #### 🛠️ Teknologi
    - Python 3.9
    - Streamlit
    - Scikit-learn
    - XGBoost
    - Plotly
    
    #### 👨‍💻 Pembuat
    Tugas Machine Learning
    
    #### 📅 Tahun
    2024
    """)

# Footer
st.markdown("---")
st.markdown("© 2024 Diamond Price Prediction | Tugas Machine Learning")
