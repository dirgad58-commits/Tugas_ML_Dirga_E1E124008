# ============================================
# DIAMOND PRICE PREDICTION - STREAMLIT APP
# VERSI FINAL - KOMPATIBEL PYTHON 3.11
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import tempfile
from pathlib import Path
import requests

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
# CEK VERSI PYTHON
# ============================================
import sys
st.sidebar.info(f"🐍 Python version: {sys.version}")

# ============================================
# FILE ID DARI GOOGLE DRIVE
# ============================================
FILE_IDS = {
    'knn_model.pkl': '1KgI3-9AUlklKorCD0FpaGmwG55YXjyoD',
    'rf_model.pkl': '1HtTeRH4nrIQVhqPm2ZZHxfaPfW6sKA5b',
    'xgb_model.pkl': '1zwsDShlkKLDJb2WjSiuC_9qMC-ANinHX',
    'scaler.pkl': '19wmvErUcLiqJhpPQ0iJFXxVah8tuxaRg',
    'le_cut.pkl': '1y9FLGI6H2t_D7qauGVpZGCrYsbqpwSqW',
    'le_color.pkl': '1usHXz-uS0wLE4l99HlHX4EVz8DFcu_6w',
    'le_clarity.pkl': '1SoI1HQgfjxkmdWnlAooN-AYE0H9ZS_Tj'
}

# ============================================
# FUNGSI DOWNLOAD MODEL
# ============================================
@st.cache_resource
def download_models():
    """
    Download semua model dari Google Drive
    """
    with st.spinner("📥 Mendownload model dari Google Drive..."):
        
        # Buat folder temporary
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir(exist_ok=True)
        
        downloaded = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(FILE_IDS)
        
        for i, (filename, file_id) in enumerate(FILE_IDS.items()):
            try:
                status_text.text(f"📥 Downloading {filename}...")
                
                # URL download
                url = f"https://drive.google.com/uc?id={file_id}"
                output_path = models_dir / filename
                
                # Download file
                gdown.download(url, str(output_path), quiet=True)
                
                # Load file
                model_name = filename.replace('.pkl', '')
                downloaded[model_name] = joblib.load(output_path)
                
                # Update progress
                progress_bar.progress((i + 1) / total_files)
                
            except Exception as e:
                st.error(f"❌ Gagal download {filename}: {e}")
                return None
        
        progress_bar.empty()
        status_text.text("✅ Semua model berhasil didownload!")
        
        return downloaded

# ============================================
# FUNGSI ENCODING MANUAL (FALLBACK)
# ============================================
def encode_manual(cut, color, clarity):
    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_map = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
    clarity_map = {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7}
    
    return cut_map[cut], color_map[color], clarity_map[clarity]

# ============================================
# MAIN APP
# ============================================
st.title("💎 Diamond Price Prediction")
st.markdown("Prediksi harga diamond menggunakan 3 algoritma Machine Learning")

# Inisialisasi session state
if 'models' not in st.session_state:
    st.session_state.models = None

# Tombol download
col1, col2 = st.columns(2)
with col1:
    if st.button("📥 Download Models dari Google Drive", type="primary"):
        models = download_models()
        if models:
            st.session_state.models = models
            st.rerun()

with col2:
    if st.button("🎮 Gunakan Mode Demo"):
        st.session_state.models = "demo"
        st.rerun()

# ============================================
# FORM INPUT (TAMPIL SETELAH MODEL READY)
# ============================================
if st.session_state.models is not None:
    st.success("✅ Model siap digunakan!")
    
    # Sidebar untuk input
    with st.sidebar:
        st.header("📋 Input Karakteristik")
        
        carat = st.number_input("💎 Carat", 0.1, 5.0, 0.5, 0.1)
        cut = st.selectbox("✂️ Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
        color = st.selectbox("🎨 Color", ["D", "E", "F", "G", "H", "I", "J"], index=3)
        clarity = st.selectbox("🔍 Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"], index=4)
        
        col1, col2 = st.columns(2)
        with col1:
            depth = st.number_input("📏 Depth %", 40.0, 80.0, 61.5, 0.1)
            x = st.number_input("📏 X (mm)", 1.0, 10.0, 3.95, 0.01)
        with col2:
            table = st.number_input("📐 Table %", 40.0, 80.0, 55.0, 0.1)
            y = st.number_input("📏 Y (mm)", 1.0, 10.0, 3.98, 0.01)
        
        z = st.number_input("📏 Z (mm)", 1.0, 10.0, 2.43, 0.01)
        
        algorithm = st.selectbox("🤖 Algoritma", ["KNN", "Random Forest", "XGBoost"], index=1)
        
        predict_btn = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)
    
    # ============================================
    # PREDIKSI
    # ============================================
    if predict_btn:
        try:
            # Encode categorical
            if st.session_state.models != "demo":
                cut_enc = st.session_state.models['le_cut'].transform([cut])[0]
                color_enc = st.session_state.models['le_color'].transform([color])[0]
                clarity_enc = st.session_state.models['le_clarity'].transform([clarity])[0]
            else:
                cut_enc, color_enc, clarity_enc = encode_manual(cut, color, clarity)
            
            # Buat dataframe
            feature_cols = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
                           'depth', 'table', 'x', 'y', 'z']
            
            input_data = pd.DataFrame([[carat, cut_enc, color_enc, clarity_enc,
                                       depth, table, x, y, z]], columns=feature_cols)
            
            # Prediksi
            if st.session_state.models != "demo":
                if algorithm == "KNN":
                    input_scaled = st.session_state.models['scaler'].transform(input_data)
                    prediction = st.session_state.models['knn_model'].predict(input_scaled)[0]
                elif algorithm == "Random Forest":
                    prediction = st.session_state.models['rf_model'].predict(input_data)[0]
                else:
                    prediction = st.session_state.models['xgb_model'].predict(input_data)[0]
            else:
                # Mode demo - rumus sederhana
                prediction = 500 + (carat * 3000) + (cut_enc * 200) + (color_enc * 100) + (clarity_enc * 150)
            
            prediction = max(0, prediction)
            
            # Tampilkan hasil
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Carat", f"{carat} ct")
            with col2:
                st.metric("Cut", cut)
            with col3:
                st.metric("Color", color)
            with col4:
                st.metric("Clarity", clarity)
            
            st.markdown("### 💰 Hasil Prediksi")
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            ">
                <h1 style="color: white; font-size: 72px;">${prediction:,.2f}</h1>
                <p style="color: white;">Algoritma: {algorithm}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.models == "demo":
                st.info("ℹ️ Mode Demo: Hasil adalah estimasi sederhana")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

else:
    st.info("👈 Klik tombol di atas untuk memulai")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("© 2024 Diamond Price Prediction | Tugas Machine Learning")
