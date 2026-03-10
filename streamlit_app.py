# ============================================
# DIAMOND PRICE PREDICTION - STREAMLIT APP
# VERSI AMAN - HANDLE ERROR INSTALL
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from pathlib import Path
import requests
import zipfile
import io

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
# CEK DAN INSTALL LIBRARY YANG DIBUTUHKAN
# ============================================
def check_and_install_libraries():
    """Cek apakah library sudah terinstall"""
    
    required_libs = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'joblib': 'joblib'
    }
    
    missing_libs = []
    
    for lib_name, pip_name in required_libs.items():
        try:
            __import__(lib_name)
        except ImportError:
            missing_libs.append(pip_name)
    
    if missing_libs:
        st.warning(f"⚠️ Library berikut belum terinstall: {', '.join(missing_libs)}")
        st.info("📦 Install manual di terminal dengan: pip install " + " ".join(missing_libs))
        return False
    
    return True

# ============================================
# FUNGSI DOWNLOAD DARI GOOGLE DRIVE (TANPA GDOWN)
# ============================================
def download_from_drive(file_id, output_path):
    """
    Download file dari Google Drive menggunakan requests
    Alternatif jika gdown bermasalah
    """
    
    # URL download dari Google Drive
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle confirmation token untuk file besar
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Download file
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    
    return output_path

# ============================================
# FUNGSI DOWNLOAD ALL MODELS
# ============================================
@st.cache_resource
def download_all_models():
    """
    Download semua model dari Google Drive
    """
    
    with st.spinner("📥 Mendownload model dari Google Drive..."):
        
        # Buat folder temporary
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir(exist_ok=True)
        
        # DAFTAR FILE ID GOOGLE DRIVE
        # GANTI DENGAN FILE ID ANDA
        file_ids = {
            'knn_model.pkl': '1KgI3-9AUlklKorCD0FpaGmwG55YXjyoD',      # GANTI
            'rf_model.pkl': '1HtTeRH4nrIQVhqPm2ZZHxfaPfW6sKA5b',       # GANTI
            'xgb_model.pkl': '1zwsDShlkKLDJb2WjSiuC_9qMC-ANinHX',      # GANTI
            'scaler.pkl': '19wmvErUcLiqJhpPQ0iJFXxVah8tuxaRg',         # GANTI
            'le_cut.pkl': '1y9FLGI6H2t_D7qauGVpZGCrYsbqpwSqW',         # GANTI
            'le_color.pkl': '1usHXz-uS0wLE4l99HlHX4EVz8DFcu_6w',       # GANTI
            'le_clarity.pkl': '1SoI1HQgfjxkmdWnlAooN-AYE0H9ZS_Tj',     # GANTI
        
        }
        
        downloaded = {}
        progress_bar = st.progress(0)
        total_files = len(file_ids)
        
        for i, (filename, file_id) in enumerate(file_ids.items()):
            try:
                # Tampilkan status
                st.caption(f"📥 Downloading {filename}...")
                
                # Path untuk menyimpan
                file_path = models_dir / filename
                
                # Download file
                download_from_drive(file_id, str(file_path))
                
                # Load file
                model_name = filename.replace('.pkl', '')
                downloaded[model_name] = joblib.load(file_path)
                
                # Update progress
                progress_bar.progress((i + 1) / total_files)
                
            except Exception as e:
                st.error(f"❌ Gagal download {filename}: {e}")
                return None
        
        progress_bar.empty()
        st.success("✅ Semua model berhasil didownload!")
        
        return downloaded

# ============================================
# FUNGSI UPLOAD MANUAL
# ============================================
def manual_upload():
    """Upload file model secara manual"""
    
    st.info("📁 Upload file model satu per satu")
    
    uploaded = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        knn = st.file_uploader("knn_model.pkl", type=['pkl'])
        if knn:
            uploaded['knn_model'] = joblib.load(knn)
        
        rf = st.file_uploader("rf_model.pkl", type=['pkl'])
        if rf:
            uploaded['rf_model'] = joblib.load(rf)
        
        xgb = st.file_uploader("xgb_model.pkl", type=['pkl'])
        if xgb:
            uploaded['xgb_model'] = joblib.load(xgb)
        
        scaler = st.file_uploader("scaler.pkl", type=['pkl'])
        if scaler:
            uploaded['scaler'] = joblib.load(scaler)
    
    with col2:
        le_cut = st.file_uploader("le_cut.pkl", type=['pkl'])
        if le_cut:
            uploaded['le_cut'] = joblib.load(le_cut)
        
        le_color = st.file_uploader("le_color.pkl", type=['pkl'])
        if le_color:
            uploaded['le_color'] = joblib.load(le_color)
        
        le_clarity = st.file_uploader("le_clarity.pkl", type=['pkl'])
        if le_clarity:
            uploaded['le_clarity'] = joblib.load(le_clarity)
        
        metrics = st.file_uploader("metrics.pkl", type=['pkl'])
        if metrics:
            uploaded['metrics'] = joblib.load(metrics)
    
    required = ['knn_model', 'rf_model', 'xgb_model', 'scaler', 'le_cut', 'le_color', 'le_clarity']
    
    if all(r in uploaded for r in required):
        st.success("✅ Semua model siap!")
        return uploaded
    
    return None

# ============================================
# FUNGSI ENCODING INPUT
# ============================================
def encode_input(models, cut, color, clarity):
    """Encode categorical variables"""
    
    cut_map = {
        'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4
    }
    
    color_map = {
        'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6
    }
    
    clarity_map = {
        'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 
        'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7
    }
    
    # Jika encoder tersedia, gunakan itu
    if 'le_cut' in models:
        cut_enc = models['le_cut'].transform([cut])[0]
    else:
        cut_enc = cut_map[cut]
    
    if 'le_color' in models:
        color_enc = models['le_color'].transform([color])[0]
    else:
        color_enc = color_map[color]
    
    if 'le_clarity' in models:
        clarity_enc = models['le_clarity'].transform([clarity])[0]
    else:
        clarity_enc = clarity_map[clarity]
    
    return cut_enc, color_enc, clarity_enc

# ============================================
# MAIN APP
# ============================================
st.title("💎 Diamond Price Prediction")
st.markdown("Prediksi harga diamond menggunakan 3 algoritma Machine Learning")

# Sidebar untuk pilihan
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    option = st.radio(
        "Pilih metode load model:",
        ["🌐 Download dari Google Drive", "📁 Upload Manual"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Petunjuk")
    
    if option == "🌐 Download dari Google Drive":
        st.info("Model akan otomatis didownload dari Google Drive")
    else:
        st.info("Upload file .pkl satu per satu")

# Load models
models = None

if option == "🌐 Download dari Google Drive":
    models = download_all_models()
else:
    models = manual_upload()

# ============================================
# FORM PREDIKSI
# ============================================
if models:
    st.markdown("---")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        carat = st.number_input("💎 Carat", 0.1, 5.0, 0.5, 0.1)
        cut = st.selectbox("✂️ Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        depth = st.number_input("📏 Depth %", 40.0, 80.0, 61.5, 0.1)
    
    with col2:
        color = st.selectbox("🎨 Color", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("🔍 Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
        table = st.number_input("📐 Table %", 40.0, 80.0, 55.0, 0.1)
    
    with col3:
        x = st.number_input("📏 X (mm)", 1.0, 10.0, 3.95, 0.01)
        y = st.number_input("📏 Y (mm)", 1.0, 10.0, 3.98, 0.01)
        z = st.number_input("📏 Z (mm)", 1.0, 10.0, 2.43, 0.01)
    
    algorithm = st.selectbox(
        "🤖 Pilih Algoritma",
        ["KNN", "Random Forest", "XGBoost"],
        index=1
    )
    
    # Tombol prediksi
    if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
        try:
            # Encode
            cut_enc, color_enc, clarity_enc = encode_input(models, cut, color, clarity)
            
            # Buat dataframe
            feature_cols = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
                           'depth', 'table', 'x', 'y', 'z']
            
            input_data = pd.DataFrame([[carat, cut_enc, color_enc, clarity_enc,
                                       depth, table, x, y, z]], columns=feature_cols)
            
            # Prediksi
            if algorithm == "KNN":
                if 'scaler' in models:
                    input_scaled = models['scaler'].transform(input_data)
                    pred = models['knn_model'].predict(input_scaled)[0]
                else:
                    st.error("❌ Scaler tidak ditemukan untuk KNN")
                    st.stop()
            elif algorithm == "Random Forest":
                pred = models['rf_model'].predict(input_data)[0]
            else:
                pred = models['xgb_model'].predict(input_data)[0]
            
            pred = max(0, pred)
            
            # Tampilkan hasil
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Input Carat", f"{carat} ct")
            with col2:
                st.metric("Cut - Color - Clarity", f"{cut} - {color} - {clarity}")
            with col3:
                st.metric("Algoritma", algorithm)
            
            # Hasil utama
            st.markdown("### 💰 Hasil Prediksi")
            
            # Card untuk hasil
            st.markdown(f"""
            <div style="
                text-align: center; 
                padding: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; 
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            ">
                <h2 style="color: white; margin: 0;">Harga Diamond</h2>
                <h1 style="color: white; font-size: 72px; margin: 10px 0;">${pred:,.2f}</h1>
                <p style="color: white; font-size: 18px;">Menggunakan algoritma: {algorithm}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Tampilkan metrics jika ada
    if 'metrics' in models:
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        
        metrics_data = models['metrics']
        
        cols = st.columns(3)
        for i, (model, metric) in enumerate(metrics_data.items()):
            with cols[i]:
                st.markdown(f"**{model}**")
                st.metric("R² Score", f"{metric['R2']:.4f}")
                st.metric("RMSE", f"${metric['RMSE']:,.2f}")
                st.metric("MAE", f"${metric['MAE']:,.2f}")

else:
    st.info("👈 Silakan pilih metode dan load model di sidebar")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>© 2024 Diamond Price Prediction | Tugas Machine Learning</p>
        <p style="font-size: 12px;">💡 Gunakan Google Drive untuk file model >100MB</p>
    </div>
    """,
    unsafe_allow_html=True
)
