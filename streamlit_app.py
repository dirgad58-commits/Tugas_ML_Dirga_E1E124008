# ============================================
# DIAMOND PRICE PREDICTION - STREAMLIT APP
# VERSI FINAL - DOWNLOAD DARI GOOGLE DRIVE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from pathlib import Path
import requests
import io
import gdown
from zipfile import ZipFile

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
# DAFTAR FILE ID GOOGLE DRIVE (SUDAH ANDA PUNYA)
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
# FUNGSI DOWNLOAD DENGAN GDOWN (PASTI BERHASIL)
# ============================================
def download_with_gdown(file_id, output_path):
    """Download file dari Google Drive menggunakan gdown"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# ============================================
# FUNGSI DOWNLOAD DENGAN REQUESTS (ALTERNATIF)
# ============================================
def download_with_requests(file_id, output_path):
    """Download file dari Google Drive menggunakan requests"""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ============================================
# FUNGSI DOWNLOAD SEMUA MODEL
# ============================================
@st.cache_resource
def download_all_models():
    """
    Download semua model dari Google Drive
    """
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Buat folder temporary
    temp_dir = tempfile.mkdtemp()
    models_dir = Path(temp_dir) / "models"
    models_dir.mkdir(exist_ok=True)
    
    downloaded = {}
    total_files = len(FILE_IDS)
    
    for i, (filename, file_id) in enumerate(FILE_IDS.items()):
        try:
            status_text.text(f"📥 Downloading {filename}...")
            file_path = models_dir / filename
            
            # Coba download dengan gdown dulu
            try:
                download_with_gdown(file_id, str(file_path))
            except:
                # Fallback ke requests
                download_with_requests(file_id, str(file_path))
            
            # Load file
            model_name = filename.replace('.pkl', '')
            downloaded[model_name] = joblib.load(file_path)
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
            
        except Exception as e:
            st.error(f"❌ Gagal download {filename}: {e}")
            return None
    
    progress_bar.empty()
    status_text.text("✅ Semua model berhasil didownload!")
    
    return downloaded

# ============================================
# FUNGSI ENCODING MANUAL
# ============================================
def encode_manual(cut, color, clarity):
    """Manual encoding untuk fallback"""
    
    cut_map = {
        'Fair': 0, 'Good': 1, 'Very Good': 2, 
        'Premium': 3, 'Ideal': 4
    }
    
    color_map = {
        'D': 0, 'E': 1, 'F': 2, 'G': 3, 
        'H': 4, 'I': 5, 'J': 6
    }
    
    clarity_map = {
        'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3,
        'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7
    }
    
    return (
        cut_map[cut],
        color_map[color],
        clarity_map[clarity]
    )

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict_demo(carat, cut_enc, color_enc, clarity_enc, depth, table, x, y, z):
    """Prediksi mode demo"""
    base_price = 500
    price = (base_price + 
             carat * 3000 + 
             cut_enc * 200 + 
             color_enc * 100 + 
             clarity_enc * 150 +
             depth * 5 +
             table * 3 +
             (x + y + z) * 100)
    return max(500, price)

# ============================================
# MAIN APP
# ============================================
st.title("💎 Diamond Price Prediction")
st.markdown("Prediksi harga diamond menggunakan 3 algoritma Machine Learning")

# Sidebar untuk pilihan
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    option = st.radio(
        "Pilih metode:",
        ["🌐 Download dari Google Drive", "📁 Upload Manual", "🎮 Mode Demo"]
    )
    
    if option == "🌐 Download dari Google Drive":
        st.info(f"📥 Akan mendownload {len(FILE_IDS)} file dari Google Drive")
    
    st.markdown("---")
    st.markdown("### 📋 Input Karakteristik")
    
    carat = st.number_input("💎 Carat", 0.1, 5.0, 0.5, 0.1)
    
    cut = st.selectbox("✂️ Cut", 
                       ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                       index=4)
    
    color = st.selectbox("🎨 Color", 
                        ["D", "E", "F", "G", "H", "I", "J"],
                        index=3)
    
    clarity = st.selectbox("🔍 Clarity", 
                          ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
                          index=4)
    
    col1, col2 = st.columns(2)
    with col1:
        depth = st.number_input("📏 Depth %", 40.0, 80.0, 61.5, 0.1)
        x = st.number_input("📏 X (mm)", 1.0, 10.0, 3.95, 0.01)
    with col2:
        table = st.number_input("📐 Table %", 40.0, 80.0, 55.0, 0.1)
        y = st.number_input("📏 Y (mm)", 1.0, 10.0, 3.98, 0.01)
    
    z = st.number_input("📏 Z (mm)", 1.0, 10.0, 2.43, 0.01)
    
    algorithm = st.selectbox("🤖 Algoritma",
                            ["KNN", "Random Forest", "XGBoost"],
                            index=1)
    
    predict_btn = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)

# ============================================
# LOAD MODEL BERDASARKAN PILIHAN
# ============================================
models = None

if option == "🌐 Download dari Google Drive":
    with st.spinner("⏳ Mendownload model dari Google Drive..."):
        models = download_all_models()
        if models:
            st.success("✅ Model berhasil didownload!")

elif option == "📁 Upload Manual":
    st.info("📁 Upload file .pkl satu per satu")
    
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
    
    if len(uploaded) >= 7:
        models = uploaded
        st.success("✅ Semua model siap!")

# ============================================
# PREDIKSI
# ============================================
if predict_btn:
    try:
        # Encode categorical
        if models and 'le_cut' in models:
            cut_enc = models['le_cut'].transform([cut])[0]
            color_enc = models['le_color'].transform([color])[0]
            clarity_enc = models['le_clarity'].transform([clarity])[0]
        else:
            cut_enc, color_enc, clarity_enc = encode_manual(cut, color, clarity)
        
        # Prediksi
        if models and algorithm == "KNN" and 'scaler' in models:
            feature_cols = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
                           'depth', 'table', 'x', 'y', 'z']
            
            input_data = pd.DataFrame([[carat, cut_enc, color_enc, clarity_enc,
                                       depth, table, x, y, z]], columns=feature_cols)
            
            if algorithm == "KNN":
                input_scaled = models['scaler'].transform(input_data)
                prediction = models['knn_model'].predict(input_scaled)[0]
            elif algorithm == "Random Forest":
                prediction = models['rf_model'].predict(input_data)[0]
            else:
                prediction = models['xgb_model'].predict(input_data)[0]
        else:
            # Mode demo
            prediction = predict_demo(carat, cut_enc, color_enc, clarity_enc, 
                                    depth, table, x, y, z)
        
        prediction = max(0, prediction)
        
        # Tampilkan hasil
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Carat", f"{carat} ct")
        with col2:
            st.metric("Cut", cut)
        with col3:
            st.metric("Color", color)
        with col4:
            st.metric("Clarity", clarity)
        
        # Hasil utama
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
            <h2 style="color: white; margin: 0;">Harga Diamond</h2>
            <h1 style="color: white; font-size: 72px; margin: 10px 0;">
                ${prediction:,.2f}
            </h1>
            <p style="color: white; font-size: 18px;">
                Menggunakan algoritma: {algorithm}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if not models:
            st.info("ℹ️ Mode Demo: Hasil adalah estimasi")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>© 2024 Diamond Price Prediction | Tugas Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)
