# ============================================
# DIAMOND PRICE PREDICTION - STREAMLIT APP
# DENGAN DOWNLOAD MODEL DARI GOOGLE DRIVE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
import tempfile
from pathlib import Path

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
# FUNGSI DOWNLOAD MODEL DARI GOOGLE DRIVE
# ============================================
@st.cache_resource
def download_models_from_drive():
    """
    Download semua model dari Google Drive
    Menggunakan gdown untuk mengambil file dari shared link
    """
    
    # Buat progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Buat folder temporary untuk menyimpan model
    temp_dir = tempfile.mkdtemp()
    models_dir = Path(temp_dir) / "models"
    models_dir.mkdir(exist_ok=True)
    
    # DAFTAR FILE DAN LINK GOOGLE DRIVE
    # Cara mendapatkan link: 
    # 1. Upload file ke Google Drive
    # 2. Share file (dapatkan link sharing)
    # 3. Extract file ID dari link
    # Contoh link: https://drive.google.com/file/d/1ABC123xxx/view
    # File ID = 1ABC123xxx
    
    files_to_download = [
        {
            'name': 'knn_model.pkl',
            'url': 'https://drive.google.com/uc?id=1ABC123xxx',  # GANTI DENGAN ID ANDA
            'description': 'KNN Model'
        },
        {
            'name': 'rf_model.pkl',
            'url': 'https://drive.google.com/uc?id=1DEF456xxx',  # GANTI DENGAN ID ANDA
            'description': 'Random Forest Model'
        },
        {
            'name': 'xgb_model.pkl',
            'url': 'https://drive.google.com/uc?id=1GHI789xxx',  # GANTI DENGAN ID ANDA
            'description': 'XGBoost Model'
        },
        {
            'name': 'scaler.pkl',
            'url': 'https://drive.google.com/uc?id=1JKL012xxx',  # GANTI DENGAN ID ANDA
            'description': 'Standard Scaler'
        },
        {
            'name': 'le_cut.pkl',
            'url': 'https://drive.google.com/uc?id=1MNO345xxx',  # GANTI DENGAN ID ANDA
            'description': 'Label Encoder - Cut'
        },
        {
            'name': 'le_color.pkl',
            'url': 'https://drive.google.com/uc?id=1PQR678xxx',  # GANTI DENGAN ID ANDA
            'description': 'Label Encoder - Color'
        },
        {
            'name': 'le_clarity.pkl',
            'url': 'https://drive.google.com/uc?id=1STU901xxx',  # GANTI DENGAN ID ANDA
            'description': 'Label Encoder - Clarity'
        },
        {
            'name': 'metrics.pkl',
            'url': 'https://drive.google.com/uc?id=1VWX234xxx',  # GANTI DENGAN ID ANDA
            'description': 'Model Metrics'
        }
    ]
    
    downloaded_files = {}
    total_files = len(files_to_download)
    
    for i, file_info in enumerate(files_to_download):
        file_path = models_dir / file_info['name']
        
        status_text.text(f"📥 Downloading {file_info['description']}...")
        
        try:
            # Download file
            gdown.download(
                file_info['url'],
                str(file_path),
                quiet=False
            )
            
            # Load file ke memory
            downloaded_files[file_info['name'].replace('.pkl', '')] = joblib.load(file_path)
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
            
        except Exception as e:
            st.error(f"❌ Gagal download {file_info['name']}: {e}")
            return None
    
    status_text.text("✅ Semua model berhasil didownload!")
    progress_bar.empty()
    
    return downloaded_files

# ============================================
# FUNGSI LOAD MODEL ALTERNATIF (MANUAL UPLOAD)
# ============================================
def manual_upload_models():
    """
    Alternatif: User upload file model secara manual
    """
    st.info("📁 Upload file model satu per satu")
    
    uploaded_files = {}
    
    # Buat kolom untuk upload
    col1, col2 = st.columns(2)
    
    with col1:
        knn_file = st.file_uploader("Upload knn_model.pkl", type=['pkl'])
        if knn_file:
            uploaded_files['knn_model'] = joblib.load(knn_file)
        
        rf_file = st.file_uploader("Upload rf_model.pkl", type=['pkl'])
        if rf_file:
            uploaded_files['rf_model'] = joblib.load(rf_file)
        
        xgb_file = st.file_uploader("Upload xgb_model.pkl", type=['pkl'])
        if xgb_file:
            uploaded_files['xgb_model'] = joblib.load(xgb_file)
        
        scaler_file = st.file_uploader("Upload scaler.pkl", type=['pkl'])
        if scaler_file:
            uploaded_files['scaler'] = joblib.load(scaler_file)
    
    with col2:
        le_cut_file = st.file_uploader("Upload le_cut.pkl", type=['pkl'])
        if le_cut_file:
            uploaded_files['le_cut'] = joblib.load(le_cut_file)
        
        le_color_file = st.file_uploader("Upload le_color.pkl", type=['pkl'])
        if le_color_file:
            uploaded_files['le_color'] = joblib.load(le_color_file)
        
        le_clarity_file = st.file_uploader("Upload le_clarity.pkl", type=['pkl'])
        if le_clarity_file:
            uploaded_files['le_clarity'] = joblib.load(le_clarity_file)
        
        metrics_file = st.file_uploader("Upload metrics.pkl", type=['pkl'])
        if metrics_file:
            uploaded_files['metrics'] = joblib.load(metrics_file)
    
    # Cek apakah semua file sudah terupload
    required_files = ['knn_model', 'rf_model', 'xgb_model', 'scaler', 
                     'le_cut', 'le_color', 'le_clarity']
    
    if all(f in uploaded_files for f in required_files):
        st.success("✅ Semua model berhasil diupload!")
        return uploaded_files
    
    return None

# ============================================
# SIDEBAR - PILIH METODE LOAD MODEL
# ============================================
st.sidebar.title("💎 Diamond Prediction")
st.sidebar.markdown("---")

load_method = st.sidebar.radio(
    "Pilih Metode Load Model:",
    ["🌐 Download dari Google Drive", "📁 Upload Manual"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Petunjuk")
st.sidebar.info(
    """
    **Google Drive Method:**
    - Model akan otomatis didownload
    - Perlu koneksi internet
    - File ID harus benar
    
    **Manual Upload:**
    - Upload file satu per satu
    - Cocok untuk ukuran kecil
    """
)

# ============================================
# MAIN APP
# ============================================
st.title("💎 Diamond Price Prediction")
st.markdown("Prediksi harga diamond menggunakan 3 algoritma Machine Learning")

# Load models berdasarkan pilihan
models = None

if load_method == "🌐 Download dari Google Drive":
    with st.spinner("⏳ Mendownload model dari Google Drive..."):
        models = download_models_from_drive()
        
        if models:
            st.success("✅ Models berhasil didownload dan diload!")
            
            # Tampilkan info model
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("KNN Model", "✅ Ready")
            with col2:
                st.metric("Random Forest", "✅ Ready")
            with col3:
                st.metric("XGBoost", "✅ Ready")

else:  # Manual Upload
    models = manual_upload_models()

# ============================================
# FORM PREDIKSI (Jika models sudah ada)
# ============================================
if models:
    st.markdown("---")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        carat = st.number_input("💎 Carat", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        cut = st.selectbox("✂️ Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        depth = st.number_input("📏 Depth %", min_value=40.0, max_value=80.0, value=61.5)
    
    with col2:
        color = st.selectbox("🎨 Color", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("🔍 Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
        table = st.number_input("📐 Table %", min_value=40.0, max_value=80.0, value=55.0)
    
    with col3:
        x = st.number_input("📏 X (mm)", min_value=1.0, max_value=10.0, value=3.95)
        y = st.number_input("📏 Y (mm)", min_value=1.0, max_value=10.0, value=3.98)
        z = st.number_input("📏 Z (mm)", min_value=1.0, max_value=10.0, value=2.43)
    
    algorithm = st.selectbox(
        "🤖 Pilih Algoritma",
        ["KNN", "Random Forest", "XGBoost"],
        index=1
    )
    
    # Tombol prediksi
    if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
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
            with st.spinner("⏳ Menghitung prediksi..."):
                if algorithm == "KNN":
                    input_scaled = models['scaler'].transform(input_data)
                    prediction = models['knn_model'].predict(input_scaled)[0]
                elif algorithm == "Random Forest":
                    prediction = models['rf_model'].predict(input_data)[0]
                else:  # XGBoost
                    prediction = models['xgb_model'].predict(input_data)[0]
                
                prediction = max(0, prediction)  # Pastikan tidak negatif
            
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
            
            # Big number untuk hasil
            st.markdown(f"""
            <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0;">
                <h2 style="color: white; margin: 0;">Predicted Price</h2>
                <h1 style="color: white; font-size: 72px; margin: 10px 0;">${prediction:,.2f}</h1>
                <p style="color: white;">Menggunakan algoritma: {algorithm}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Tampilkan metrics jika ada
    if 'metrics' in models:
        st.markdown("---")
        st.markdown("### 📊 Model Performance Metrics")
        
        metrics_data = models['metrics']
        
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, model_metrics) in enumerate(metrics_data.items()):
            with [col1, col2, col3][i % 3]:
                st.markdown(f"**{model_name}**")
                st.metric("R² Score", f"{model_metrics['R2']:.4f}")
                st.metric("RMSE", f"${model_metrics['RMSE']:,.2f}")
                st.metric("MAE", f"${model_metrics['MAE']:,.2f}")

else:
    st.info("👈 Silakan pilih metode load model di sidebar untuk memulai")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>© 2024 Diamond Price Prediction | Tugas Machine Learning</p>
        <p style="font-size: 12px;">Model didownload dari Google Drive</p>
    </div>
    """,
    unsafe_allow_html=True
)
