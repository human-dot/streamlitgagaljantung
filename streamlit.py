import streamlit as st
from joblib import load
import numpy as np

# Muat model dan scaler
model_dt_path = 'DECISIONTREE_model.joblib'
scaler_dt_path = 'DECISIONTREE_scaler.joblib'
model_knn_path = 'KNN_model.joblib'
scaler_knn_path = 'KNN_scaler.joblib'

try:
    model_dt = load(model_dt_path)
    scaler_dt = load(scaler_dt_path)
    model_knn = load(model_knn_path)
    scaler_knn = load(scaler_knn_path)
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    model_dt, scaler_dt, model_knn, scaler_knn = None, None, None, None

# Fitur input dengan deskripsi
feature_names = {
    "age": {"description": "Usia pasien (dalam tahun)", "input_type": "number"},
    "sex": {"description": "Jenis kelamin pasien", "input_type": "select", "options": ["Pria", "Wanita"]},
    "cp": {"description": "Tipe Nyeri Dada", "input_type": "select", "options": [
        "Typical angina/Nyeri dada khas (0)", 
        "Atypical angina/Nyeri dada atipikal (1)", 
        "Non-anginal pain/Nyeri dada non-anginal (2)", 
        "Asymptomatic/Tidak ada gejala (3)"
    ]},
    "trestbps": {"description": "Tekanan darah saat istirahat (dalam mm Hg)", "input_type": "number"},
    "chol": {"description": "Jumlah kolesterol dalam darah (dalam mg/dl)", "input_type": "number"},
    "fbs": {"description": "Apakah gula darah puasa lebih dari 120 mg/dl?", "input_type": "select", "options": [">120 mg/dl (1)", "Normal (0)"]},
    "restecg": {"description": "Hasil elektrokardiogram saat istirahat", "input_type": "select", "options": [
        "Normal/Tidak ada kelainan (0)", 
        "ST-T wave abnormality/Kelainan gelombang ST-T (1)", 
        "Hypertrophy/Pembesaran ventrikel jantung (2)"
    ]},
    "thalach": {"description": "Denyut jantung maksimal saat latihan", "input_type": "number"},
    "exang": {"description": "Apakah pasien memiliki angina yang diperburuk oleh olahraga?", "input_type": "select", "options": ["Yes (1)", "No (0)"]},
    "oldpeak": {"description": "Depresi ST pada saat olahraga relatif terhadap saat istirahat", "input_type": "number"},
    "slope": {"description": "Kemiringan segmen ST pada saat olahraga puncak", "input_type": "select", "options": [
        "Downsloping/Menurun (0)", 
        "Flat/Datar (1)", 
        "Upsloping/Meningkat (2)"
    ]},
    "ca": {"description": "Jumlah pembuluh utama (0–3) yang terlihat pada fluoroskopi", "input_type": "number"},
    "thal": {"description": "Tipe thalassemia pasien", "input_type": "select", "options": [
        "Normal (1)", 
        "Fixed defect/Kelainan tetap (2)", 
        "Reversible defect/Kelainan yang dapat sembuh (3)"
    ]}
}

# Fungsi prediksi
def predict_with_model(user_input, model, scaler):
    if not model or not scaler:
        st.error("Model atau scaler tidak tersedia.")
        return None
    user_input_scaled = scaler.transform([user_input])
    return model.predict(user_input_scaled)[0]

# CSS untuk background dan styling
st.markdown("""
    <style>
        /* Ganti warna latar belakang */
        body {
            background-color: #f0f8ff;  /* Warna latar belakang (Light Blue) */
        }

        /* Title styling */
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #ff4b4b;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        /* Container untuk form */
        .stApp {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="title">Prediksi Penyakit Gagal Jantung</div>', unsafe_allow_html=True)
st.write("Masukkan nilai untuk setiap fitur di bawah ini:")

user_input = []

# Input fitur dengan deskripsi
for feature, details in feature_names.items():
    if details["input_type"] == "number":
        # Input numerik
        value = st.number_input(details["description"], value=0.0 if "mm Hg" not in details["description"] else 120.0, format="%.2f")
    elif details["input_type"] == "select":
        # Input pilihan dropdown
        value = st.selectbox(details["description"], options=details["options"])

    user_input.append(value)

if st.button("Prediksi"):
    prediction_dt = predict_with_model(user_input, model_dt, scaler_dt)
    prediction_knn = predict_with_model(user_input, model_knn, scaler_knn)

    if prediction_dt is not None:
        st.success(f"[Decision Tree] Prediksi: {prediction_dt}")
    if prediction_knn is not None:
        st.success(f"[KNN] Prediksi: {prediction_knn}")
