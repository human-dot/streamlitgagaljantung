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

# Fitur input
feature_names = {
    "age": "Usia",
    "sex": "Jenis Kelamin",
    "cp": {
        "description": "Tipe Nyeri Dada",
        "categories": [
            "Typical angina/Nyeri dada khas (0)",
            "Atypical angina/Nyeri dada atipikal (1)",
            "Non-anginal pain/Nyeri dada non-anginal (2)",
            "Asymptomatic/Tidak ada gejala (3)"
        ]
    },
    "trestbps": "Tekanan Darah Istirahat (mm Hg)",
    "chol": "Kolesterol (mg/dl)",
    "fbs": {
        "description": "Gula Darah Puasa",
        "categories": [">120 mg/dl (1)", "Normal (0)"]
    },
    "restecg": {
        "description": "Hasil Elektrokardiogram Istirahat",
        "categories": [
            "Normal/Tidak ada kelainan (0)",
            "ST-T wave abnormality/Kelainan gelombang ST-T (1)",
            "Hypertrophy/Pembesaran ventrikel jantung (2)"
        ]
    },
    "thalach": "Denyut Jantung Maksimal",
    "exang": {
        "description": "Angina yang Diperburuk oleh Olahraga",
        "categories": ["Yes (1)", "No (0)"]
    },
    "oldpeak": "Depresi ST saat olahraga relatif terhadap istirahat",
    "slope": {
        "description": "Kemiringan Segmen ST pada Olahraga Puncak",
        "categories": [
            "Downsloping/Menurun (0)",
            "Flat/Datar (1)",
            "Upsloping/Meningkat (2)"
        ]
    },
    "ca": "Jumlah pembuluh utama (0–3) berwarna oleh fluoroskopi",
    "thal": {
        "description": "Tipe Thalassemia",
        "categories": [
            "Normal (1)",
            "Fixed defect/Kelainan tetap (2)",
            "Reversible defect/Kelainan yang dapat sembuh (3)"
        ]
    }
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
        /* Background styling */
        body {
            background-image: url('https://www.heart.org/-/media/Healthy-Living/Healthy-for-Good/Move-More/Banner.jpg'); 
            background-size: cover;
            background-attachment: fixed;
        }

        /* Title styling */
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #ff4b4b;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        /* Container for form */
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

# Input usia
age = st.number_input("Usia", min_value=1, max_value=120, value=30, step=1)
user_input.append(age)

# Input jenis kelamin
sex = st.selectbox("Jenis Kelamin", options=["Pria", "Wanita"])
user_input.append(1 if sex == "Pria" else 0)

# Input fitur lainnya
for feature, details in feature_names.items():
    if feature in ["age", "sex"]:
        continue

    if isinstance(details, dict):  # Kategorikal
        st.subheader(details["description"])
        categories = details["categories"]
        options = [int(cat.split("(")[-1][0]) for cat in categories]
        value = st.selectbox(feature, options=options, format_func=lambda x: categories[options.index(x)])
    else:  # Numerik
        value = st.number_input(details, value=0.0 if "mm Hg" not in details else 120.0, format="%.2f")
    
    user_input.append(value)

if st.button("Prediksi"):
    prediction_dt = predict_with_model(user_input, model_dt, scaler_dt)
    prediction_knn = predict_with_model(user_input, model_knn, scaler_knn)

    if prediction_dt is not None:
        st.success(f"[Decision Tree] Prediksi: {prediction_dt}")
    if prediction_knn is not None:
        st.success(f"[KNN] Prediksi: {prediction_knn}")
