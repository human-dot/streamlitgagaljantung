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
    model_dt = None
    scaler_dt = None
    model_knn = None
    scaler_knn = None
    st.error(f"Error: {e}")

# Nama fitur
feature_names = {
    "age": {"description": "Usia"},
    "sex": {"description": "Jenis Kelamin", "options": ["Pria", "Wanita"]},
    "cp": {
        "description": "Tipe Nyeri Dada",
        "options": [
            "Tidak ada gejala",
            "Nyeri dada atipikal",
            "Nyeri dada non-anginal",
            "Nyeri dada khas",
        ],
    },
    "trestbps": {"description": "Tekanan Darah Istirahat (mm Hg)"},
    "chol": {"description": "Kolesterol (mg/dl)"},
    "fbs": {
        "description": "Gula Darah Puasa",
        "options": ["Tidak Normal >120 mg/dl", "Normal"],
    },
    "restecg": {
        "description": "Hasil Elektrokardiogram Istirahat",
        "options": [
            "Pembesaran ventrikel jantung",
            "Normal",
            "Kelainan ST-T",
        ],
    },
    "thalach": {"description": "Denyut Jantung Maksimal"},
    "exang": {
        "description": "Angina yang Diperburuk oleh Olahraga",
        "options": ["Yes", "No"],
    },
    "oldpeak": {
        "description": "Depresi ST saat olahraga relatif terhadap istirahat",
    },
    "slope": {
        "description": "Kemiringan Segmen ST pada Olahraga Puncak",
        "options": [
            "Menurun",
            "Datar",
            "Meningkat",
        ],
    },
    "ca": {"description": "Jumlah pembuluh utama (0–3) berwarna oleh fluoroskopi"},
    "thal": {
        "description": "Thalassemia",
        "options": [
            "Kelainan tetap",
            "Normal",
            "Kelainan yang dapat sembuh",
        ],
    },
}

def predict_with_model(model, scaler, user_input):
    try:
        user_input_scaled = scaler.transform([user_input])
        prediction = model.predict(user_input_scaled)
        return prediction[0]
    except Exception as e:
        return f"Error: {e}"

# Judul aplikasi
st.title("Prediksi Penyakit Gagal Jantung")
st.write("Masukkan nilai untuk setiap fitur di bawah ini:")

# Input user untuk setiap fitur
user_input = []

for feature, details in feature_names.items():
    if "options" in details:
        # Fitur dengan opsi menggunakan dropdown
        value = st.selectbox(details["description"], options=details["options"])
        try:
            # Ambil angka dalam tanda kurung (misalnya "(1)")
            user_input.append(int(value.split("(")[1].strip(")")))
        except (IndexError, ValueError):
            st.error(f"Kesalahan dalam opsi untuk fitur '{details['description']}'. Pastikan format opsi benar.")
            user_input = None
            break
    else:
        # Fitur tanpa opsi menggunakan input numerik
        value = st.number_input(details["description"], value=0.0)
        user_input.append(value)

# Tombol prediksi
if st.button("Prediksi"):
    if model_dt and scaler_dt and model_knn and scaler_knn:
        if user_input:
            # Prediksi dengan kedua model
            prediction_dt = predict_with_model(model_dt, scaler_dt, user_input)
            prediction_knn = predict_with_model(model_knn, scaler_knn, user_input)

            st.success(f"[Decision Tree] Prediksi: Stadium Penyakit Gagal Jantung = {prediction_dt}")
            st.success(f"[KNN] Prediksi: Stadium Penyakit Gagal Jantung = {prediction_knn}")

            # Saran berdasarkan prediksi
            if prediction_dt == 0 and prediction_knn == 0:
                st.info("Hasil prediksi menunjukkan tidak ada risiko gagal jantung. Tetap jaga kesehatan dengan pola makan sehat, olahraga teratur, dan kontrol stres.")
            else:
                st.warning("Hasil prediksi menunjukkan adanya risiko gagal jantung. Berikut adalah beberapa tips untuk pemulihan:")
                st.write("- Ikuti saran dokter dengan baik dan konsumsi obat secara teratur.")
                st.write("- Perbaiki pola makan, hindari makanan tinggi lemak jenuh dan gula.")
                st.write("- Tingkatkan aktivitas fisik ringan sesuai anjuran medis.")
                st.write("- Jangan lupa beristirahat cukup dan hindari stres berlebihan.")
                st.write("Semangat! Kesembuhan adalah proses yang membutuhkan kesabaran dan ketekunan.")
        else:
            st.error("Input tidak valid.")
    else:
        st.error("Model atau scaler tidak ditemukan.")
