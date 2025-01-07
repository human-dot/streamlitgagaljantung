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
    model_dt = None
    scaler_dt = None
    model_knn = None
    scaler_knn = None

# Fitur input
feature_names = {
    "age": "Usia",
    "sex": "Jenis Kelamin",
    "cp": ["Typical angina/Nyeri dada khas (0)", "Atypical angina/Nyeri dada atipikal (1)", "Non-anginal pain/Nyeri dada non-anginal (2)", "Asymptomatic/Tidak ada gejala (3)"],
    "trestbps": "Tekanan Darah Istirahat (mm Hg)",
    "chol": "Kolesterol (mg/dl)",
    "fbs": [">120 mg/dl (1)", "Normal (0)"],
    "restecg": ["Normal/Tidak ada kelainan (0)", "ST-T wave abnormality/Kelainan gelombang ST-T (1)", "Hypertrophy/Pembesaran ventrikel jantung (2)"],
    "thalach": "Denyut Jantung Maksimal",
    "exang": ["Yes (1)", "No (0)"],
    "oldpeak": "Depresi ST saat olahraga relatif terhadap istirahat",
    "slope": ["Downsloping/Menurun (0)", "Flat/Datar (1)", "Upsloping/Meningkat (2)"],
    "ca": "Jumlah pembuluh utama (0–3) berwarna oleh fluoroskopi",
    "thal": ["Normal (1)", "Fixed defect/Kelainan tetap (2)", "Reversible defect/Kelainan yang dapat sembuh (3)"]
}

# Fungsi untuk prediksi dengan Decision Tree
def predict_with_decision_tree(user_input):
    if not model_dt or not scaler_dt:
        st.error("Model atau scaler Decision Tree tidak ditemukan.")
        return None

    user_input_scaled = scaler_dt.transform([user_input])
    prediction = model_dt.predict(user_input_scaled)
    return prediction[0]

# Fungsi untuk prediksi dengan KNN
def predict_with_knn(user_input):
    if not model_knn or not scaler_knn:
        st.error("Model atau scaler KNN tidak ditemukan.")
        return None

    user_input_scaled = scaler_knn.transform([user_input])
    prediction = model_knn.predict(user_input_scaled)
    return prediction[0]

# Streamlit UI
st.title("Prediksi Penyakit Gagal Jantung")
st.write("Masukkan nilai untuk setiap fitur di bawah ini:")

user_input = []

# Input usia menggunakan number input
age = st.number_input("Usia", min_value=1, max_value=120, value=25, step=1)
user_input.append(age)

# Input jenis kelamin menggunakan dropdown
sex = st.selectbox("Jenis Kelamin", options=["Pria", "Wanita"])
user_input.append(1 if sex == "Pria" else 0)

# Input fitur lainnya
for feature, description in feature_names.items():
    if feature in ["age", "sex"]:
        continue  # Sudah diatur sebelumnya

    # Fitur yang diubah menjadi input numerik biasa
    if feature in ["trestbps", "chol", "thalach", "oldpeak", "ca"]:
        value = st.number_input(description, value=0.0, format="%.2f")
    elif isinstance(description, list):  # Jika fitur memiliki opsi dropdown
        labels = description
        options = [int(label.split("(")[-1][0]) for label in labels]
        value = st.selectbox(feature, options=options, format_func=lambda x: labels[options.index(x)])
    else:  # Fitur numerik lainnya
        value = st.number_input(description, value=0.0, format="%.2f")

    user_input.append(value)

if st.button("Prediksi"):
    try:
        prediction_dt = predict_with_decision_tree(user_input)
        prediction_knn = predict_with_knn(user_input)

        if prediction_dt is not None and prediction_knn is not None:
            st.success(f"[Decision Tree] Prediksi: Stadium Penyakit Gagal Jantung = {prediction_dt}")
            st.success(f"[KNN] Prediksi: Stadium Penyakit Gagal Jantung = {prediction_knn}")

            # Saran berdasarkan prediksi
            if prediction_dt == 0 and prediction_knn == 0:
                st.info("Anda tidak memiliki penyakit gagal jantung. Berikut adalah beberapa saran untuk mencegah penyakit jantung:")
                st.write("- Makan makanan sehat, seperti buah dan sayuran.")
                st.write("- Rutin berolahraga minimal 30 menit setiap hari.")
                st.write("- Hindari merokok dan konsumsi alkohol berlebihan.")
                st.write("- Jaga berat badan yang sehat.")
                st.write("- Rutin memeriksakan kesehatan ke dokter.")
            else:
                st.warning("Anda mungkin terkena penyakit gagal jantung. Berikut adalah beberapa cara untuk sembuh:")
                st.write("- Ikuti saran dokter dan lakukan pengobatan yang direkomendasikan.")
                st.write("- Perbaiki pola makan dengan mengurangi garam dan lemak jenuh.")
                st.write("- Lakukan aktivitas fisik secara teratur sesuai anjuran dokter.")
                st.write("- Kelola stres dengan baik, seperti meditasi atau yoga.")
                st.write("- Rutin memeriksakan kesehatan dan mengikuti kontrol medis.")
    except Exception as e:
        st.error(f"Error: {e}")
