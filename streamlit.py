import streamlit as st
from joblib import load
import numpy as np

# Muat model dan scaler
model_dt_path = 'DECISIONTREE_model.joblib'
scaler_dt_path = 'DECISIONTREE_scaler.joblib'

model_knn_path = 'KNN_model.joblib'
scaler_knn_path = 'KNN_model.joblib'

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
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

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
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(value)

if st.button("Prediksi dengan Decision Tree"):
    try:
        prediction_dt = predict_with_decision_tree(user_input)
        if prediction_dt is not None:
            st.success(f"[Decision Tree] Prediksi: Stadium Penyakit Gagal Jantung = {prediction_dt}")
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Prediksi dengan KNN"):
    try:
        prediction_knn = predict_with_knn(user_input)
        if prediction_knn is not None:
            st.success(f"[KNN] Prediksi: Stadium Penyakit Gagal Jantung = {prediction_knn}")
    except Exception as e:
        st.error(f"Error: {e}")
