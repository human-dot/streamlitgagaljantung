import tkinter as tk
from tkinter import messagebox
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
    print(f"Error: {e}")

# Fungsi untuk melakukan prediksi menggunakan Decision Tree
def predict_with_decision_tree():
    if not model_dt or not scaler_dt:
        messagebox.showerror("Error", "Model atau scaler Decision Tree tidak ditemukan.")
        return

    try:
        user_input = [float(entries[feature].get()) for feature in feature_names]
    except ValueError:
        messagebox.showerror("Input Error", "Pastikan semua input berupa angka valid.")
        return

    user_input_scaled = scaler_dt.transform([user_input])
    prediction = model_dt.predict(user_input_scaled)
    result = f"[Decision Tree] Prediksi: Stadium Penyakit Gagal Jantung = {prediction[0]}"
    messagebox.showinfo("Hasil Prediksi", result)

# Fungsi untuk melakukan prediksi menggunakan KNN
def predict_with_knn():
    if not model_knn or not scaler_knn:
        messagebox.showerror("Error", "Model atau scaler KNN tidak ditemukan.")
        return

    try:
        user_input = [float(entries[feature].get()) for feature in feature_names]
    except ValueError:
        messagebox.showerror("Input Error", "Pastikan semua input berupa angka valid.")
        return

    user_input_scaled = scaler_knn.transform([user_input])
    prediction = model_knn.predict(user_input_scaled)
    result = f"[KNN] Prediksi: Stadium Penyakit Gagal Jantung = {prediction[0]}"
    messagebox.showinfo("Hasil Prediksi", result)

# GUI dengan Tkinter
root = tk.Tk()
root.title("Prediksi Penyakit Gagal Jantung")
root.geometry("400x700")

feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

entries = {}

# Label dan Entry untuk setiap fitur
tk.Label(root, text="Masukkan nilai untuk setiap fitur", font=("Arial", 14)).pack(pady=10)
for feature in feature_names:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=feature, width=15, anchor="w").pack(side="left")
    entry = tk.Entry(frame)
    entry.pack(side="right", padx=5)
    entries[feature] = entry

# Tombol untuk melakukan prediksi dengan Decision Tree
predict_button_dt = tk.Button(
    root, text="Prediksi (Decision Tree)", command=predict_with_decision_tree, bg="green", fg="white", font=("Arial", 12)
)
predict_button_dt.pack(pady=10)

# Tombol untuk melakukan prediksi dengan KNN
predict_button_knn = tk.Button(
    root, text="Prediksi (KNN)", command=predict_with_knn, bg="blue", fg="white", font=("Arial", 12)
)
predict_button_knn.pack(pady=10)

# Jalankan aplikasi GUI
root.mainloop()