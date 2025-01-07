import tkinter as tk
from tkinter import messagebox
from joblib import load
import numpy as np

# Muat model dan scaler
model_path = 'DECISIONTREE_model.joblib'
scaler_path = 'DECISIONTREE_scaler.joblib'

try:
    loaded_model = load(model_path)
    scaler = load(scaler_path)
except FileNotFoundError as e:
    loaded_model = None
    scaler = None
    print(f"Error: {e}")

# Fungsi untuk melakukan prediksi
def predict_heart_failure():
    if not loaded_model or not scaler:
        messagebox.showerror("Error", "Model atau scaler tidak ditemukan. Pastikan file tersedia.")
        return

    # Ambil input dari entri
    try:
        user_input = [float(entries[feature].get()) for feature in feature_names]
    except ValueError:
        messagebox.showerror("Input Error", "Pastikan semua input berupa angka valid.")
        return

    # Normalisasi data
    user_input_scaled = scaler.transform([user_input])

    # Prediksi
    prediction = loaded_model.predict(user_input_scaled)
    result = f"Prediksi: Stadium Penyakit Gagal Jantung = {prediction[0]}"
    messagebox.showinfo("Hasil Prediksi", result)

# GUI dengan Tkinter
root = tk.Tk()
root.title("Prediksi Penyakit Gagal Jantung")
root.geometry("400x600")

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

# Tombol untuk melakukan prediksi
predict_button = tk.Button(root, text="Prediksi", command=predict_heart_failure, bg="blue", fg="white", font=("Arial", 12))
predict_button.pack(pady=20)

# Jalankan aplikasi GUI
root.mainloop()
