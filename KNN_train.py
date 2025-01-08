import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Import SMOTE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
import joblib

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Baca dataset
data = pd.read_csv('bismillah.csv')  # Ganti dengan nama file CSV Anda

# Pisahkan fitur (X) dan label (y)
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].values
y = data['num'].values  # Ganti 'label' dengan nama kolom label Anda

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Terapkan SMOTE untuk menangani ketidakseimbangan kelas
smote = SMOTE(random_state=1234)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1234)

# Membuat dan melatih model KNN dengan k=15
clf = KNN(k=4)
clf.fit(X_train, y_train)

# Prediksi pada data testing
predictions = clf.predict(X_test)

# Menampilkan hasil prediksi
print(predictions)

# Menghitung akurasi
acc = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {acc}')

# Simpan model KNN
model_filename = 'KNN_model.joblib'
scaler_filename = 'KNN_scaler.joblib'

joblib.dump(clf, model_filename)
print(f'Model KNN telah disimpan sebagai {model_filename}')

# Simpan StandardScaler
joblib.dump(scaler, scaler_filename)
print(f'StandardScaler telah disimpan sebagai {scaler_filename}')
