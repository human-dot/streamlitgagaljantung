import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Import SMOTE
from DECISIONTREE import DecisionTree
import joblib

# Baca dataset
data = pd.read_csv('cleaned_data.csv')  # Ganti dengan nama file CSV Anda

# Pisahkan fitur (X) dan label (y)
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].values
y = data['num'].values  # Ganti 'label' dengan nama kolom label Anda

# Normalisasi menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Terapkan SMOTE untuk menangani ketidakseimbangan kelas
smote = SMOTE(random_state=1234)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1234)

# Inisialisasi dan latih model DecisionTree
clf = DecisionTree(max_depth=20)
clf.fit(X_train, y_train)

# Prediksi pada data testing
predictions = clf.predict(X_test)

# Fungsi untuk menghitung akurasi
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

# Hitung dan cetak akurasi
acc = accuracy(y_test, predictions)
print(f'Accuracy: {acc}')

# Simpan model DecisionTree
model_filename = 'DECISIONTREE_model.joblib'
scaler_filename = 'DECISIONTREE_scaler.joblib'

joblib.dump(clf, model_filename)
print(f'Model DecisionTree telah disimpan sebagai {model_filename}')

# Simpan StandardScaler
joblib.dump(scaler, scaler_filename)
print(f'StandardScaler telah disimpan sebagai {scaler_filename}')
