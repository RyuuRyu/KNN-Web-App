import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Membaca dataset
df = pd.read_excel('./model/Dataset Nilai Mahasiswa.xlsx')
#2. Preprocessing Data

# Menentukan Lulus dengan bobot lebih pada IPK
df['Lulus'] = df.apply(lambda row: 1 if (row['IPK'] >= 2.3 and row['PREDIKAT'] in ['A', 'B', 'C']) else 0, axis=1)

# Memilih fitur dan target
X = df[['ABSENSI', 'TUGAS', 'UTS', 'UAS', 'IPK']]
y = df['Lulus']

# Memeriksa dan menangani data yang hilang
print("Jumlah nilai hilang per kolom:", df.isnull().sum())
df = df.dropna()

# 3. Exploratory Data Analysis (EDA)
print("Dataset Preview:")
print(X.head())
print("\nDistribusi Kelulusan:")
print(df['Lulus'].value_counts())

# Visualisasi korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(X.join(df['Lulus']).corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi Antar Fitur')
plt.savefig('./model/image/correlation_heatmap_weighted_features.png')
plt.close()

# 4. Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Terapkan SMOTE pada data pelatihan
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nDistribusi Kelulusan (Setelah SMOTE - Data Pelatihan):")
print(pd.Series(y_train_resampled).value_counts())

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 6. Pelatihan Model KNN dengan bobot jarak
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train_resampled)

# 7. Prediksi dan Evaluasi
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAkurasi Model:", accuracy)
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Tidak Lulus', 'Lulus']))

# 8. Visualisasi Hasil Prediksi
if len(X_test) > 0 and 'UAS' in X_test.columns and 'IPK' in X_test.columns:
    plt.figure(figsize=(8, 6))
    try:
        sns.scatterplot(x=X_test['UAS'], y=X_test['IPK'], hue=y_pred, style=y_test, palette='deep')
        plt.title('Prediksi Kelulusan vs Data Aktual')
        plt.xlabel('UAS (Weighted)')
        plt.ylabel('IPK (Weighted)')
        plt.legend(title='Prediksi (Simbol: Aktual)')
        plt.savefig('./model/image/prediction_scatter_weighted_features.png')
        plt.close()
    except Exception as e:
        print(f"Error dalam visualisasi scatter plot: {e}")
else:
    print("Data X_test kosong atau kolom UAS/IPK tidak ada.")

# 10. Optimasi Parameter (Mencari k terbaik)
k_values = range(1, 21)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_scaled, y_train_resampled)
    scores.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(8, 6))
plt.plot(k_values, scores, marker='o')
plt.title('Akurasi vs Nilai K')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.grid(True)
plt.savefig('./model/image/k_optimization_updated_ipk_weighted.png')
plt.close()

# 11. Contoh Prediksi Mahasiswa Baru
new_student = pd.DataFrame([[70*0.2, 80*0., 75*0.5, 60*0.7, 2.8*1.0]], columns=['ABSENSI', 'TUGAS', 'UTS', 'UAS', 'IPK'])
new_student_scaled = scaler.transform(new_student)
prediction = knn.predict(new_student_scaled)
print("\nPrediksi Kelulusan Mahasiswa Baru:", "Lulus" if prediction[0] == 1 else "Tidak Lulus")

# Menyimpan model dan scaler pada direktori 'model'
joblib.dump(knn, 'model/knn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')