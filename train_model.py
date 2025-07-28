import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Membaca dataset
df = pd.read_excel('data/Dataset IPK Mahasiswa.xlsx')

# 2. Preprocessing Data
# Tambahkan fitur baru: Standar Deviasi IPK, Tren IPK, dan Kemiringan Regresi
df['STDEV_IPK'] = df[['SEMESTER 1', 'SEMESTER 2', 'SEMESTER 3', 'SEMESTER 4', 'SEMESTER 5', 'SEMESTER 6', 'SEMESTER 7']].std(axis=1)
df['TREND_IPK'] = df['SEMESTER 7'] - df['SEMESTER 1']

# Hitung kemiringan regresi linier untuk setiap mahasiswa
def calculate_slope(ipk_values):
    X = np.arange(len(ipk_values)).reshape(-1, 1)  # Indeks semester (0 sampai 6)
    y = np.array(ipk_values).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0][0]  # Kembalikan kemiringan

ipk_columns = ['SEMESTER 1', 'SEMESTER 2', 'SEMESTER 3', 'SEMESTER 4', 'SEMESTER 5', 'SEMESTER 6', 'SEMESTER 7']
df['SLOPE_IPK'] = df[ipk_columns].apply(lambda row: calculate_slope(row), axis=1)

# Buat fitur dengan bobot (beri bobot lebih besar pada semester akhir)
weights = [0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25]
weighted_ipk = df[ipk_columns].multiply(weights)
df['WEIGHTED_AVG_IPK'] = weighted_ipk.sum(axis=1)

# Memilih fitur (semua semester, std dev, tren, kemiringan, dan rata-rata tertimbang) dan target
X = df[['SEMESTER 1', 'SEMESTER 2', 'SEMESTER 3', 'SEMESTER 4', 'SEMESTER 5', 'SEMESTER 6', 'SEMESTER 7', 'STDEV_IPK', 'TREND_IPK', 'SLOPE_IPK', 'WEIGHTED_AVG_IPK']]
y = df['STATUS'].map({'LULUS': 1, 'TIDAK LULUS': 0})


print("Jumlah nilai hilang per kolom:")
print(X.isnull().sum())
if X.isnull().sum().sum() > 0:
    df = df.fillna(df.mean())
df = df.dropna()

# Debug: Periksa jumlah baris dan nilai unik
print("Jumlah baris setelah dropna:", len(df))
print("Nilai unik RATA-RATA IPK:", df['RATA-RATA IPK'].unique())

# 3. Exploratory Data Analysis (EDA)
print("Dataset Preview:")
print(df[['SEMESTER 1', 'SEMESTER 2', 'SEMESTER 3', 'SEMESTER 4', 'SEMESTER 5', 'SEMESTER 6', 'SEMESTER 7', 'STDEV_IPK', 'TREND_IPK', 'SLOPE_IPK', 'WEIGHTED_AVG_IPK', 'STATUS']].head())
print("\nDistribusi Status:")
print(y.value_counts())

# Visualisasi distribusi WEIGHTED_AVG_IPK
if len(df) > 1:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='WEIGHTED_AVG_IPK', hue='STATUS', kde=(len(df['WEIGHTED_AVG_IPK'].unique()) > 1))
    plt.title('Distribusi WEIGHTED_AVG_IPK berdasarkan Status')
    plt.savefig('image/weighted_ipk_distribution.png')
    plt.close()
else:
    print("Data tidak cukup untuk visualisasi histogram.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_cv = KNeighborsClassifier(n_neighbors=5, weights='distance')

cv_scores = cross_val_score(knn_cv, X, y, cv=cv, scoring='accuracy')
print("\nCross Validation Accuracy per Fold:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# 4. Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Terapkan SMOTE pada data pelatihan
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

plt.figure(figsize=(7, 5))
y_train.value_counts().plot(kind='bar', color=['#2563eb', '#f59e42'])
plt.title('Distribusi Kelas Sebelum SMOTE')
plt.xlabel('Status')
plt.ylabel('Jumlah')
plt.xticks(ticks=[0,1], labels=['LULUS', 'TIDAK LULUS'], rotation=0)
plt.tight_layout()
plt.savefig('image/distribusi_kelas_sebelum_smote.png')
plt.close()

plt.figure(figsize=(6, 4))
pd.Series(y_train_resampled).value_counts().plot(kind='bar', color=['#2563eb', '#f59e42'])
plt.title('Distribusi Kelas Sesudah SMOTE')
plt.xlabel('Status')
plt.ylabel('Jumlah')
plt.xticks(ticks=[0,1], labels=['LULUS', 'TIDAK LULUS'], rotation=0)
plt.tight_layout()
plt.savefig('image/distribusi_kelas_sesudah_smote.png')
plt.close()

print("\nDistribusi Status (Setelah SMOTE - Data Pelatihan):")
print(pd.Series(y_train_resampled).value_counts())

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 6. Pelatihan Model KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train_resampled)

# 7. Prediksi dan Evaluasi
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAkurasi Model:", accuracy)
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['TIDAK LULUS', 'LULUS']))

# 8. Simpan model dan scaler
joblib.dump(knn, 'model/knn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')