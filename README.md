# 🧠 Konsep Machine Learning: Regresi Linear Sederhana

Repositori ini berisi penjelasan sederhana mengenai konsep Regresi Linear, salah satu algoritma paling fundamental dalam statistik dan machine learning untuk tujuan prediksi.

---

## 📈 Apa itu Regresi Linear?

Secara sederhana, **Regresi Linear** adalah sebuah metode untuk menemukan hubungan berbentuk **garis lurus** antara dua variabel.

Tujuannya adalah untuk memodelkan hubungan antara:
1.  **Variabel Independen (X):** Variabel yang dianggap sebagai penyebab atau yang digunakan untuk melakukan prediksi.
2.  **Variabel Dependen (Y):** Variabel yang ingin kita prediksi atau akibatnya.

**Analogi:** Bayangkan Anda memiliki sekumpulan titik data pada sebuah grafik. Regresi Linear mencoba menggambar satu garis lurus yang paling "pas" atau paling mewakili semua titik data tersebut.



Contoh sederhananya adalah:
* Memprediksi **nilai ujian** (Y) seorang mahasiswa berdasarkan **lama waktu belajar** (X).
* Memprediksi **harga rumah** (Y) berdasarkan **luas tanahnya** (X).

---

## 🔢 Matematika di Baliknya

Model Regresi Linear mengikuti persamaan garis lurus yang kita kenal dari sekolah:

$$ Y = mX + c $$

Di mana:
-   **Y** adalah nilai yang ingin kita prediksi (variabel dependen).
-   **X** adalah nilai yang kita gunakan untuk memprediksi (variabel independen).
-   **m** adalah **slope** atau **koefisien regresi**. Angka ini menunjukkan seberapa besar perubahan pada Y jika X bertambah satu unit.
-   **c** adalah **intercept**. Ini adalah nilai Y ketika X bernilai 0.

Tujuan utama dari algoritma Regresi Linear adalah **menemukan nilai `m` dan `c` terbaik** yang dapat meminimalkan kesalahan (error) antara nilai prediksi dan nilai sebenarnya. Metode yang paling umum digunakan untuk ini adalah *Ordinary Least Squares (OLS)*.

---

## 💻 Contoh Implementasi Kode

Berikut adalah contoh sederhana menggunakan Python dengan library `scikit-learn` untuk membuat model regresi linear.

Kita akan mencoba memprediksi nilai ujian berdasarkan jam belajar.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Data Sampel
# X = Jam Belajar, y = Nilai Ujian
jam_belajar = np.array([1, 2, 3, 4, 5, 7, 8]).reshape(-1, 1)
nilai_ujian = np.array([60, 65, 70, 72, 75, 85, 90])

# 2. Membuat dan Melatih Model
model = LinearRegression()
model.fit(jam_belajar, nilai_ujian)

# 3. Melihat Hasil (Slope dan Intercept)
slope_m = model.coef_[0]
intercept_c = model.intercept_
print(f"Model Persamaan Garis: Y = {slope_m:.2f}X + {intercept_c:.2f}")

# 4. Membuat Prediksi
# Berapa prediksi nilai jika seseorang belajar selama 6 jam?
prediksi_6_jam = model.predict(np.array([[6]]))
print(f"Prediksi nilai untuk 6 jam belajar: {prediksi_6_jam[0]:.2f}")

# 5. Visualisasi Hasil
plt.scatter(jam_belajar, nilai_ujian, color='blue', label='Data Asli')
plt.plot(jam_belajar, model.predict(jam_belajar), color='red', label='Garis Regresi')
plt.title('Jam Belajar vs Nilai Ujian')
plt.xlabel('Jam Belajar')
plt.ylabel('Nilai Ujian')
plt.legend()
plt.grid(True)
plt.show()