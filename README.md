<div align="center">
  <div style="background: linear-gradient(to bottom right, #10b981, #4f46e5); width: 64px; height: 64px; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px;">
    <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="white" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
  </div>
  <h1 align="center">FaceMatch.</h1>
  <p align="center">
    <strong>Sistem Verifikasi Wajah Masa Kecil & Dewasa Berbasis Hybrid AI</strong>
    <br/>
    <em>ArcFace Deep Learning & Principal Component Analysis (PCA)</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Flask-Web_Framework-black.svg" alt="Flask">
    <img src="https://img.shields.io/badge/OpenCV-Image_Processing-green.svg" alt="OpenCV">
    <img src="https://img.shields.io/badge/TailwindCSS-Styling-06B6D4.svg" alt="Tailwind">
  </p>
</div>

---

**FaceMatch** adalah sistem cerdas yang dirancang untuk memverifikasi kecocokan identitas seseorang antara **foto masa kecil** dan **foto masa kini**. Sistem ini menggunakan arsitektur hybrid yang menggabungkan ekstraksi fitur non-linear dari model *ArcFace* dengan proyeksi matriks linier melalui dekomposisi *SVD/PCA*.

> **Tugas Proyek Mata Kuliah Aljabar Linier | UNNES 2025**  
> **Kelompok:** Nouval · Tirta · Farritz

---

## ✨ Fitur Utama

- **Penyelarasan Geometri Wajah (*Face Alignment*)**: Secara otomatis mendeteksi mata, hidung, dan mulut untuk merotasi serta memotong wajah (*cropping*) agar sejajar.
- **Hybrid Scoring Engine**: Menggabungkan akurasi model *Deep Learning* modern dengan stabilitas matematis *Principal Component Analysis* (PCA).
- **Eigenface Analysis**: Visualisasi tingkat lanjut yang menghasilkan lapisan bayang-bayang (*ghosting overlay*) dan peta panas (*difference heatmap*) untuk memperlihatkan kecocokan struktur wajah.
- **Antarmuka Interaktif Modern**: Dilengkapi dengan fitur *Dark Mode*, pratinjau foto (*full size lightbox*), dan responsivitas penuh.
- **Privasi Terjamin**: Proses klasifikasi sepenuhnya terjadi di dalam memori server (RAM); foto pengguna tidak pernah disimpan atau diarsipkan secara persisten.

---

## 🛠 Instalasi & Menjalankan Aplikasi

Pastikan Python 3.9 atau yang lebih baru telah terinstal di PC Anda.

**1. Kloning Repositori & Install Dependensi**
```bash
# Lakukan clone atau unduh zip
git clone https://github.com/username/face-match.git
cd face-match

# Install semua library yang dibutuhkan
pip install -r requirements.txt
```

**2. Jalankan Server Web (Flask)**
```bash
python web.py
```
> *Catatan: Sistem pada Windows secara otomatis dikonfigurasi untuk berjalan pada `threaded=False` demi mencegah bentrok alokasi memori pada pustaka TensorFlow/Keras.*

**3. Buka Antarmuka**  
Akses [http://127.0.0.1:5000](http://127.0.0.1:5000) pada browser Anda.

---

## 🔬 Arsitektur Matematis & Pipeline

Proses verifikasi wajah terbagi menjadi 8 alur pemrosesan matriks dan komputasi skalar:

### 1. Ekstraksi Region of Interest (ROI)
Gambar input $I$ beresolusi variabel dideteksi menggunakan algoritma Cascade/SSD, lalu dicrop menjadi $C \in \mathbb{R}^{112 \times 112 \times 3}$.

### 2. Proyeksi Hipersferik (ArcFace)
Matriks piksel dipetakan ke ruang fitur tingkat tinggi melalui model jaringan saraf tiruan menghasilkan vektor berdimensi 512:
$$x = f_{ArcFace}(C) \quad \text{di mana} \quad x \in \mathbb{R}^{512}, \|x\|_2 = 1$$

### 3. Jarak Kosinus Fundamental
Kemiripan dasar dihitung via *Dot Product* di ruang L2-normalized:
$$S_{onnx} = x_1 \cdot x_2 = \cos(\theta)$$

### 4. Transformasi PCA Linier
Vektor $x$ diproyeksikan ke subruang berdimensi lebih rendah (95 dimensi) yang menyimpan 95% varians data latih populasi FGNET menggunakan matriks *Eigenvector* $V_k$:
$$p = V_k^T (x - \mu) \quad \text{di mana} \quad p \in \mathbb{R}^{95}$$

### 5. Jarak Kosinus Tereduksi
Menghitung kemiripan pada subruang hasil dekomposisi *Singular Value Decomposition* (SVD):
$$S_{pca} = \frac{p_1 \cdot p_2}{\|p_1\| \|p_2\|}$$

### 6. Hybrid Decision Rule
Sistem menggunakan aproksimasi skor tertimbang (70% Deep Learning, 30% Aljabar Linier) untuk keputusan akhir:
$$\text{Hybrid Score} = 0.70 \times S_{onnx} + 0.30 \times S_{pca}$$
Jika $\text{Hybrid Score} \geq 0.50$, maka kedua foto dinyatakan identik.

---

## 📁 Struktur Direktori

```text
face_project/
├── web.py                   # Entry point aplikasi Flask
├── train.py                 # (Opsional) Skrip pelatihan PCA luring
├── requirements.txt         # Daftar pustaka dependensi Python
├── README.md                # Dokumentasi utama proyek
├── templates/
│   └── index.html           # Berkas UI Frontend (TailwindCSS + JS)
├── model/
│   ├── model_wajah.pkl      # Matriks eigen & rata-rata hasil latih FGNET
│   └── (bobot model lainnya otomatis diunduh saat runtime pertama kali)
└── src/
    ├── embedder.py          # Wrapper pipeline ArcFace
    ├── scorer.py            # Logika Hybrid & Aljabar SVD/PCA
    └── visualizer.py        # Mesin rendering visualisasi matplotlib/OpenCV
```

---

<div align="center">
  <p>Dibuat dengan ❤️ untuk Kelas Aljabar Linier</p>
</div>
