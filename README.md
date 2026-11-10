# 🛡️ Sistem Deteksi Anomali Transaksi (Audit AI)

Aplikasi web berbasis **Python & Streamlit** untuk membantu auditor internal dan forensik mendeteksi transaksi keuangan mencurigakan secara otomatis menggunakan pendekatan **Hybrid Rule-Based, Machine Learning (Isolation Forest), dan Deep Learning (Autoencoder)**, dilengkapi dengan **Asisten AI (Google Gemini)**.

## ✨ Fitur Utama
1. **Deteksi Transaksi Duplikat**: Mengidentifikasi pembayaran ganda pada vendor dan nominal yang sama dalam rentang waktu berdekatan.
2. **Deteksi Angka Bulat Berulang**: Menandai transaksi nominal bulat besar yang berulang secara tidak wajar oleh entitas yang sama.
3. **Deteksi Nominal di Luar Pola (ML & DL)**:
   - **Isolation Forest**: Memisahkan outlier multidimensi berdasarkan rasio kategori, rasio historis user, dan waktu transaksi.
   - **Deep Autoencoder Neural Network**: Menghitung *reconstruction error* untuk menangkap anomali pola kompleks.
4. **Analisis Forensik Benford's Law**: Membandingkan distribusi digit pertama nominal aktual terhadap kurva teoritis Benford.
5. **Asisten Auditor AI (Gemini 2.5 Flash)**: Menghasilkan draf Laporan Hasil Audit (LHA) eksekutif dan tanya-jawab langsung terhadap temuan anomali.

## 🚀 Cara Menjalankan Secara Lokal

```bash
# 1. Clone repositori
git clone [https://github.com/username-anda/audit-anomaly-detector.git](https://github.com/username-anda/audit-anomaly-detector.git)
cd audit-anomaly-detector

# 2. Buat virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Jalankan aplikasi Streamlit
streamlit run app.py
