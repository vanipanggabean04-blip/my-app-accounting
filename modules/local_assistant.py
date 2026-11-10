import pandas as pd

def generate_local_audit_summary(df_anomalies: pd.DataFrame, total_tx: int, total_amount: float) -> str:
    """
    Menyusun laporan audit eksekutif secara otomatis menggunakan pendekatan berbasis statistik dan aturan (Rule-Based).
    """
    if len(df_anomalies) == 0:
        return "### 📄 Laporan Hasil Audit Eksekutif\n\nBerdasarkan pemeriksaan, **tidak ditemukan transaksi mencurigakan** pada dataset ini. Seluruh transaksi berada dalam batas kewajaran parameter normal."
        
    total_risk_amount = df_anomalies['Amount'].sum()
    top_user = df_anomalies['User_ID'].mode()[0] if not df_anomalies.empty else "N/A"
    top_vendor = df_anomalies['Vendor'].mode()[0] if not df_anomalies.empty else "N/A"
    
    dup_count = int(df_anomalies['flag_duplicate'].sum())
    round_count = int(df_anomalies['flag_repeating_round'].sum())
    outlier_count = int(df_anomalies['flag_pattern_outlier'].sum())
    
    report = f"""### 📄 Laporan Hasil Audit Eksekutif (LHA)
*Penyusun: Thevani Panggabean (NIM: 5304230064)*

---

#### 1. Ringkasan Eksekutif
* **Total Transaksi Diperiksa**: {total_tx:,} transaksi dengan nilai kumulatif **Rp {total_amount:,.0f}**.
* **Total Temuan Anomali**: Ditemukan **{len(df_anomalies)} transaksi** mencurigakan dengan total nilai berisiko sebesar **Rp {total_risk_amount:,.0f}** ({(len(df_anomalies)/max(1, total_tx)*100):.1f}% dari total volume).
* **Tingkat Risiko Utama**: Risiko tertinggi terkonsentrasi pada entitas **{top_user}** dengan vendor dominan **{top_vendor}**.

#### 2. Analisis Kategori Temuan
* **Indikasi Transaksi Duplikat ({dup_count} Kasus)**: Ditemukan pencairan atau pengajuan pembayaran ganda dengan vendor dan nominal yang sama dalam rentang waktu yang berdekatan. Hal ini memerlukan konfirmasi bagian keuangan untuk menghindari *double payment*.
* **Angka Bulat Berulang ({round_count} Kasus)**: Terdeteksi transaksi dengan nominal angka bulat besar yang berulang secara tidak wajar oleh user/vendor yang sama, mengindikasikan potensi pemecahan faktur (*split invoicing*).
* **Nominal di Luar Pola ({outlier_count} Kasus)**: Lonjakan nilai transaksi ekstrem yang jauh melampaui rata-rata historis kategori, sebagian besar dieksekusi di luar jam operasional normal.

#### 3. Rekomendasi Tindak Lanjut (Action Plan)
1. **Verifikasi Dokumen Pendukung**: Meminta pertanggungjawaban kepada entitas **{top_user}** untuk melampirkan faktur fisik/resi asli atas transaksi bernilai tinggi.
2. **Audit Kepatuhan Vendor**: Melakukan konfirmasi langsung kepada vendor **{top_vendor}** terkait keabsahan transaksi duplikat.
3. **Peningkatan Kontrol Sistem Internal**: Menerapkan validasi otomatis pada sistem pencatatan keuangan untuk mencegah input faktur dengan nomor referensi serupa dalam rentang 1x24 jam.
    """
    return report

def answer_local_audit_question(user_question: str, df_anomalies: pd.DataFrame) -> str:
    """
    Menjawab pertanyaan pengguna berdasarkan pencarian kata kunci pada data temuan anomali.
    """
    q_lower = user_question.lower()
    
    if "user" in q_lower or "siapa" in q_lower or "pengguna" in q_lower:
        top_u = df_anomalies['User_ID'].value_counts().head(3)
        res = "#### 🤖 Jawaban Asisten Audit:\n\nUser/Pengguna yang paling sering terlibat dalam transaksi mencurigakan adalah:\n"
        for usr, count in top_u.items():
            res += f"- **{usr}**: {count} transaksi mencurigakan\n"
        return res
        
    elif "vendor" in q_lower or "pemberi" in q_lower:
        top_v = df_anomalies['Vendor'].value_counts().head(3)
        res = "#### 🤖 Jawaban Asisten Audit:\n\nVendor yang paling sering muncul pada daftar temuan anomali adalah:\n"
        for vnd, count in top_v.items():
            res += f"- **{vnd}**: {count} transaksi\n"
        return res
        
    elif "tertinggi" in q_lower or "besar" in q_lower or "nominal" in q_lower:
        top_amt = df_anomalies.sort_values('Amount', ascending=False).head(3)
        res = "#### 🤖 Jawaban Asisten Audit:\n\nTransaksi dengan nominal terbesar dalam temuan anomali:\n"
        for _, r in top_amt.iterrows():
            res += f"- **{r['Transaction_ID']}** ({r['Vendor']}): Rp {r['Amount']:,.0f} [Skor Risiko: {r['Risk_Score']}] - *{r['Anomaly_Reasons']}*\n"
        return res
        
    elif "duplikat" in q_lower:
        dups = df_anomalies[df_anomalies['flag_duplicate']]
        return f"#### 🤖 Jawaban Asisten Audit:\n\nTerdapat **{len(dups)} transaksi** yang terdeteksi memiliki indikasi duplikat. Silakan cek pada Tab 'Daftar Investigasi Anomali' dengan filter *Indikasi Duplikat*."
        
    else:
        return f"#### 🤖 Jawaban Asisten Audit:\n\nBerdasarkan analisis pada **{len(df_anomalies)} transaksi mencurigakan**, temuan didominasi oleh anomali pola nominal dan pengulangan angka bulat. Coba tanyakan terkait *'siapa user tertinggi'*, *'vendor'*, *'nominal terbesar'*, atau *'duplikat'*."
