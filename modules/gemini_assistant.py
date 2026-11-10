from google import genai
import pandas as pd

def generate_audit_summary(api_key: str, df_anomalies: pd.DataFrame, total_tx: int, total_amount: float) -> str:
    """
    Membuat ringkasan laporan audit eksekutif secara otomatis menggunakan Gemini AI.
    """
    if not api_key:
        return "⚠️ Masukkan Gemini API Key di Sidebar terlebih dahulu untuk menggunakan fitur Asisten Auditor AI."
        
    try:
        client = genai.Client(api_key=api_key)
        
        top_anomalies = df_anomalies.sort_values('Risk_Score', ascending=False).head(15)
        records_text = top_anomalies[[
            'Transaction_ID', 'Timestamp', 'User_ID', 'Vendor', 'Category', 'Amount', 'Risk_Score', 'Anomaly_Reasons'
        ]].to_string(index=False)
        
        prompt = f"""
        Anda adalah Senior Financial & IT Auditor bersertifikat (CISA/CFE).
        Berikut adalah ringkasan hasil deteksi sistem AI Audit pada dataset transaksi perusahaan:
        - Total Transaksi Diperiksa: {total_tx}
        - Total Nilai Transaksi Keseluruhan: Rp {total_amount:,.0f}
        - Jumlah Transaksi Mencurigakan (Anomali): {len(df_anomalies)}
        - Total Nilai Berisiko (Anomali): Rp {df_anomalies['Amount'].sum():,.0f}

        Daftar 15 Transaksi dengan Skor Risiko Tertinggi:
        {records_text}

        Buatlah Laporan Audit Eksekutif dalam Bahasa Indonesia yang terstruktur, tajam, dan profesional mencakup:
        1. **Ringkasan Eksekutif**: Gambaran tingkat risiko dari temuan saat ini.
        2. **Analisis Pola Temuan Utama**: Bahas temuan terkait Transaksi Duplikat, Angka Bulat Berulang, dan Nominal di Luar Pola (sebutkan ID Transaksi, User, atau Vendor yang paling mencolok).
        3. **Rekomendasi Tindak Lanjut (Action Plan)**: Langkah investigasi konkret yang harus dilakukan tim audit internal terhadap pihak terkait.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Gagal menghubungi Gemini API: {str(e)}"

def ask_audit_assistant(api_key: str, user_question: str, df_anomalies: pd.DataFrame) -> str:
    """
    Fitur tanya-jawab interaktif dengan data anomali menggunakan Gemini AI.
    """
    if not api_key:
        return "⚠️ Masukkan Gemini API Key di Sidebar terlebih dahulu."
        
    try:
        client = genai.Client(api_key=AQ.Ab8RN6IiXLp0TpmfW5iMfYSvsaViuqjZwJ8OhF9u0wIxLmy8Eg)
        
        context_data = df_anomalies[[
            'Transaction_ID', 'Timestamp', 'User_ID', 'Vendor', 'Category', 'Amount', 'Risk_Score', 'Anomaly_Reasons'
        ]].head(30).to_csv(index=False)
        
        prompt = f"""
        Anda adalah Asisten AI Auditor Keuangan. Jawab pertanyaan pengguna secara akurat berdasarkan data transaksi mencurigakan berikut:
        
        DATA TRANSAKSI ANOMALI (CSV):
        {context_data}
        
        PERTANYAAN AUDITOR:
        {user_question}
        
        Berikan jawaban yang lugas, sebutkan Transaction_ID dan nominalnya jika relevan.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Terjadi kesalahan saat memproses pertanyaan: {str(e)}"
