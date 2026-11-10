import streamlit as st
import pandas as pd
import numpy as np
import os

from modules.data_generator import generate_sample_audit_data
from modules.detector import AuditAnomalyDetector, compute_benfords_law
from modules.visualizer import plot_anomaly_scatter, plot_benfords_law, plot_anomaly_breakdown
from modules.gemini_assistant import generate_audit_summary, ask_audit_assistant

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Deteksi Anomali Transaksi (Audit AI)",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling & UI Polish
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }
    .author-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .author-card h4 {
        margin: 0 0 8px 0;
        color: #38bdf8;
        font-size: 16px;
    }
    .author-card p {
        margin: 4px 0;
        font-size: 13px;
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama dengan Desain Modern
st.markdown("""
    <div style='padding: 20px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 25px;'>
        <h1 style='color: #0f172a; margin-bottom: 5px;'>🛡️ Intelligent Audit: Sistem Deteksi Anomali Transaksi</h1>
        <p style='color: #64748b; font-size: 16px;'>Platform Audit Keuangan Otomatis berbasis <b>Hybrid Machine Learning (Isolation Forest)</b> & <b>Deep Learning (Autoencoder)</b> untuk mendeteksi transaksi duplikat, angka bulat berulang, dan nominal di luar pola[cite: 1].</p>
    </div>
""", unsafe_allow_html=True)

# ================= SIDEBAR KONFIGURASI =================
with st.sidebar:
    # Profil Pembuat
    st.markdown("""
        <div class="author-card">
            <h4>👤 Profil Pembuat</h4>
            <p><b>Nama:</b> Thevani Panggabean</p>
            <p><b>Kelas:</b> AKP 7C</p>
            <p><b>NIM:</b> 5304230064</p>
        </div>
    """, unsafe_allow_html=True)

    st.header("⚙️ Panel Konfigurasi Audit")
    
    # 1. Sumber Data
    st.subheader("1. Sumber Data Transaksi")
    data_source = st.radio(
        "Pilih Sumber Data:",
        ["Gunakan Data Simulasi Otomatis", "Upload File CSV / Excel"]
    )
    
    uploaded_file = None
    if data_source == "Upload File CSV / Excel":
        uploaded_file = st.file_uploader("Unggah file (.csv atau .xlsx)", type=["csv", "xlsx"])
        st.caption("Kolom wajib: `Transaction_ID`, `Timestamp`, `User_ID`, `Vendor`, `Category`, `Amount`")
        
    st.divider()
    
    # 2. Parameter Model AI
    st.subheader("2. Parameter Mesin AI")
    model_choice = st.selectbox(
        "Arsitektur Deteksi Pola:",
        [
            "Hybrid (Isolation Forest + Autoencoder)",
            "Machine Learning (Isolation Forest)",
            "Deep Learning (Autoencoder)"
        ]
    )
    
    # --- DESKRIPSI / PENJELASAN MODEL DI DASHBOARD AWAL ---
    with st.expander("📖 Penjelasan Arsitektur Model AI"):
        if model_choice == "Hybrid (Isolation Forest + Autoencoder)":
            st.markdown(
                "**Hybrid Ensemble Mode**:\n"
                "Menggabungkan kekuatan *Isolation Forest* (ML) dan *Autoencoder* (Deep Learning). "
                "Sistem menghitung skor komposit untuk menangkap anomali linear maupun penyimpangan pola kompleks secara maksimal."
            )
        elif model_choice == "Machine Learning (Isolation Forest)":
            st.markdown(
                "**Isolation Forest (ML)**:\n"
                "Mengisolasi observasi dengan merandom pemisahan nilai fitur. "
                "Anomali dicirikan oleh jalur pemisahan yang pendek (lebih cepat diisolasi), sangat efektif mendeteksi *outlier* multidimensi."
            )
        else:
            st.markdown(
                "**Deep Learning Autoencoder**:\n"
                "Jaringan saraf tiruan (*Neural Network*) simetris (*Bottleneck*) yang dilatih mengenali pola normal. "
                "Transaksi dengan galeri rekonstruksi (*MSE*) tertinggi ditandai sebagai anomali."
            )

    contamination = st.slider(
        "Sensitivitas Anomali Pola (%):",
        min_value=1, max_value=15, value=5, step=1,
        help="Persentase estimasi transaksi di luar pola yang dicari oleh algoritma ML/DL."
    ) / 100.0
    
    st.divider()
    
    # 3. Konfigurasi Gemini API Key
    st.subheader("3. Asisten AI (Gemini API)")
    default_key = ""
    if "GEMINI_API_KEY" in st.secrets:
        default_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        default_key = os.environ.get("GEMINI_API_KEY")
        
    gemini_api_key = st.text_input(
        "Masukkan Gemini API Key:",
        value=default_key,
        type="password",
        help="Opsional: Digunakan pada tab Asisten AI untuk membuat laporan audit naratif otomatis."
    )

# ================= LOAD & VALIDASI DATA =================
@st.cache_data
def load_default_data():
    return generate_sample_audit_data(n_normal=250)

if data_source == "Gunakan Data Simulasi Otomatis":
    raw_df = load_default_data()
else:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
    else:
        st.info("👈 Silakan unggah file CSV/Excel di Sidebar, atau pilih **Gunakan Data Simulasi Otomatis** untuk memulai.")
        st.stop()

required_cols = {'Transaction_ID', 'Timestamp', 'User_ID', 'Vendor', 'Category', 'Amount'}
if not required_cols.issubset(set(raw_df.columns)):
    missing = required_cols - set(raw_df.columns)
    st.error(f"Format kolom tidak sesuai! Kolom berikut belum ada di file Anda: `{', '.join(missing)}`")
    st.stop()

# ================= JALANKAN DETEKSI AI =================
detector = AuditAnomalyDetector(contamination=contamination, model_type=model_choice)
result_df = detector.run_detection(raw_df)
anomalies_df = result_df[result_df['Is_Anomaly']].sort_values('Risk_Score', ascending=False)

# ================= METRIK RINGKASAN EKSEKUTIF =================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transaksi", f"{len(result_df):,}")
with col2:
    st.metric(
        "Transaksi Mencurigakan",
        f"{len(anomalies_df):,}",
        f"{(len(anomalies_df)/len(result_df)*100):.1f}% dari total",
        delta_color="inverse"
    )
with col3:
    st.metric("Duplikat & Angka Bulat", f"{int(result_df['flag_duplicate'].sum() + result_df['flag_repeating_round'].sum())} Temuan")
with col4:
    risk_value = anomalies_df['Amount'].sum()
    st.metric("Total Nilai Berisiko", f"Rp {risk_value:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ================= TABS UTAMA =================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Peta Visualisasi & Pola",
    "🔍 Daftar Investigasi Anomali",
    "📈 Analisis Benford's Law",
    "🤖 Asisten Auditor AI (Gemini)"
])

with tab1:
    st.plotly_chart(plot_anomaly_scatter(result_df), use_container_width=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(plot_anomaly_breakdown(result_df), use_container_width=True)
    with c_right:
        st.subheader("Top 5 Entitas / User dengan Risiko Tertinggi")
        user_risk = anomalies_df.groupby('User_ID').agg(
            Jumlah_Anomali=('Transaction_ID', 'count'),
            Total_Nominal_Anomali=('Amount', 'sum'),
            Rata_Skor_Risiko=('Risk_Score', 'mean')
        ).sort_values('Total_Nominal_Anomali', ascending=False).head(5)
        st.dataframe(
            user_risk.style.format({
                'Total_Nominal_Anomali': 'Rp {:,.0f}',
                'Rata_Skor_Risiko': '{:.1f}'
            }),
            use_container_width=True
        )

with tab2:
    st.subheader("Daftar Transaksi Mencurigakan (Berdasarkan Skor Risiko)")
    
    filter_type = st.multiselect(
        "Filter Berdasarkan Jenis Temuan:",
        ["Indikasi Duplikat", "Angka Bulat Berulang", "Nominal di Luar Pola"],
        default=["Indikasi Duplikat", "Angka Bulat Berulang", "Nominal di Luar Pola"]
    )
    
    filtered_anomalies = anomalies_df.copy()
    mask = pd.Series(False, index=filtered_anomalies.index)
    if "Indikasi Duplikat" in filter_type:
        mask |= filtered_anomalies['flag_duplicate']
    if "Angka Bulat Berulang" in filter_type:
        mask |= filtered_anomalies['flag_repeating_round']
    if "Nominal di Luar Pola" in filter_type:
        mask |= filtered_anomalies['flag_pattern_outlier']
        
    filtered_anomalies = filtered_anomalies[mask]
    
    display_cols = [
        'Transaction_ID', 'Timestamp', 'User_ID', 'Vendor',
        'Category', 'Amount', 'Risk_Score', 'Anomaly_Reasons'
    ]
    if 'Description' in filtered_anomalies.columns:
        display_cols.append('Description')
        
    st.dataframe(
        filtered_anomalies[display_cols].style.format({'Amount': 'Rp {:,.0f}', 'Risk_Score': '{:.1f}'}),
        use_container_width=True,
        height=420
    )
    
    csv_export = filtered_anomalies[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Laporan Transaksi Mencurigakan (CSV)",
        data=csv_export,
        file_name="hasil_audit_anomali_transaksi.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Uji Kepatuhan Digit Pertama (Benford's Law)")
    st.markdown(
        "**Benford's Law** digunakan oleh auditor forensik untuk mendeteksi apakah nominal transaksi "
        "terjadi secara alami atau sengaja dimanipulasi/dikarang oleh manusia."
    )
    benford_df = compute_benfords_law(result_df['Amount'])
    st.plotly_chart(plot_benfords_law(benford_df), use_container_width=True)
    st.dataframe(benford_df, use_container_width=True)

with tab4:
    st.subheader("🤖 Asisten Auditor Eksekutif (Powered by Gemini 2.5 Flash)")
    st.markdown("Gunakan fitur ini untuk menyusun draf **Laporan Hasil Audit (LHA)** otomatis atau berkonsultasi mengenai temuan data.")
    
    col_ai_1, col_ai_2 = st.columns([1, 1])
    
    with col_ai_1:
        st.markdown("#### 📄 Generator Laporan Audit Otomatis")
        if st.button("✨ Buat Kesimpulan & Rekomendasi Audit", type="primary"):
            with st.spinner("Sedang menganalisis temuan dan menyusun laporan audit..."):
                report_md = generate_audit_summary(
                    api_key=gemini_api_key,
                    df_anomalies=anomalies_df,
                    total_tx=len(result_df),
                    total_amount=result_df['Amount'].sum()
                )
                st.session_state['audit_report'] = report_md
                
        if 'audit_report' in st.session_state:
            st.markdown(st.session_state['audit_report'])
            
    with col_ai_2:
        st.markdown("#### 💬 Tanya Jawab Data Temuan")
        user_q = st.text_input(
            "Tanyakan sesuatu tentang transaksi mencurigakan:",
            placeholder="Contoh: Siapa user dan vendor yang paling sering melakukan transaksi angka bulat?"
        )
        if st.button("Kirim Pertanyaan"):
            if user_q.strip():
                with st.spinner("Menganalisis data transaksi..."):
                    answer = ask_audit_assistant(gemini_api_key, user_q, anomalies_df)
                    st.markdown(answer)
            else:
                st.warning("Silakan tulis pertanyaan terlebih dahulu.")

# Footer Estetik
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 13px; padding: 10px;'>"
    "Dibuat dengan ❤️ oleh <b>Thevani Panggabean</b> | Kelas: <b>AKP 7C</b> | NIM: <b>5304230064</b><br>"
    "Sistem Deteksi Anomali Transaksi (Audit AI) • Powered by Python & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
