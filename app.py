import streamlit as st
import pandas as pd
import numpy as np
import os

from modules.data_generator import generate_sample_audit_data
from modules.detector import AuditAnomalyDetector, compute_benfords_law
from modules.visualizer import plot_anomaly_scatter, plot_benfords_law, plot_anomaly_breakdown
from modules.local_assistant import generate_local_audit_summary, answer_local_audit_question

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Deteksi Anomali Transaksi (Audit AI)",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling & UI Polish (Ultra Modern Light Theme + Larger Sidebar)
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #f4f7fb;
    }
    
    /* Metrics Card Styling */
    .stMetric {
        background-color: #ffffff;
        padding: 20px 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        border: 1px solid #eef2f6;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
    }
    
    /* ================= SIDEBAR MODERNIZATION & LARGER UI ================= */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Memperbesar ukuran teks elemen kontrol di Sidebar */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] .stRadio label span, 
    [data-testid="stSidebar"] .stSelectbox label span,
    [data-testid="stSidebar"] .stSlider label span {
        font-size: 16px !important; 
        color: #1e293b !important;
    }
    
    /* Memperbesar dan Menonjolkan File Uploader */
    [data-testid="stFileUploadDropzone"] {
        padding: 2.5rem 1.5rem !important;
        border: 2px dashed #3b82f6 !important;
        background-color: #f0f7ff !important;
        border-radius: 14px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #e0efff !important;
        border-color: #2563eb !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        font-size: 16px !important;
    }

    /* Author Card with gradient accent */
    .author-card {
        background: #ffffff;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
    .author-card h4 {
        margin: 0 0 8px 0;
        color: #0f172a !important;
        font-size: 18px !important;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .author-card p {
        margin: 5px 0;
        font-size: 15px !important;
        color: #475569 !important;
        line-height: 1.5;
    }
    
    /* Sidebar section panels */
    .sidebar-section {
        background: #f8fafc;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 18px;
        border: 1px solid #e2e8f0;
    }
    .sidebar-section h4 {
        font-size: 17px !important;
        color: #0f172a !important;
        margin-bottom: 12px !important;
    }
    /* ==================================================================== */
    
    /* Headers Customization */
    .main-header {
        padding: 24px 32px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .main-header h1 {
        color: #0f172a;
        margin-bottom: 8px;
        font-weight: 900;
        font-size: 2.2rem;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #64748b;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.5;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0fdf4 !important;
        border-top: 3px solid #22c55e !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama dengan Desain Modern
st.markdown("""
    <div class="main-header">
        <h1>🛡️ Intelligent Audit System</h1>
        <p>Platform Audit Keuangan Otomatis berbasis <b>Kecerdasan Buatan (AI)</b> menggunakan <b>Hybrid Machine Learning (Isolation Forest)</b> & <b>Deep Learning (Autoencoder)</b> untuk mendeteksi anomali duplikasi, pengulangan angka bulat, dan pola ekstrem secara presisi.</p>
    </div>
""", unsafe_allow_html=True)

# ================= SIDEBAR KONFIGURASI =================
with st.sidebar:
    # Profil Pembuat (Clean Light Theme & Larger)
    st.markdown("""
        <div class="author-card">
            <h4>👤 Profil Pengembang</h4>
            <p><b>Tim:</b> Thevani & Ziha</p>
            <p><b>Kelas:</b> AKP 7C</p>
            <p><b>NIM:</b> 5304230064 & 5304230083</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Panel Kontrol Audit")
    
    # 1. Sumber Data
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📁 1. Sumber Data Transaksi")
    data_source = st.radio(
        "Pilih Sumber Data:",
        ["Gunakan Data Simulasi Otomatis", "Upload File CSV / Excel"],
        label_visibility="collapsed"
    )
    
    uploaded_file = None
    if data_source == "Upload File CSV / Excel":
        uploaded_file = st.file_uploader("Unggah file (.csv atau .xlsx)", type=["csv", "xlsx"])
        st.caption("Kolom wajib: `Transaction_ID`, `Timestamp`, `User_ID`, `Vendor`, `Category`, `Amount`")
    st.markdown('</div>', unsafe_allow_html=True)
        
    # 2. Parameter Model AI
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🧠 2. Parameter Mesin AI")
    model_choice = st.selectbox(
        "Arsitektur Deteksi Pola:",
        [
            "Hybrid (Isolation Forest + Autoencoder)",
            "Machine Learning (Isolation Forest)",
            "Deep Learning (Autoencoder)"
        ]
    )
    
    with st.expander("📖 Panduan Arsitektur Model"):
        if model_choice == "Hybrid (Isolation Forest + Autoencoder)":
            st.markdown(
                "**Hybrid Ensemble Mode**:\n\n"
                "Menggabungkan kekuatan *Isolation Forest* dan *Autoencoder*. "
                "Menghitung skor komposit untuk menangkap anomali linear maupun penyimpangan pola kompleks."
            )
        elif model_choice == "Machine Learning (Isolation Forest)":
            st.markdown(
                "**Isolation Forest (ML)**:\n\n"
                "Mengisolasi observasi dengan pemisahan acak. "
                "Sangat efektif mendeteksi *outlier* multidimensi pada data keuangan."
            )
        else:
            st.markdown(
                "**Deep Learning Autoencoder**:\n\n"
                "Jaringan saraf tiruan simetris (*Bottleneck*) untuk rekonstruksi pola normal. "
                "Galeri rekonstruksi (*MSE*) tertinggi menandai anomali."
            )

    contamination = st.slider(
        "Sensitivitas Anomali Pola (%):",
        min_value=1, max_value=15, value=5, step=1,
        help="Persentase estimasi transaksi di luar pola yang dicari oleh algoritma AI."
    ) / 100.0
    st.markdown('</div>', unsafe_allow_html=True)

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
    "🤖 Asisten AI Auditor"
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
    st.subheader("🤖 Asisten AI Auditor")
    st.markdown("Fitur asisten ini menghasilkan draf **Laporan Hasil Audit (LHA)** dan menjawab pertanyaan data secara otomatis.")
    
    col_ai_1, col_ai_2 = st.columns([1, 1])
    
    with col_ai_1:
        st.markdown("#### 📄 Generator Laporan Audit Otomatis")
        if st.button("✨ Buat Laporan Hasil Audit (LHA)", type="primary"):
            with st.spinner("Menyusun laporan audit otomatis..."):
                report_md = generate_local_audit_summary(
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
            placeholder="Contoh: Siapa user tertinggi, vendor, atau nominal terbesar?"
        )
        if st.button("Kirim Pertanyaan"):
            if user_q.strip():
                with st.spinner("Menganalisis data..."):
                    answer = answer_local_audit_question(user_q, anomalies_df)
                    st.markdown(answer)
            else:
                st.warning("Silakan tulis pertanyaan terlebih dahulu.")

# Footer Estetik
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 14px; padding: 20px; background-color: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 20px;'>"
    "Dibuat dengan ❤️ oleh <b>Thevani & Ziha</b> | Kelas: <b>AKP 7C</b> | NIM: <b>5304230064 & 5304230083</b><br>"
    "<span style='font-size: 12px; color: #94a3b8;'>Sistem Deteksi Anomali Transaksi (Intelligent Audit System) • Powered by Python & Streamlit</span>"
    "</div>",
    unsafe_allow_html=True
)
