import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_anomaly_scatter(df: pd.DataFrame):
    """Scatter plot Waktu vs Nominal Transaksi yang membedakan transaksi Normal dan Anomali."""
    color_map = {
        'Normal': '#2b5c8f',
        'Mencurigakan (Anomali)': '#e63946'
    }
    
    fig = px.scatter(
        df,
        x='Timestamp',
        y='Amount',
        color='Status',
        size='Risk_Score',
        hover_data=['Transaction_ID', 'User_ID', 'Vendor', 'Category', 'Anomaly_Reasons', 'Risk_Score'],
        color_discrete_map=color_map,
        title='Peta Sebaran Transaksi & Titik Anomali',
        labels={'Amount': 'Nominal Transaksi (Rp)', 'Timestamp': 'Waktu Transaksi'}
    )
    fig.update_layout(
        xaxis_title="Waktu Transaksi",
        yaxis_title="Nominal (Rp)",
        legend_title="Status Audit",
        template="plotly_white",
        height=450
    )
    return fig

def plot_benfords_law(benford_df: pd.DataFrame):
    """Grafik perbandingan distribusi digit pertama aktual vs kurva standar Benford's Law."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=benford_df['Digit_Pertama'],
        y=benford_df['Aktual_Persen'],
        name='Data Aktual (%)',
        marker_color='#457b9d'
    ))
    
    fig.add_trace(go.Scatter(
        x=benford_df['Digit_Pertama'],
        y=benford_df['Standar_Benford_Persen'],
        name="Standar Benford's Law (%)",
        mode='lines+markers',
        line=dict(color='#e63946', width=3)
    ))
    
    fig.update_layout(
        title="Analisis Benford's Law (Deteksi Manipulasi Digit Pertama)",
        xaxis=dict(title='Digit Pertama Nominal (1 - 9)', tickmode='linear'),
        yaxis=dict(title='Persentase Kemunculan (%)'),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    return fig

def plot_anomaly_breakdown(df: pd.DataFrame):
    """Bar chart ringkasan jumlah temuan berdasarkan jenis anomali."""
    counts = {
        'Transaksi Duplikat': int(df['flag_duplicate'].sum()),
        'Angka Bulat Berulang': int(df['flag_repeating_round'].sum()),
        'Nominal di Luar Pola (AI)': int(df['flag_pattern_outlier'].sum())
    }
    summary_df = pd.DataFrame({
        'Jenis Anomali': list(counts.keys()),
        'Jumlah Transaksi': list(counts.values())
    })
    
    fig = px.bar(
        summary_df,
        x='Jenis Anomali',
        y='Jumlah Transaksi',
        color='Jenis Anomali',
        text='Jumlah Transaksi',
        color_discrete_sequence=['#e76f51', '#f4a261', '#e63946'],
        title='Proporsi Temuan Berdasarkan Kategori Anomali'
    )
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        height=400
    )
    return fig
