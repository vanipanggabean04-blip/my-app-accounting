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
