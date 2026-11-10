import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class AuditAnomalyDetector:
    def __init__(self, contamination: float = 0.06, model_type: str = "Hybrid (Isolation Forest + Autoencoder)"):
        self.contamination = contamination
        self.model_type = model_type
        self.scaler = StandardScaler()

    def preprocess_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Jika ada timestamp gagal parse, isi dengan waktu default
        if df['Timestamp_dt'].isna().any():
            df['Timestamp_dt'] = df['Timestamp_dt'].fillna(pd.Timestamp('2026-01-01 12:00:00'))
            
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
        
        # 1. Fitur Waktu (Jam transaksi & Akhir pekan)
        df['hour'] = df['Timestamp_dt'].dt.hour
        df['is_weekend'] = df['Timestamp_dt'].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_after_hours'] = ((df['hour'] < 7) | (df['hour'] > 20)).astype(int)
        
        # 2. Fitur Statistik Pola Kategori (Rasio & Z-Score per Kategori)
        cat_stats = df.groupby('Category')['Amount'].transform(lambda x: x.mean() if len(x) > 0 else 1.0)
        cat_std = df.groupby('Category')['Amount'].transform(lambda x: x.std() if len(x) > 1 else 1.0).replace(0, 1.0)
        
        df['amount_to_cat_mean_ratio'] = df['Amount'] / (cat_stats + 1e-5)
        df['cat_z_score'] = (df['Amount'] - cat_stats) / (cat_std + 1e-5)
        
        # 3. Fitur Statistik Pola User
        user_mean = df.groupby('User_ID')['Amount'].transform(lambda x: x.mean() if len(x) > 0 else 1.0)
        df['amount_to_user_mean_ratio'] = df['Amount'] / (user_mean + 1e-5)
        
        # 4. Deteksi Rule 1: Transaksi Duplikat (Vendor & Nominal sama dalam rentang waktu <= 24 jam)
        df = df.sort_values(['Vendor', 'Amount', 'Timestamp_dt'])
        df['time_diff_minutes'] = df.groupby(['Vendor', 'Amount'])['Timestamp_dt'].diff().dt.total_seconds() / 60.0
        
        # Tandai cả transaksi pertama dan kedua jika selisih <= 1440 menit (1 hari)
        is_dup_second = (df['time_diff_minutes'].notna()) & (df['time_diff_minutes'] <= 1440)
        is_dup_first = df.groupby(['Vendor', 'Amount'])['Timestamp_dt'].diff(-1).dt.total_seconds().abs() / 60.0 <= 1440
        df['flag_duplicate'] = (is_dup_second | is_dup_first)
        
        # 5. Deteksi Rule 2: Angka Bulat Berulang (Kelipatan >= 500.000 dan muncul >= 3 kali oleh User/Vendor yang sama)
        df['is_round_number'] = (df['Amount'] >= 500000) & ((df['Amount'] % 500000) == 0)
        round_counts = df[df['is_round_number']].groupby(['User_ID', 'Amount'])['Transaction_ID'].transform('count')
        df['round_repeat_count'] = 0
        df.loc[df['is_round_number'], 'round_repeat_count'] = round_counts
        df['flag_repeating_round'] = df['round_repeat_count'] >= 3
        
        df = df.sort_index()
        return df

    def run_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess_and_engineer_features(df)
        
        feature_cols = [
            'Amount',
            'amount_to_cat_mean_ratio',
            'cat_z_score',
            'amount_to_user_mean_ratio',
            'hour',
            'is_weekend',
            'is_after_hours'
        ]
        
        X = df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # --- MODEL 1: Machine Learning (Isolation Forest) ---
        iso_forest = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            random_state=42
        )
        iso_preds = iso_forest.fit_predict(X_scaled) # -1 untuk anomali, 1 untuk normal
        # Konversi decision_function ke skor 0 - 100 (semakin tinggi semakin mencurigakan)
        raw_iso_scores = -iso_forest.decision_function(X_scaled)
        iso_score_norm = np.interp(raw_iso_scores, (raw_iso_scores.min(), raw_iso_scores.max()), (0, 100))
        
        # --- MODEL 2: Deep Learning (Autoencoder Neural Network) ---
        # Arsitektur Bottleneck: 7 fitur -> 12 neuron -> 3 neuron (latent space) -> 12 neuron -> 7 fitur output
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(12, 3, 12),
            activation='relu',
            solver='adam',
            max_iter=600,
            random_state=42
        )
        autoencoder.fit(X_scaled, X_scaled)
        X_reconstructed = autoencoder.predict(X_scaled)
        
        # Hitung Reconstruction Error (Mean Squared Error per transaksi)
        mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
        ae_threshold = np.percentile(mse, 100 * (1 - self.contamination))
        ae_preds = np.where(mse >= ae_threshold, -1, 1)
        ae_score_norm = np.interp(mse, (mse.min(), mse.max()), (0, 100))
        
        # Gabungkan Skor Berdasarkan Pilihan Model
        if self.model_type == "Machine Learning (Isolation Forest)":
            ai_score = iso_score_norm
            ai_flag = (iso_preds == -1)
        elif self.model_type == "Deep Learning (Autoencoder)":
            ai_score = ae_score_norm
            ai_flag = (ae_preds == -1)
        else:
            # Hybrid Ensemble
            ai_score = (0.5 * iso_score_norm) + (0.5 * ae_score_norm)
            ai_flag = (iso_preds == -1) | (ae_preds == -1)
            
        df['flag_pattern_outlier'] = ai_flag | (df['cat_z_score'].abs() > 3.0)
        
        # Hitung Skor Risiko Komposit (0 - 100)
        risk_scores = ai_score.copy()
        risk_scores = np.where(df['flag_duplicate'], np.maximum(risk_scores + 35, 85), risk_scores)
        risk_scores = np.where(df['flag_repeating_round'], np.maximum(risk_scores + 30, 80), risk_scores)
        df['Risk_Score'] = np.clip(np.round(risk_scores, 1), 0, 100)
        
        # Susun Alasan Temuan (Explainability)
        reasons = []
        for _, row in df.iterrows():
            r_list = []
            if row['flag_duplicate']:
                r_list.append("Indikasi Duplikat")
            if row['flag_repeating_round']:
                r_list.append(f"Angka Bulat Berulang ({int(row['round_repeat_count'])}x)")
            if row['flag_pattern_outlier']:
                if row['cat_z_score'] > 2.5:
                    r_list.append(f"Nominal Ekstrim ({row['amount_to_cat_mean_ratio']:.1f}x rata-rata kategori)")
                elif row['is_after_hours'] == 1:
                    r_list.append("Pola Tidak Wajar (Transaksi Luar Jam Kerja)")
                else:
                    r_list.append("Anomali Pola Multidimensi (AI)")
                    
            if not r_list:
                reasons.append("Normal")
            else:
                reasons.append(" | ".join(r_list))
                
        df['Anomaly_Reasons'] = reasons
        df['Is_Anomaly'] = df['Anomaly_Reasons'] != "Normal"
        df['Status'] = np.where(df['Is_Anomaly'], 'Mencurigakan (Anomali)', 'Normal')
        
        # Hapus kolom bantu internal agar tabel tetap rapi
        drop_cols = ['Timestamp_dt', 'hour', 'is_weekend', 'is_after_hours', 'time_diff_minutes']
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

def compute_benfords_law(amounts: pd.Series) -> pd.DataFrame:
    """
    Menghitung distribusi digit pertama (Benford's Law) untuk mendeteksi manipulasi angka buatan manusia.
    """
    valid_amounts = amounts[amounts >= 10].astype(str).str.lstrip('0').str[0]
    valid_amounts = pd.to_numeric(valid_amounts, errors='coerce').dropna().astype(int)
    valid_amounts = valid_amounts[valid_amounts.between(1, 9)]
    
    total_count = len(valid_amounts)
    actual_counts = valid_amounts.value_counts().reindex(range(1, 10), fill_value=0)
    actual_pct = (actual_counts / max(1, total_count)) * 100
    
    # Rumus Teoritis Benford: P(d) = log10(1 + 1/d)
    benford_pct = [np.log10(1 + 1/d) * 100 for d in range(1, 10)]
    
    return pd.DataFrame({
        'Digit_Pertama': list(range(1, 10)),
        'Aktual_Persen': np.round(actual_pct.values, 2),
        'Standar_Benford_Persen': np.round(benford_pct, 2)
    })
