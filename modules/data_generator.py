import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_audit_data(n_normal: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Membuat dataset transaksi simulasi yang berisi transaksi normal
    dan 3 jenis anomali: Duplikat, Angka Bulat Berulang, dan Nominal di Luar Pola.
    """
    np.random.seed(seed)
    
    vendors = ['PT Sumber Makmur', 'CV Tekno Solusi', 'UD Berkah Jaya', 'PT Global Logistik', 'CV Prima Mandiri']
    categories = ['Operasional', 'IT & Software', 'ATK & Perlengkapan', 'Perjalanan Dinas', 'Maintenance']
    users = ['USR-01 (Budi)', 'USR-02 (Siti)', 'USR-03 (Andi)', 'USR-04 (Dewi)']
    
    base_time = datetime(2026, 10, 1, 8, 0, 0)
    data = []
    
    # 1. Generate Transaksi Normal (Nominal tidak bulat, distribusi wajar)
    for i in range(1, n_normal + 1):
        cat = np.random.choice(categories)
        if cat == 'ATK & Perlengkapan':
            amt = np.random.normal(450000, 120000)
        elif cat == 'Operasional':
            amt = np.random.normal(1850000, 450000)
        elif cat == 'IT & Software':
            amt = np.random.normal(4200000, 850000)
        elif cat == 'Perjalanan Dinas':
            amt = np.random.normal(2750000, 600000)
        else:
            amt = np.random.normal(1500000, 350000)
            
        # Tambahkan angka acak ratusan/ribuan agar tidak bulat sempurna
        amt = max(75000, round(amt, -2) + np.random.randint(100, 9900))
        
        ts = base_time + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 9),
            minutes=np.random.randint(0, 59)
        )
        
        data.append({
            'Transaction_ID': f'TRX-{1000 + i}',
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'User_ID': np.random.choice(users),
            'Vendor': np.random.choice(vendors),
            'Category': cat,
            'Amount': float(amt),
            'Description': f'Pembayaran tagihan {cat.lower()}'
        })
        
    # 2. Suntikkan Anomali Tipe 1: Transaksi Duplikat (Waktu berdekatan, nominal & vendor sama)
    for j in range(4):
        sample_row = data[j * 15].copy()
        orig_ts = datetime.strptime(sample_row['Timestamp'], '%Y-%m-%d %H:%M:%S')
        dup_ts = orig_ts + timedelta(minutes=np.random.randint(1, 8))
        
        sample_row['Transaction_ID'] = f'TRX-900{j+1}'
        sample_row['Timestamp'] = dup_ts.strftime('%Y-%m-%d %H:%M:%S')
        sample_row['Description'] = sample_row['Description'] + ' (Re-submit)'
        data.append(sample_row)
        
    # 3. Suntikkan Anomali Tipe 2: Angka Bulat Berulang (Misal: Rp 5.000.000 berulang kali oleh 1 user)
    for k in range(6):
        ts = base_time + timedelta(days=k*3 + 1, hours=14, minutes=10*k)
        data.append({
            'Transaction_ID': f'TRX-910{k+1}',
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'User_ID': 'USR-03 (Andi)',
            'Vendor': 'CV Prima Mandiri',
            'Category': 'Operasional',
            'Amount': 5000000.0,
            'Description': 'Biaya jasa konsultasi bulanan'
        })
        
    # 4. Suntikkan Anomali Tipe 3: Nominal di Luar Pola (Lonjakan ekstrim & jam tengah malam)
    outlier_cases = [
        ('TRX-9901', '2026-10-12 02:15:00', 'USR-01 (Budi)', 'PT Global Logistik', 'ATK & Perlengkapan', 28540000.0, 'Pengadaan khusus mendadak'),
        ('TRX-9902', '2026-10-18 23:45:00', 'USR-04 (Dewi)', 'CV Tekno Solusi', 'IT & Software', 64200000.0, 'Upgrade lisensi server darurat'),
        ('TRX-9903', '2026-10-25 03:10:00', 'USR-03 (Andi)', 'PT Sumber Makmur', 'Maintenance', 39000000.0, 'Perbaikan instalasi gedung'),
    ]
    for tid, ts, usr, vnd, cat, amt, desc in outlier_cases:
        data.append({
            'Transaction_ID': tid,
            'Timestamp': ts,
            'User_ID': usr,
            'Vendor': vnd,
            'Category': cat,
            'Amount': amt,
            'Description': desc
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df
