import os
import numpy as np
import pandas as pd

def load_raw_data(filepath="data/raw/data set.csv"):
    """
    Loads raw electricity consumption dataset.
    Falls back to alternate common locations if filepath is not found.
    """
    candidates = [
        filepath,
        "data set.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "data set.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data set.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[DataLoader] Found dataset at: {path}")
            df = pd.read_csv(path)
            ids = df["CONS_NO"]
            labels = df["FLAG"]
            date_cols = [c for c in df.columns if c not in ["CONS_NO", "FLAG"]]
            usage = df[date_cols].astype(float)
            return df, ids, labels, usage
            
    return None, None, None, None

def generate_synthetic_data(num_customers=300, num_days=140, theft_ratio=0.1, random_state=42):
    """
    Generates a realistic synthetic smart meter dataset adhering to the SGCC schema:
    - CONS_NO: customer hash IDs
    - FLAG: 0 (normal) or 1 (theft)
    - Date columns: daily kWh readings with typical residential patterns, noise, missing values,
      and theft patterns (drop in consumption, high zero streaks, flattened usage).
    """
    np.random.seed(random_state)
    print(f"[DataLoader] Generating synthetic benchmark dataset ({num_customers} consumers, {num_days} days, {theft_ratio*100:.1f}% theft)...")
    
    dates = pd.date_range(start="2014-01-01", periods=num_days, freq="D")
    date_cols = [d.strftime("%m/%d/%Y") for d in dates]
    
    n_theft = int(num_customers * theft_ratio)
    labels = np.array([1] * n_theft + [0] * (num_customers - n_theft))
    np.random.shuffle(labels)
    
    cons_ids = [f"{np.random.bytes(16).hex().upper()}" for _ in range(num_customers)]
    
    # Base consumption simulation
    usage_data = np.zeros((num_customers, num_days))
    for i in range(num_customers):
        base_load = np.random.uniform(5.0, 30.0)
        weekly_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(num_days) / 7)
        noise = np.random.normal(0, base_load * 0.15, size=num_days)
        series = np.maximum(0, base_load * weekly_pattern + noise)
        
        # Inject theft signatures for FLAG=1
        if labels[i] == 1:
            theft_start = np.random.randint(num_days // 3, num_days // 2)
            theft_type = np.random.choice(["scale_down", "zero_streak", "drop_to_flat"])
            if theft_type == "scale_down":
                series[theft_start:] *= np.random.uniform(0.1, 0.35)
            elif theft_type == "zero_streak":
                streak_len = np.random.randint(15, 40)
                series[theft_start:min(num_days, theft_start + streak_len)] = 0.0
            elif theft_type == "drop_to_flat":
                series[theft_start:] = np.random.uniform(0.2, 1.5, size=num_days - theft_start)
                
        # Random missing values (NaNs) similar to real SGCC smart meter data
        nan_mask = np.random.rand(num_days) < 0.05
        series[nan_mask] = np.nan
        usage_data[i, :] = series
        
    df_usage = pd.DataFrame(usage_data, columns=date_cols)
    df = pd.DataFrame({"CONS_NO": cons_ids, "FLAG": labels})
    df = pd.concat([df, df_usage], axis=1)
    
    return df, pd.Series(cons_ids, name="CONS_NO"), pd.Series(labels, name="FLAG"), df_usage

def clean_usage_data(usage):
    """
    Cleans smart meter usage data:
    1. Mode imputation per day-column (matches paper preprocessing)
    2. Outlier capping at 99.5th percentile
    """
    print("[DataLoader] Preprocessing usage data: mode imputation & 99.5th percentile outlier clipping...")
    usage_clean = usage.copy()
    for col in usage_clean.columns:
        mode_val = usage_clean[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else 0.0
        usage_clean[col] = usage_clean[col].fillna(fill_val)
        
    cap_val = np.nanpercentile(usage_clean.values, 99.5)
    usage_clean = usage_clean.clip(upper=cap_val)
    return usage_clean
