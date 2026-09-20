import numpy as np
import pandas as pd

def longest_zero_streak(row):
    """Calculates the longest consecutive sequence of zero-consumption days."""
    streak = 0
    max_streak = 0
    for val in row:
        if val == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

def extract_global_features(usage_clean):
    """
    Extracts high-level statistical features across the entire observation window.
    """
    print("[Features] Extracting global statistical features...")
    features = pd.DataFrame(index=usage_clean.index)
    
    features["mean_usage"] = usage_clean.mean(axis=1)
    features["std_usage"] = usage_clean.std(axis=1)
    features["min_usage"] = usage_clean.min(axis=1)
    features["max_usage"] = usage_clean.max(axis=1)
    features["zero_days"] = (usage_clean == 0).sum(axis=1)
    features["peak_to_avg"] = features["max_usage"] / (features["mean_usage"] + 1e-6)
    features["total_usage"] = usage_clean.sum(axis=1)
    
    # Trend: difference between last 30 days and first 30 days
    window_days = min(30, usage_clean.shape[1] // 3)
    first_period = usage_clean.iloc[:, :window_days].mean(axis=1)
    last_period = usage_clean.iloc[:, -window_days:].mean(axis=1)
    features["trend"] = last_period - first_period
    
    features["cv_usage"] = features["std_usage"] / (features["mean_usage"] + 1e-6)
    features["pct_drop"] = np.where(
        first_period > 0.5,
        (first_period - last_period) / (first_period + 1e-6),
        0.0
    )
    features["pct_drop"] = features["pct_drop"].clip(-5, 5)
    features["max_zero_streak"] = usage_clean.apply(longest_zero_streak, axis=1)
    features["median_usage"] = usage_clean.median(axis=1)
    features["skew_usage"] = usage_clean.skew(axis=1)
    
    low_thresh = features["mean_usage"] * 0.1
    features["low_usage_days"] = (usage_clean.lt(low_thresh, axis=0)).sum(axis=1)
    
    chunk_size = max(7, min(30, usage_clean.shape[1] // 4))
    n_chunks = usage_clean.shape[1] // chunk_size
    if n_chunks > 1:
        chunk_means = np.array([
            usage_clean.iloc[:, i * chunk_size:(i + 1) * chunk_size].mean(axis=1).values
            for i in range(n_chunks)
        ]).T
        features["month_to_month_std"] = chunk_means.std(axis=1)
    else:
        features["month_to_month_std"] = 0.0
        
    return features

def extract_windowed_features(usage_clean, window_size=7):
    """
    Extracts localized rolling features for weekly sub-windows.
    """
    n_days = usage_clean.shape[1]
    n_windows = n_days // window_size
    print(f"[Features] Extracting {n_windows} weekly windowed features (window size: {window_size} days)...")
    
    windowed = pd.DataFrame(index=usage_clean.index)
    for w in range(n_windows):
        chunk = usage_clean.iloc[:, w * window_size:(w + 1) * window_size]
        windowed[f"mean_w{w}"] = chunk.mean(axis=1)
        windowed[f"std_w{w}"] = chunk.std(axis=1)
        windowed[f"zero_days_w{w}"] = (chunk == 0).sum(axis=1)
        windowed[f"max_w{w}"] = chunk.max(axis=1)
        
    return windowed

def build_all_features(usage_clean, window_size=7):
    """
    Combines global statistical features and weekly windowed features.
    """
    global_feats = extract_global_features(usage_clean)
    window_feats = extract_windowed_features(usage_clean, window_size=window_size)
    combined = pd.concat([global_feats, window_feats], axis=1)
    print(f"[Features] Total feature matrix shape: {combined.shape}")
    return combined
