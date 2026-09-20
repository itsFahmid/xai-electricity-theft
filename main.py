import os
import sys
import numpy as np
import pandas as pd

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data_loader import load_raw_data, generate_synthetic_data, clean_usage_data
from src.features import build_all_features
from src.train import train_xgboost, save_model_artifacts
from src.evaluate import (evaluate_predictions, sweep_thresholds,
                          run_shap_explainability, run_lime_explainability)

def main():
    print("=" * 70)
    print("      XAI ELECTRICITY THEFT DETECTION PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    df, ids, labels, usage = load_raw_data()
    if df is None:
        print("\n[Notice] Real raw dataset ('data set.csv' or 'data/raw/data set.csv') not found.")
        print("[Notice] Using synthetic smart meter benchmark data to run the full pipeline.")
        df, ids, labels, usage = generate_synthetic_data(num_customers=350, num_days=140, theft_ratio=0.12)
    else:
        print(f"[DataLoader] Successfully loaded dataset with {df.shape[0]} consumers and {usage.shape[1]} days.")
        
    print(f"Total consumers: {len(labels)}, Theft cases: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    
    # 2. Preprocess & Clean
    usage_clean = clean_usage_data(usage)
    
    # 3. Feature Engineering
    features_v2 = build_all_features(usage_clean, window_size=7)
    
    # Save processed features
    os.makedirs("data/processed", exist_ok=True)
    features_path = "data/processed/features_v2.csv"
    features_v2.to_csv(features_path, index=False)
    print(f"[Features] Saved engineered features to: {features_path}")
    
    # 4. Train Model
    X = features_v2.values
    y = labels.values
    model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test) = train_xgboost(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save Model Artifacts
    save_model_artifacts(model, scaler, list(features_v2.columns), output_dir="models")
    
    # 5. Evaluate
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    evaluate_predictions(y_test, y_prob, threshold=0.5)
    sweep_thresholds(y_test, y_prob)
    
    # 6. SHAP Explainability
    sample_size = min(100, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    run_shap_explainability(
        model, 
        X_sample, 
        feature_names=features_v2.columns, 
        output_plot="models/shap_summary_plot.png"
    )
    
    # 7. LIME Explainability
    lime_explainer = run_lime_explainability(
        model=model,
        scaler=scaler,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        feature_names=list(features_v2.columns),
        n_examples=5,
        output_dir="models",
    )
    
    print("\n" + "=" * 70)
    print(" Pipeline execution finished successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
