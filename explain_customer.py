"""Explain a single consumer from saved artifacts.

Usage:  python explain_customer.py --row 0
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.lime_explainer import TheftLimeExplainer

MODEL_DIR = "models"
FEATURES_CSV = "data/processed/features_v2.csv"


def load_artifacts(model_dir=MODEL_DIR):
    model = XGBClassifier()                       # NOT joblib.load (§5)
    model.load_model(f"{model_dir}/final_xgboost_windowed_model.json")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    explainer = TheftLimeExplainer.load(model, scaler, model_dir)
    return model, scaler, explainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True,
                    help="row index into data/processed/features_v2.csv")
    ap.add_argument("--features", default=FEATURES_CSV)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    model, scaler, explainer = load_artifacts()
    feats = pd.read_csv(args.features)
    instance_raw = feats.iloc[args.row].to_numpy(dtype=np.float64)

    report = explainer.explain(instance_raw, customer_id=f"row_{args.row}",
                               top_n=args.top)
    print("=" * 66)
    print("ELECTRICITY THEFT DETECTION — EXPLANATION REPORT")
    print(f"Customer: {report['customer_id']}     Verdict: {report['prediction']}")
    print("=" * 66)
    print(report["explanation_text"])


if __name__ == "__main__":
    main()
