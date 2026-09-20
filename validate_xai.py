"""Validation script for LIME integration.

Verifies every acceptance criterion in LIME_IMPLEMENTATION.md Section 7.
Exits with code 0 if all criteria pass, non-zero on any failure.
"""
import importlib.metadata
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

from src.data_loader import generate_synthetic_data, clean_usage_data
from src.features import build_all_features
from src.train import train_xgboost
from src.lime_explainer import TheftLimeExplainer, build_background_sample, N_GLOBAL_FEATURES

def run_validation():
    print("=" * 70)
    print("      LIME INTEGRATION VALIDATION (LIME_IMPLEMENTATION.md Section 7)")
    print("=" * 70)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Criterion 1: lime version is 0.2.0.1
    # -------------------------------------------------------------------------
    print("\n[Criterion 1] Checking lime version...")
    lime_ver = importlib.metadata.version("lime")
    print(f"  Installed lime version: {lime_ver}")
    assert lime_ver == "0.2.0.1", f"Expected lime==0.2.0.1, got {lime_ver}"
    results["1. lime version == 0.2.0.1"] = (True, lime_ver)

    # -------------------------------------------------------------------------
    # Generate full-width benchmark dataset (1,200 consumers, 1,035 days)
    # -------------------------------------------------------------------------
    print("\nGenerating full-width benchmark dataset (1,200 consumers, 1,035 days)...")
    df, ids, labels, usage = generate_synthetic_data(
        num_customers=1200, num_days=1035, theft_ratio=0.085, random_state=42
    )
    usage_clean = clean_usage_data(usage)
    features_full = build_all_features(usage_clean, window_size=7)
    
    # -------------------------------------------------------------------------
    # Criterion 4: Feature count at full width == 603
    # -------------------------------------------------------------------------
    print("\n[Criterion 4] Checking feature count at full width (1,035 days)...")
    n_feats = features_full.shape[1]
    print(f"  Feature count: {n_feats}")
    assert n_feats == 603, f"Expected exactly 603 features, got {n_feats}"
    results["4. Feature count at full width == 603"] = (True, f"{n_feats} features")

    # -------------------------------------------------------------------------
    # Train model and extract splits
    # -------------------------------------------------------------------------
    print("\nTraining XGBoost model on full feature set...")
    X = features_full.values
    y = labels.values
    model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test) = train_xgboost(
        X, y, test_size=0.2, random_state=42
    )
    
    # -------------------------------------------------------------------------
    # Criterion 3: Scaler inverse round-trip error < 1e-9
    # -------------------------------------------------------------------------
    print("\n[Criterion 3] Checking scaler inverse round-trip error...")
    X_train_raw = scaler.inverse_transform(X_train_scaled)
    X_train_roundtrip = scaler.transform(X_train_raw)
    roundtrip_err = float(np.max(np.abs(X_train_scaled - X_train_roundtrip)))
    print(f"  Max inverse round-trip error: {roundtrip_err:.3e}")
    assert roundtrip_err < 1e-9, f"Roundtrip error {roundtrip_err} >= 1e-9"
    results["3. Scaler inverse round-trip error < 1e-9"] = (True, f"{roundtrip_err:.2e}")

    X_test_raw = scaler.inverse_transform(X_test_scaled)
    background_raw = build_background_sample(X_train_raw, y_train, n=5000)
    
    explainer = TheftLimeExplainer(
        model=model,
        scaler=scaler,
        feature_names=list(features_full.columns),
        background_raw=background_raw,
        num_samples=1000,
        random_state=42
    )
    explainer.save("models")

    # -------------------------------------------------------------------------
    # Criterion 9: models/lime_background.npz size < 25 MB
    # -------------------------------------------------------------------------
    print("\n[Criterion 9] Checking models/lime_background.npz file size...")
    bg_file = os.path.join("models", "lime_background.npz")
    assert os.path.exists(bg_file), f"{bg_file} does not exist"
    bg_size_mb = os.path.getsize(bg_file) / (1024 * 1024)
    print(f"  Background file size: {bg_size_mb:.2f} MB")
    assert bg_size_mb < 25.0, f"Background size {bg_size_mb:.2f} MB >= 25 MB"
    results["9. models/lime_background.npz < 25 MB"] = (True, f"{bg_size_mb:.2f} MB")

    # -------------------------------------------------------------------------
    # Criterion 11: theft_probability strictly matches model.predict_proba
    # -------------------------------------------------------------------------
    print("\n[Criterion 11] Verifying theft_probability comes from model.predict_proba...")
    sample_raw = X_test_raw[0]
    expected_prob = float(model.predict_proba(scaler.transform(sample_raw.reshape(1, -1)))[0, 1])
    sample_report = explainer.explain(sample_raw, customer_id="chk_c11")
    reported_prob = sample_report["theft_probability"]
    assert abs(reported_prob - expected_prob) < 1e-7, (
        f"Mismatch: reported={reported_prob}, expected={expected_prob}"
    )
    assert reported_prob != sample_report["surrogate_local_pred"], (
        "theft_probability matched surrogate_local_pred unexpectedly"
    )
    results["11. theft_probability from predict_proba"] = (True, "Confirmed")

    # -------------------------------------------------------------------------
    # Evaluation of 20 test instances: 10 highest-risk + 10 random
    # -------------------------------------------------------------------------
    print("\nExplaining 20 test instances (10 highest-risk + 10 random)...")
    probs = model.predict_proba(X_test_scaled)[:, 1]
    sorted_indices = np.argsort(-probs)
    high_risk_10 = sorted_indices[:10]
    remaining = sorted_indices[10:]
    rng = np.random.RandomState(42)
    random_10 = rng.choice(remaining, size=10, replace=False)
    eval_indices = np.concatenate([high_risk_10, random_10])

    r2_values_flagged = []
    durations = []
    degenerate_count = 0
    
    # For probability band table mirroring Section 3/D4
    band_r2s = {"p >= 0.9 (confident theft)": [], "0.5 <= p < 0.9 (flagged)": [], "p < 0.1 (confident normal)": [], "other": []}

    for idx in eval_indices:
        t0 = time.time()
        rep = explainer.explain(X_test_raw[idx], customer_id=f"inst_{idx}")
        dt = time.time() - t0
        durations.append(dt)
        
        p = rep["theft_probability"]
        r2 = rep["fidelity_r2"]
        local_pred = rep["surrogate_local_pred"]
        
        # Check degenerate trap: MAE == 0 and R2 == 0
        mae_approx = abs(p - local_pred)
        if mae_approx == 0.0 and r2 == 0.0:
            degenerate_count += 1
            
        if p >= 0.5:
            r2_values_flagged.append(r2)
            
        if p >= 0.9:
            band_r2s["p >= 0.9 (confident theft)"].append(r2)
        elif p >= 0.5:
            band_r2s["0.5 <= p < 0.9 (flagged)"].append(r2)
        elif p < 0.1:
            band_r2s["p < 0.1 (confident normal)"].append(r2)
        else:
            band_r2s["other"].append(r2)

    # -------------------------------------------------------------------------
    # Criterion 5: Mean surrogate R^2, instances with p >= 0.5 >= 0.70
    # -------------------------------------------------------------------------
    print("\n[Criterion 5] Checking mean surrogate R^2 on instances with p >= 0.5...")
    mean_r2_flagged = float(np.mean(r2_values_flagged)) if r2_values_flagged else 0.0
    print(f"  Mean R^2 (p >= 0.5, n={len(r2_values_flagged)}): {mean_r2_flagged:.3f}")
    assert mean_r2_flagged >= 0.70, f"Mean R^2 {mean_r2_flagged:.3f} < 0.70"
    results["5. Mean surrogate R^2 (p >= 0.5) >= 0.70"] = (True, f"{mean_r2_flagged:.3f}")

    # -------------------------------------------------------------------------
    # Criterion 7: No explanation with R2 == 0 reported as valid
    # -------------------------------------------------------------------------
    print("\n[Criterion 7] Checking for degenerate R2 == 0 trap...")
    print(f"  Degenerate count: {degenerate_count}")
    assert degenerate_count == 0, f"Found {degenerate_count} degenerate explanations with R2 == 0"
    results["7. No R2 == 0 degenerate explanations"] = (True, "0 occurrences")

    # -------------------------------------------------------------------------
    # Criterion 8: Time per explanation < 0.10 s
    # -------------------------------------------------------------------------
    print("\n[Criterion 8] Checking explanation latency...")
    mean_time = float(np.mean(durations))
    print(f"  Mean latency: {mean_time:.3f} s/instance (max: {np.max(durations):.3f} s)")
    assert mean_time < 0.10, f"Mean latency {mean_time:.3f} s >= 0.10 s"
    results["8. Time per explanation < 0.10 s"] = (True, f"{mean_time:.3f} s")

    # -------------------------------------------------------------------------
    # Criterion 6: Mean |Spearman rho|, SHAP vs LIME, 15 global features >= 0.50
    # -------------------------------------------------------------------------
    print("\n[Criterion 6] Computing SHAP vs LIME Spearman correlation on 10 highest-risk instances...")
    shap_explainer = shap.TreeExplainer(model)
    cols = explainer.subspace_cols
    
    spearman_rhos = []
    for idx in high_risk_10:
        raw_row = X_test_raw[idx]
        def sub_pred(X_sub):
            b = np.repeat(raw_row.reshape(1, -1), len(X_sub), axis=0)
            b[:, cols] = X_sub
            return model.predict_proba(scaler.transform(b))
            
        e = explainer._explainer.explain_instance(
            raw_row[cols], sub_pred, num_features=len(cols), num_samples=1000, labels=(1,)
        )
        lime_weights = np.zeros(N_GLOBAL_FEATURES)
        for c_idx, w in e.local_exp[1]:
            lime_weights[c_idx] = w
            
        sv = shap_explainer.shap_values(X_test_scaled[idx].reshape(1, -1))[0]
        shap_weights = sv[:N_GLOBAL_FEATURES]
        
        rho, _ = spearmanr(np.abs(lime_weights), np.abs(shap_weights))
        if not np.isnan(rho):
            spearman_rhos.append(abs(rho))
            
    mean_spearman = float(np.mean(spearman_rhos)) if spearman_rhos else 0.0
    print(f"  Mean |Spearman rho| (10 highest-risk): {mean_spearman:.3f}")
    assert mean_spearman >= 0.50, f"Mean Spearman rho {mean_spearman:.3f} < 0.50"
    results["6. Mean |Spearman rho| (SHAP vs LIME) >= 0.50"] = (True, f"{mean_spearman:.3f}")

    # -------------------------------------------------------------------------
    # Criterion 10: explain_customer.py --row 0 prints report
    # -------------------------------------------------------------------------
    print("\n[Criterion 10] Running explain_customer.py --row 0 via subprocess...")
    cmd = [sys.executable, "explain_customer.py", "--row", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"explain_customer.py failed with return code {proc.returncode}:\n{proc.stderr}"
    assert "ELECTRICITY THEFT DETECTION" in proc.stdout, "Header missing in explain_customer output"
    print("  explain_customer.py executed cleanly.")
    results["10. explain_customer.py --row 0 runs cleanly"] = (True, "Pass")

    # -------------------------------------------------------------------------
    # Criterion 2: python main.py runs end to end and prints 5 LIME blocks
    # -------------------------------------------------------------------------
    print("\n[Criterion 2] Running python main.py via subprocess...")
    cmd_main = [sys.executable, "main.py"]
    proc_main = subprocess.run(cmd_main, capture_output=True, text=True)
    assert proc_main.returncode == 0, f"main.py failed with returncode {proc_main.returncode}:\n{proc_main.stderr}"
    lime_block_count = proc_main.stdout.count("--- LIME #")
    print(f"  main.py executed cleanly with {lime_block_count} LIME blocks.")
    assert lime_block_count == 5, f"Expected 5 LIME blocks, got {lime_block_count}"
    results["2. python main.py runs end to end with 5 LIME blocks"] = (True, "Pass (5 blocks)")

    # -------------------------------------------------------------------------
    # Table of R^2 by probability band (Section 3/D4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("      MEASURED SURROGATE R^2 BY PROBABILITY BAND (Section 3 / D4)")
    print("=" * 70)
    print(f"{'Probability Band':<35s} | {'Count':<6s} | {'Mean R^2':<8s}")
    print("-" * 55)
    for band_name, r2_list in band_r2s.items():
        cnt = len(r2_list)
        avg_r2 = f"{np.mean(r2_list):.3f}" if cnt > 0 else "N/A"
        print(f"{band_name:<35s} | {cnt:<6d} | {avg_r2:<8s}")

        
    # -------------------------------------------------------------------------
    # Summary of All Criteria
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("      ACCEPTANCE CRITERIA SUMMARY (Section 7)")
    print("=" * 70)
    all_passed = True
    for crit, (passed, val) in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {crit}: {val}")
        if not passed:
            all_passed = False
            
    print("=" * 70)
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA IN Section 7 PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("SOME CRITERIA FAILED.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(run_validation())
