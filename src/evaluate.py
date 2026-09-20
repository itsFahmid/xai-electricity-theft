import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import shap

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    Computes classification performance metrics at a given probability threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- Evaluation Results (Threshold = {threshold:.2f}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }

def sweep_thresholds(y_true, y_prob, thresholds=None):
    """
    Evaluates metric trade-offs across various decision thresholds.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print("\n--- Decision Threshold Trade-offs ---")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        print(f"Threshold = {t:.2f} | Precision = {p:.3f} | Recall = {r:.3f} | F1 = {f:.3f}")

def run_shap_explainability(model, X_sample, feature_names, output_plot="models/shap_summary_plot.png"):
    """
    Fits SHAP TreeExplainer on XGBoost model, computes Shapley values, and saves summary plot.
    """
    print("\n[XAI / SHAP] Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"[XAI / SHAP] SHAP feature importance summary plot saved to: {output_plot}")
    
    explainer_path = os.path.join(os.path.dirname(output_plot) or ".", "shap_explainer.pkl")
    joblib.dump(explainer, explainer_path)
    print(f"[XAI / SHAP] Explainer serialized to: {explainer_path}")
    
    return explainer, shap_values


def run_lime_explainability(model, scaler, X_train_scaled, y_train,
                            X_test_scaled, y_test, feature_names,
                            n_examples=5, output_dir="models"):
    """Builds the LIME explainer, prints example explanations, saves artifacts."""
    from src.lime_explainer import TheftLimeExplainer, build_background_sample

    print("\n[XAI/LIME] Building raw-space LIME explainer...")
    X_train_raw = scaler.inverse_transform(X_train_scaled)   # exact for MinMax
    X_test_raw = scaler.inverse_transform(X_test_scaled)
    background = build_background_sample(X_train_raw, y_train, n=5000)

    explainer = TheftLimeExplainer(model=model, scaler=scaler,
                                   feature_names=list(feature_names),
                                   background_raw=background)

    # Explain the highest-risk consumers: that is where LIME is faithful (D4).
    probs = model.predict_proba(X_test_scaled)[:, 1]
    order = np.argsort(-probs)[:n_examples]

    for rank, i in enumerate(order, 1):
        report = explainer.explain(X_test_raw[i], customer_id=f"test_idx_{i}")
        actual = "THEFT" if y_test[i] == 1 else "NORMAL"
        print(f"\n--- LIME #{rank}  (test index {i}, ground truth: {actual}) ---")
        print(report["explanation_text"])
        
        # Temporal / windowed story via SHAP (§6/T6)
        win_factors = explain_windows_with_shap(model, X_test_scaled[i], list(feature_names))
        if win_factors:
            print("\nTop windowed/temporal factors (SHAP):")
            for wf in win_factors:
                d_start, d_end = wf["days"]
                arrow = "raises risk " if wf["shap_value"] > 0 else "lowers risk"
                print(f"  * {wf['feature']:<20s} (days {d_start:3d}..{d_end:3d})  {wf['shap_value']:+.4f}  {arrow}")

    explainer.save(output_dir)
    return explainer


def explain_windows_with_shap(model, X_instance_scaled, feature_names,
                              top_n=5, window_size=7):
    """Top windowed features for one consumer, from exact SHAP values."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_instance_scaled.reshape(1, -1))[0]
    win = [(i, n) for i, n in enumerate(feature_names) if i >= 15]
    ranked = sorted(win, key=lambda t: abs(sv[t[0]]), reverse=True)[:top_n]
    out = []
    for i, name in ranked:
        w = int(name.rsplit("_w", 1)[1])
        out.append({"feature": name, "window": w,
                    "days": (w * window_size, (w + 1) * window_size - 1),
                    "shap_value": float(sv[i])})
    return out


