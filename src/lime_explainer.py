"""Raw-space LIME explanations for the XGBoost electricity-theft detector.

Design notes (see LIME_IMPLEMENTATION.md):
  - LIME runs on RAW features; the scaler is wrapped inside the predict fn.
  - The surrogate is fit over the 15 global features only; windowed features
    are held at the instance's own values.
  - The reported probability always comes from the model, never from LIME.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import joblib
import numpy as np
import lime.lime_tabular

N_GLOBAL_FEATURES = 15          # indices 0..14 of the feature matrix
DEFAULT_NUM_SAMPLES = 1000      # measured knee; more does not help
MIN_FIDELITY_R2 = 0.60          # below this the explanation is not trustworthy
DEFAULT_FLAG_THRESHOLD = 0.5    # only explain consumers at or above this


class TheftLimeExplainer:
    """Local, human-readable explanations for flagged consumers."""

    def __init__(self, model, scaler, feature_names: Sequence[str],
                 background_raw: np.ndarray,
                 subspace_cols: Sequence[int] | None = None,
                 num_samples: int = DEFAULT_NUM_SAMPLES,
                 random_state: int = 42):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)
        self.num_samples = int(num_samples)
        self.random_state = int(random_state)
        self.subspace_cols = (list(range(N_GLOBAL_FEATURES))
                              if subspace_cols is None else list(subspace_cols))

        self.background_raw = np.asarray(background_raw, dtype=np.float64)
        if self.background_raw.shape[1] != len(self.feature_names):
            raise ValueError(
                f"background has {self.background_raw.shape[1]} columns but "
                f"{len(self.feature_names)} feature names were given")

        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.background_raw[:, self.subspace_cols],
            feature_names=[self.feature_names[c] for c in self.subspace_cols],
            mode="classification",
            class_names=["Normal Usage", "Suspected Theft"],
            discretize_continuous=True,
            discretizer="quartile",
            random_state=self.random_state,
        )

    # -- prediction -------------------------------------------------------
    def _predict_raw(self, X_raw: np.ndarray) -> np.ndarray:
        """Model probabilities from RAW features (scaler applied internally)."""
        return self.model.predict_proba(self.scaler.transform(X_raw))

    def theft_probability(self, instance_raw: np.ndarray) -> float:
        return float(self._predict_raw(np.asarray(instance_raw).reshape(1, -1))[0, 1])

    # -- explanation ------------------------------------------------------
    def explain(self, instance_raw: np.ndarray, customer_id: str | None = None,
                top_n: int = 8) -> dict:
        """Explain one consumer. Returns a report dict; see §6/T2 for schema."""
        instance_raw = np.asarray(instance_raw, dtype=np.float64).reshape(1, -1)
        cols = self.subspace_cols
        theft_prob = float(self._predict_raw(instance_raw)[0, 1])

        # Perturbations vary only the subspace; everything else stays as-is.
        def subspace_predict(X_sub: np.ndarray) -> np.ndarray:
            batch = np.repeat(instance_raw, len(X_sub), axis=0)
            batch[:, cols] = X_sub
            return self._predict_raw(batch)

        exp = self._explainer.explain_instance(
            instance_raw[0, cols],
            subspace_predict,
            num_features=len(cols),   # fit on ALL subspace features (D3)
            num_samples=self.num_samples,
            labels=(1,),
        )

        contributions = exp.as_list(label=1)
        contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
        fidelity_r2 = float(exp.score)

        warnings: list[str] = []
        if fidelity_r2 < MIN_FIDELITY_R2:
            warnings.append(
                f"Low surrogate fidelity (R2={fidelity_r2:.2f} < {MIN_FIDELITY_R2}). "
                "Treat these reasons as indicative only; rely on SHAP.")
        if theft_prob < DEFAULT_FLAG_THRESHOLD:
            warnings.append(
                "Consumer is not flagged; LIME fidelity is poor on confidently-"
                "normal accounts (measured R2 ~0.21).")

        return {
            "customer_id": customer_id or "unknown",
            "theft_probability": theft_prob,          # from the MODEL
            "prediction": ("SUSPECTED THEFT" if theft_prob >= DEFAULT_FLAG_THRESHOLD
                           else "NORMAL USAGE"),
            "fidelity_r2": fidelity_r2,
            "surrogate_local_pred": float(exp.local_pred[0]),  # diagnostic only
            "reasons": contributions[:top_n],
            "warnings": warnings,
            "explanation_text": self._format(contributions[:top_n], theft_prob,
                                             fidelity_r2, warnings),
        }

    @staticmethod
    def _format(reasons, theft_prob, r2, warnings) -> str:
        lines = [f"Theft probability (model): {theft_prob:.1%}",
                 f"Explanation fidelity (R2):  {r2:.2f}", "",
                 "Contributing factors:"]
        for i, (desc, weight) in enumerate(reasons, 1):
            arrow = "raises risk " if weight > 0 else "lowers risk"
            lines.append(f"  {i}. {desc:<46s} {weight:+.4f}  {arrow}")
        if warnings:
            lines += ["", "Warnings:"] + [f"  ! {w}" for w in warnings]
        return "\n".join(lines)

    # -- persistence (D5) -------------------------------------------------
    def save(self, output_dir: str = "models") -> None:
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir, "lime_background.npz"),
                            X=self.background_raw.astype(np.float32))
        cfg = {"feature_names": self.feature_names,
               "subspace_cols": self.subspace_cols,
               "num_samples": self.num_samples,
               "random_state": self.random_state}
        with open(os.path.join(output_dir, "lime_config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"[XAI/LIME] Background + config saved to '{output_dir}/'")

    @classmethod
    def load(cls, model, scaler, model_dir: str = "models") -> "TheftLimeExplainer":
        with open(os.path.join(model_dir, "lime_config.json")) as fh:
            cfg = json.load(fh)
        bg = np.load(os.path.join(model_dir, "lime_background.npz"))["X"].astype(np.float64)
        return cls(model=model, scaler=scaler, feature_names=cfg["feature_names"],
                   background_raw=bg, subspace_cols=cfg["subspace_cols"],
                   num_samples=cfg["num_samples"], random_state=cfg["random_state"])


def build_background_sample(X_train_raw: np.ndarray, y_train: np.ndarray,
                            n: int = 5000, random_state: int = 42) -> np.ndarray:
    """Class-stratified background sample, preserving the theft ratio."""
    rng = np.random.RandomState(random_state)
    if len(X_train_raw) <= n:
        return X_train_raw.copy()
    pos = np.flatnonzero(y_train == 1)
    neg = np.flatnonzero(y_train == 0)
    n_pos = max(1, int(round(n * len(pos) / len(X_train_raw))))
    n_pos = min(n_pos, len(pos))
    take = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, min(n - n_pos, len(neg)), replace=False)])
    rng.shuffle(take)
    return X_train_raw[take]
