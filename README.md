# Explainable AI (XAI) for Electricity Theft Detection

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![XAI-SHAP](https://img.shields.io/badge/XAI-SHAP-red.svg)](https://shap.readthedocs.io/)
[![XAI-LIME](https://img.shields.io/badge/XAI-LIME-green.svg)](https://github.com/marcotcr/lime)
[![License-MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end Machine Learning and dual Explainable AI (XAI) pipeline designed to detect electricity theft and consumption anomalies from smart-meter time-series data. 

This project models consumption patterns based on the benchmark **State Grid Corporation of China (SGCC)** dataset, which tracks **42,372 electricity consumers** across **1,035 days** with ground-truth labels (`FLAG = 1` for theft, `0` for normal usage).

---

## Architecture & Dual-XAI Framework

Traditional black-box machine learning models output only a suspicion score (e.g., `theft_probability = 98.4%`). Utility companies and regulatory bodies cannot levy fines or launch physical meter inspections without transparent, legally defensible evidence.

This repository implements a **cooperative dual-layer XAI framework** combining **SHAP** and **LIME** where each method serves a distinct, complementary purpose:

```
                            SMART METER TIME SERIES
                                      │
                                      ▼
                        Data Cleaning & Preprocessing
                       (Mode Imputation & 99.5% Clip)
                                      │
                                      ▼
                           603 Engineered Features
                   (15 Global Statistics + 588 Windowed)
                                      │
                                      ▼
                        XGBoost Classifier (Tuned)
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       GLOBAL & TEMPORAL XAI                       LOCAL CASE-FILE XAI
         SHAP TreeExplainer                          TheftLimeExplainer
  • Exact mathematical attribution             • Raw feature space (unscaled)
  • Covers all 603 features                    • Human-readable threshold rules
  • Pinpoints exact calendar weeks             • Fast local linear surrogate
  • Additivity error < 3e-6                    • High fidelity on flagged accounts (R² ≈ 0.91)
```

### Division of Responsibility: SHAP vs. LIME

| Concern | SHAP (`TreeExplainer`) | LIME (`TheftLimeExplainer`) | Why |
| :--- | :---: | :---: | :--- |
| **Primary Role** | Authoritative Feature Attribution | Human-Readable Case Reports | SHAP is mathematically exact for trees; LIME generates interpretable threshold rules. |
| **Feature Coverage** | **All 603 features** | **15 Global Features** | 1,000 perturbation samples cannot cover 603 dimensions without severe sparsity. |
| **Units of Explanation** | Relative impact ($\Delta$ probability) | **Raw feature units** (kWh, days, ratios) | Investigators need concrete physical cutoffs (e.g., `max_zero_streak > 45.0 days`). |
| **Temporal Granularity** | Identifies specific weeks (`mean_w10`) | Held constant at instance values | SHAP maps exact tampering onset to calendar day ranges. |
| **Surrogate Fidelity** | Exact ($2.2 \times 10^{-16}$ error) | Approximate ($R^2 \approx 0.85\text{--}0.91$ on thefts) | LIME monitors local $R^2$ and emits explicit warnings on low-confidence accounts. |
| **Cross-Method Check** | Baseline ground truth | Verification partner | High rank correlation ($\text{Spearman } \|\rho\| \approx 0.74$) confirms consensus. |

---

## 1. Deep Dive: SHAP Implementation

### Why `TreeExplainer`?
For gradient boosted decision trees (XGBoost), model-agnostic explainers (`KernelExplainer`) require exponential sample masking ($\mathcal{O}(2^M)$), which would take hours to evaluate on 603 features. `shap.TreeExplainer` leverages the internal tree structures to compute **exact Shapley values** in polynomial time ($\mathcal{O}(TLD^2)$), ensuring zero additivity error.

### Global Attribution (Summary Beeswarm Plot)
SHAP aggregates Shapley values across the population to quantify macro-level theft predictors:
- **Feature Importance Ranking**: Features are sorted vertically by their mean absolute SHAP value.
- **Directional Impact**: Each point represents a consumer. Points to the right of zero indicate increased theft risk; points to the left indicate normal consumption patterns.
- **Value Spectrum**: Dot color indicates whether the consumer's feature value was high (red) or low (blue).

The summary beeswarm plot is automatically computed and exported to:
```text
models/shap_summary_plot.png
```

### Temporal Window Attribution
To make the 588 weekly window features actionable, `explain_windows_with_shap()` translates abstract column identifiers into physical inspection dates:
$$\text{Window index } w \implies \text{Days } [w \times 7, (w+1) \times 7 - 1]$$
For example, `zero_days_w66` is mapped directly to **Days 462–468**, pinpointing the exact week consumption vanished.

---

## 2. Deep Dive: LIME Implementation

Standard LIME implementations often mislead when applied blindly to preprocessed tabular data. Our implementation adheres to five core engineering designs:

### Key Design Decisions (D1–D5)

1. **D1: Raw (Unscaled) Feature Space**:
   MinMax normalization compresses features into $[0, 1]$, causing standard LIME discretizers to emit meaningless statements like `0.23 < max_zero_streak <= 0.45`. Our `TheftLimeExplainer` accepts **raw, unscaled features**, wrapping the scaler internally inside the prediction pipeline:
   $$\text{predict\_fn}(X_{\text{raw}}) = \text{model.predict\_proba}(\text{scaler.transform}(X_{\text{raw}}))$$
   Explanations are emitted in real physical units (`max_zero_streak > 45.00 days`).

2. **D2: 15-Global Feature Subspace**:
   Perturbing all 603 features across 1,000 samples results in extreme dimensional sparsity and degrades surrogate $R^2$ to $\sim 0.50$. Restricting the perturbation subspace to the 15 global statistics increases surrogate fidelity to **$R^2 \approx 0.91$** while running **$17\times$ faster** ($0.014\text{ s}$ per customer). Windowed features are held constant at the instance's observed values.

3. **D3: Full Subspace Fitting**:
   LIME's `num_features` parameter refits the Ridge surrogate using only selected columns. We fit over all 15 subspace dimensions to preserve maximum surrogate fidelity, sorting for display only at presentation time.

4. **D4: Gated Explanations & Fidelity Auditing**:
   When a model is saturated on confidently normal customers ($p < 0.1$), the local loss landscape is nearly flat, resulting in degraded linear surrogate fits ($R^2 \approx 0.12\text{--}0.21$). LIME is faithful precisely where needed ($p \ge 0.5$, $R^2 \ge 0.85$). Every explanation report includes an automated audit:
   - Automated warning if $R^2 < 0.60$.
   - Automated warning if $p < 0.50$.
   - The reported probability always originates from `model.predict_proba`, never from LIME's surrogate intercept.

5. **D5: Compressed Background Sampling**:
   Pickling the entire training set inside `LimeTabularExplainer` consumes over 80 MB on disk. We persist a class-stratified background sample (5,000 instances) in a compressed `.npz` archive alongside metadata in `.json` (**1.60 MB total**), reconstructing the explainer on demand in $0.15\text{ s}$.

---

## 3. Sample Case-File Report

Executing `python explain_customer.py --row 26` outputs an audit report:

```text
==================================================================
ELECTRICITY THEFT DETECTION — EXPLANATION REPORT
Customer: row_26     Verdict: SUSPECTED THEFT
==================================================================
Theft probability (model): 98.0%
Explanation fidelity (R2):  0.91

Contributing factors:
  1. cv_usage > 0.32                                +0.6108  raises risk 
  2. month_to_month_std > 1.05                      +0.0724  raises risk 
  3. peak_to_avg > 1.64                             +0.0398  raises risk 
  4. 1965.27 < total_usage <= 2938.86               +0.0325  raises risk 
  5. max_usage > 34.34                              +0.0289  raises risk 
  6. max_zero_streak > 1.00                         -0.0166  lowers risk
  7. skew_usage > -0.98                             +0.0162  raises risk 
  8. low_usage_days > 6.00                          +0.0110  raises risk 

Top windowed/temporal factors (SHAP):
  * mean_w10             (days  70.. 76)  +0.0378  raises risk 
  * zero_days_w12        (days  84.. 90)  +0.0215  raises risk 
```

---

## Project Structure

```text
xai-electricity-theft/
├── data/
│   ├── raw/                  # Raw 'data set.csv' (gitignored)
│   └── processed/            # Engineered features (e.g. features_v2.csv)
├── models/                   # Serialized models, scalers, and XAI artifacts
│   ├── final_xgboost_windowed_model.json   # Trained production XGBoost model
│   ├── scaler.pkl                          # Fitted MinMaxScaler
│   ├── feature_columns.pkl                 # Feature column order registry
│   ├── shap_explainer.pkl                  # Serialized SHAP TreeExplainer
│   ├── shap_summary_plot.png               # Global SHAP beeswarm plot
│   ├── lime_background.npz                 # Compressed stratified background sample
│   └── lime_config.json                    # LIME explainer configuration metadata
├── notebooks/
│   └── 01_explore_data.ipynb # Comprehensive 65-cell EDA and research notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading, mode imputation & synthetic benchmark
│   ├── features.py           # Global statistical and rolling weekly feature engineering
│   ├── train.py              # XGBoost training with class-imbalance weighting
│   ├── evaluate.py           # Evaluation metrics, threshold sweeps, SHAP & LIME
│   └── lime_explainer.py     # TheftLimeExplainer implementation & persistence
├── main.py                   # End-to-end automated execution pipeline
├── explain_customer.py       # Standalone CLI tool for individual meter explanations
├── validate_xai.py           # Acceptance test suite (verifies all 11 criteria)
├── requirements.txt          # Pinned dependency manifest
├── .gitignore
└── README.md
```

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/itsFahmid/xai-electricity-theft.git
cd xai-electricity-theft
```

### 2. Configure Virtual Environment
```bash
# Windows (PowerShell)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run End-to-End Pipeline
Executes data ingestion, feature generation, model training, threshold evaluation, and both SHAP and LIME explanations:
```bash
python main.py
```
> **Data Auto-Fallback**: If `data/raw/data set.csv` is not present, `main.py` automatically synthesizes a realistic residential load benchmark (with embedded theft signatures) so the full pipeline can be verified immediately.

### 2. Generate Customer Explanation Report
Explain any individual consumer meter by index row:
```bash
python explain_customer.py --row 0
python explain_customer.py --row 42 --top 10
```

### 3. Run XAI Test Suite
Validate all 11 quantitative performance, fidelity, and alignment criteria:
```bash
python validate_xai.py
```

### 4. Interactive Research Notebook
Launch Jupyter Lab to explore raw data distributions, LSTM comparisons, and SMOTE experiments:
```bash
jupyter lab notebooks/01_explore_data.ipynb
```

---

## Benchmark & Validation Results

Running `python validate_xai.py` on full-width smart meter readings (1,035 days, 603 features) yields the following results:

| # | Criterion | Threshold | Measured Result | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | LIME Package Release | `== 0.2.0.1` | `0.2.0.1` | **PASSED** |
| **2** | Pipeline Automated Execution | 5 LIME Blocks | 5 Blocks Emitted | **PASSED** |
| **3** | Scaler Inverse Round-trip Error | $< 10^{-9}$ | $2.22 \times 10^{-16}$ | **PASSED** |
| **4** | Feature Matrix Dimensionality | $== 603$ | $603$ features | **PASSED** |
| **5** | Mean Surrogate $R^2$ on Flagged Cases ($p \ge 0.5$) | $\ge 0.70$ | **$0.909$** | **PASSED** |
| **6** | SHAP $\leftrightarrow$ LIME Spearman Correlation ($\|\rho\|$) | $\ge 0.50$ | **$0.734$** | **PASSED** |
| **7** | Degenerate Fits Filter ($R^2 = 0$, $\text{MAE} = 0$) | 0 occurrences | 0 occurrences | **PASSED** |
| **8** | Explanation Latency per Customer | $< 0.10\text{ s}$ | **$0.014\text{ s}$** | **PASSED** |
| **9** | Serialized Background File Size | $< 25\text{ MB}$ | **$1.60\text{ MB}$** | **PASSED** |
| **10** | Standalone Customer CLI Tool | Zero Exceptions | Exit Code 0 | **PASSED** |
| **11** | Attribution Integrity Audit | Strict Model Source | Confirmed (`predict_proba`) | **PASSED** |

### Surrogate Fidelity Across Prediction Bands

| Consumer Prediction Band | Sample Count ($n$) | Mean Surrogate $R^2$ | Interpretation |
| :--- | :---: | :---: | :--- |
| **$p \ge 0.90$ (High-Confidence Theft)** | 10 | **$0.909$** | Highly faithful linear approximation; strong case-file utility |
| **$0.50 \le p < 0.90$ (Flagged)** | — | **$0.85\text{--}0.88$** | Reliable local fit; actionable for inspection teams |
| **$p < 0.10$ (Confident Normal)** | 9 | **$0.119$** | Model saturated (zero gradient); warnings emitted |
