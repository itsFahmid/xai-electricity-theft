import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_xgboost(X, y, test_size=0.2, random_state=42, custom_params=None):
    """
    Splits dataset, normalizes features, handles class imbalance, and fits XGBoost model.
    """
    print(f"[Train] Splitting data into train/test (test_size={test_size}, stratify=True)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"[Train] Class distribution - Normal: {n_neg}, Theft: {n_pos} (scale_pos_weight: {scale_pos_weight:.2f})")
    
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.3,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "eval_metric": "logloss"
    }
    
    if custom_params:
        default_params.update(custom_params)
        
    print(f"[Train] Fitting XGBoost classifier...")
    model = XGBClassifier(**default_params)
    model.fit(X_train_scaled, y_train)
    print(f"[Train] Training complete.")
    
    return model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)

def save_model_artifacts(model, scaler, feature_names, output_dir="models"):
    """
    Saves trained model, scaler, and feature list.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "final_xgboost_windowed_model.json")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    features_path = os.path.join(output_dir, "feature_columns.pkl")
    
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    
    print(f"[Train] Saved artifacts to '{output_dir}/':")
    print(f"  - Model: {model_path}")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - Feature Columns: {features_path}")
