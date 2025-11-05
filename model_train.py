"""
model_train.py
Trains XGBoost regression model to predict instantaneous power consumption
based on smoothed EV telemetry signals and lag features.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import re
from helper import sanitize_columns


def clean_feature_names(df):
    """Sanitize column names for XGBoost compatibility."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_") for c in df.columns
    ]
    return df


def train_xgboost(df):
    print("🚀 Training XGBoost regression model...")

    # Sanitize feature names
    df = sanitize_columns(df)

    # Define features dynamically
    possible_features = [c for c in df.columns if "smoothed" in c or "lag" in c]
    target = "power_w_smoothed"

    print(f"Found {len(possible_features)} possible feature columns.")
    print("Example features:", possible_features[:10])
    print(f"Target column present? {target in df.columns}")

    if target not in df.columns:
        raise ValueError(f"❌ Target column '{target}' missing!")

    # Split into features/target
    X = df[possible_features].copy()
    y = df[target].copy()

    # Remove NaNs (if any remain)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    print(f"After cleaning: {len(X)} samples, {X.shape[1]} features")

    if len(X) == 0:
        print("No valid rows remain for training. Check preprocessing.")
        return None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Trained | MAE: {mae:.3f}, R²: {r2:.3f}")
    return model
