"""
feature_engineering.py
Applies smoothing and lag-based feature engineering for time-series modeling.
Ensures minimal NaN loss and proper numeric continuity.
"""

import pandas as pd
import numpy as np

def create_smooth_and_lag_features(df, window=5, lags=[1, 2, 3]):
    print(f"Applying {window}-second rolling mean smoothing...")

    # --- Ensure numeric columns ---
    numeric_cols = [
        "Battery Voltage [V]", "Battery Current [A]",
        "Velocity [km/h]", "Throttle [%]",
        "Motor Torque [Nm]", "Longitudinal Acceleration [m/s^2]",
        "Heating Power CAN [kW]"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    # --- Compute instantaneous power ---
    df["power_w"] = df["Battery Voltage [V]"] * df["Battery Current [A]"]

    # --- Apply rolling smoothing ---
    smooth_cols = [
        "Velocity [km/h]", "Throttle [%]",
        "Motor Torque [Nm]", "Longitudinal Acceleration [m/s^2]",
        "Heating Power CAN [kW]", "power_w"
    ]
    for col in smooth_cols:
        if col in df.columns:
            df[f"{col}_smoothed"] = df[col].rolling(window=window, min_periods=1).mean()

    # --- Create lag features ---
    print("📈 Creating lag features...")
    for lag in lags:
        for base in ["Throttle [%]", "Velocity [km/h]", "Motor Torque [Nm]"]:
            if base in df.columns:
                df[f"{base}_lag{lag}"] = df[base].shift(lag)

    # --- Smart NaN handling ---
    # Fill missing data (from rolling and lag)
    df = df.fillna(method="bfill").fillna(method="ffill")

    # Drop any rows where *critical target* is still NaN
    df = df[df["power_w_smoothed"].notna()].reset_index(drop=True)

    print(f"✅ Feature engineering complete. Remaining samples: {len(df)}")
    return df
