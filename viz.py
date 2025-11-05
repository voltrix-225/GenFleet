"""
viz.py
Visualization utilities for GenFleet project.
Generates performance plots for smoothed telemetry and model predictions.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from helper import sanitize_columns

# Plot raw vs smoothed velocity
def plot_velocity(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df["Time [s]"], df["Velocity [km/h]"], alpha=0.4, label="Raw Velocity")
    if "Velocity [km/h]_smoothed" in df.columns:
        plt.plot(df["Time [s]"], df["Velocity [km/h]_smoothed"], linewidth=2, label="Smoothed Velocity", color="red")
    plt.title("Vehicle Velocity (Raw vs Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [km/h]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# Plot actual vs predicted power
def plot_power_predictions(model, df):
    target = "power_w_smoothed"
    features = [c for c in df.columns if "smoothed" in c or "lag" in c]
    X = df[features]
    y_true = df[target]
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true[:1000], y_pred[:1000], alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", lw=2)
    plt.xlabel("Actual Power [W]")
    plt.ylabel("Predicted Power [W]")
    plt.title("Actual vs Predicted Instantaneous Power")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # optional error plot
    plt.figure(figsize=(8, 3))
    plt.plot(y_true[:1000].values - y_pred[:1000], color="orange")
    plt.title("Prediction Error (first 1000 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Error [W]")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# Feature importance from XGBoost
def plot_feature_importance(model):
    importance = model.get_booster().get_score(importance_type="weight")
    if not importance:
        print("⚠️ No feature importance data found in model.")
        return

    keys = list(importance.keys())
    values = list(importance.values())
    sorted_idx = np.argsort(values)[::-1]

    plt.figure(figsize=(8, 4))
    plt.barh(np.array(keys)[sorted_idx], np.array(values)[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (by weight)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def plot_power_predictions(model, df):
    # sanitize columns so names match those used in training
    df = sanitize_columns(df)
    
    target = "power_w_smoothed"
    features = [c for c in df.columns if "smoothed" in c or "lag" in c]
