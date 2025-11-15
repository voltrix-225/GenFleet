"""
viz.py
Visualization utilities for GenFleet project.
Supports both showing plots and saving plots for Flask UI.
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from helper import sanitize_columns

# -------------------------------
# 1️⃣ Plot raw vs smoothed velocity
# -------------------------------
def plot_velocity(df, savepath=None):
    df = sanitize_columns(df)

    plt.figure(figsize=(10, 4))

    if "Velocity [km/h]" in df.columns:
        plt.plot(df["Time [s]"], df["Velocity [km/h]"], alpha=0.4, label="Raw Velocity")

    if "Velocity [km_h]_smoothed" in df.columns:
        plt.plot(
            df["Time [s]"],
            df["Velocity [km_h]_smoothed"],
            linewidth=2,
            label="Smoothed Velocity",
            color="red"
        )

    plt.title("Vehicle Velocity (Raw vs Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [km/h]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
        plt.close()
    else:
        plt.show()


# -------------------------------
# 2️⃣ Plot actual vs predicted power
# -------------------------------
def plot_power_predictions(model, df, savepath=None):
    df = sanitize_columns(df)

    df = df.copy()

    # Target column must exist
    if "power_w_smoothed" not in df.columns:
        print("❌ Missing power_w_smoothed column")
        return

    # Get feature cols used in training
    feature_cols = [c for c in df.columns if "smoothed" in c or "lag" in c]
    X = df[feature_cols]
    y_true = df["power_w_smoothed"]

    y_pred = model.predict(X)

    # Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true[:2000], y_pred[:2000], alpha=0.5)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red",
        lw=2
    )
    plt.xlabel("Actual Power [W]")
    plt.ylabel("Predicted Power [W]")
    plt.title("Actual vs Predicted Instantaneous Power")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
        plt.close()
    else:
        plt.show()


# -------------------------------
# 3️⃣ Feature importance
# -------------------------------
def plot_feature_importance(model, savepath=None):
    df = sanitize_columns(df)

    booster = model.get_booster()
    importance = booster.get_score(importance_type="weight")

    if not importance:
        print("⚠️ No feature importance found.")
        return

    keys = np.array(list(importance.keys()))
    values = np.array(list(importance.values()))
    sorted_idx = np.argsort(values)[::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(keys[sorted_idx], values[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
        plt.close()
    else:
        plt.show()
