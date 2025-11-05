"""
data_processing.py
Prepares raw BMW i3 EV dataset for modeling.
- Reads semicolon-delimited telemetry
- Fixes encodings and numeric parsing
- Auto-detects trip boundaries
- Outputs cleaned and structured CSVs
"""

import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path="dataset/Trip_data.csv", sep=";", encoding="latin1"):
    print(f"Loading raw dataset from {file_path}")
    df = pd.read_csv(file_path, sep=sep, encoding=encoding, engine="python")

    # Convert relevant columns to numeric safely
    numeric_cols = [
        "Time [s]", "Velocity [km/h]", "Throttle [%]",
        "Battery Voltage [V]", "Battery Current [A]",
        "Motor Torque [Nm]", "Longitudinal Acceleration [m/s^2]",
        "Ambient Temperature [�C]", "SoC [%]"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    # Drop rows missing essential telemetry
    df = df.dropna(subset=["Time [s]", "Battery Voltage [V]", "Battery Current [A]"]).reset_index(drop=True)

    return df


def detect_trips(df):
    """Detect trip boundaries automatically based on SoC jumps or time resets."""
    df["soc_diff"] = df["SoC [%]"].diff()
    df["time_diff"] = df["Time [s]"].diff()
    df["new_trip_flag"] = (df["soc_diff"] > 2) | (df["time_diff"] < 0)
    df["trip_id"] = df["new_trip_flag"].cumsum()

    print(f"Detected {df['trip_id'].nunique()} trips in total.")
    return df


def summarize_trips(df, output_path="dataset/Trip_data_trip_summary.csv"):
    """Create trip-level summary metrics."""
    def summarize_trip(g):
        out = {}
        out["trip_duration_s"] = g["Time [s]"].iloc[-1] - g["Time [s]"].iloc[0]
        out["distance_km"] = (g["Velocity [km/h]"] * np.diff(g["Time [s]"], prepend=0) / 3600).sum()
        out["avg_speed_kmh"] = g["Velocity [km/h]"].mean()
        if "SoC [%]" in g.columns:
            out["soc_start_pct"] = g["SoC [%]"].iloc[0]
            out["soc_end_pct"] = g["SoC [%]"].iloc[-1]
            out["soc_drop_pct"] = out["soc_start_pct"] - out["soc_end_pct"]
        power = g["Battery Voltage [V]"] * g["Battery Current [A]"]
        out["trip_energy_kwh"] = (power * np.diff(g["Time [s]"], prepend=0) / 3_600_000).sum()
        if "Ambient Temperature [�C]" in g.columns:
            out["ambient_temp_mean_c"] = g["Ambient Temperature [�C]"].mean()
        return pd.Series(out)

    trip_summary = df.groupby("trip_id").apply(summarize_trip).reset_index()
    trip_summary.to_csv(output_path, index=False)
    print(f"Trip summary saved at: {output_path}")
    return trip_summary
