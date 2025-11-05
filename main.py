"""
main.py
Main entry point for GenFleet pipeline.
- Loads and cleans EV telemetry
- Detects trips and builds summaries
- Applies feature engineering
- Trains predictive model
"""

from data_processing import load_and_clean_data, detect_trips, summarize_trips
from feature_engineering import create_smooth_and_lag_features
from model_train import train_xgboost
from helper import ensure_dir
from viz import plot_velocity, plot_power_predictions, plot_feature_importance


"""
main.py
Main entry point for GenFleet EV Analytics Pipeline.

Pipeline Overview:
1️⃣ Load & clean the raw BMW i3 dataset
2️⃣ Auto-detect trips and create trip summaries
3️⃣ Apply smoothing + lag feature engineering
4️⃣ Train an XGBoost regression model
5️⃣ Visualize smoothed telemetry, predictions, and feature importance
"""

from helper import ensure_dir
from data_processing import load_and_clean_data, detect_trips, summarize_trips
from feature_engineering import create_smooth_and_lag_features
from model_train import train_xgboost
from viz import plot_velocity, plot_power_predictions, plot_feature_importance
from helper import sanitize_columns


def main():
    print("\nStarting GenFleet EV Analytics Pipeline\n")

    # STEP 0 — Ensure folders exist
    ensure_dir("dataset")
    ensure_dir("results")

    # STEP 1 — Load & clean
    print("Loading and cleaning data...")
    df = load_and_clean_data("dataset/Trip_data.csv")

    # STEP 2 — Trip detection & summary (optional, mostly for analysis)
    print("\nDetecting individual trips...")
    df = detect_trips(df)
    summarize_trips(df)

    # STEP 3 — Feature engineering (rolling + lag)
    print("\nPerforming feature engineering...")
    df_fe = create_smooth_and_lag_features(df)

    # STEP 4 — Model training
    print("\nTraining XGBoost model...")
    model = train_xgboost(df_fe)

    # STEP 5 — Visualizations
    if model is not None:
        print("\nGenerating visualizations...")
        plot_velocity(df_fe)
        plot_power_predictions(model, df_fe)
        plot_feature_importance(model)

    print("\nPipeline execution complete. All steps finished successfully.\n")


if __name__ == "__main__":
    main()
