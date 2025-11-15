from flask import Flask, render_template, request, jsonify
from data_processing import load_and_clean_data, detect_trips, summarize_trips
from feature_engineering import create_smooth_and_lag_features
from model_train import train_xgboost
from viz import plot_velocity, plot_power_predictions, plot_feature_importance
from helper import ensure_dir, sanitize_columns
from chatbot import chat_response
import os

app = Flask(__name__)

@app.route("/chat_ui")
def chat_ui():
    return render_template("chat.html")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_pipeline():
    try:
        ensure_dir("results")

        df = load_and_clean_data("dataset/Trip_data.csv")
        df = detect_trips(df)
        summarize_trips(df)
        df_fe = create_smooth_and_lag_features(df)
        model = train_xgboost(df_fe)

        # Generate plots → save in /static/plots/
        ensure_dir("static/plots")
        plot_velocity(df_fe, savepath="static/plots/velocity.png")
        plot_power_predictions(model, df_fe, savepath="static/plots/power.png")
        plot_feature_importance(model, savepath="static/plots/importance.png")

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    reply = chat_response(user_msg)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
