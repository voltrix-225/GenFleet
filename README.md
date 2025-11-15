
# ⚡ GenFleet – EV Telemetry Analytics Dashboard

GenFleet is a lightweight EV analytics platform that processes real BMW i3 telemetry, performs feature engineering, trains an ML model, and displays results through a clean Flask web dashboard with a built-in EV chatbot.

---

## 🚀 Features
* **Data Processing:** Cleans & standardizes raw tab-delimited EV telemetry
* **Trip Detection:** Auto-splits driving sessions and generates summaries
* **Feature Engineering:** Rolling smoothing + lag features + power calculation
* **ML Model:** XGBoost regression to predict instantaneous power consumption
* **Visualizations:** Velocity smoothing, power prediction accuracy, feature importance
* **Flask UI:** Run pipeline, view plots, access chatbot
* **Chatbot:** Answers EV-related questions (SoC, energy, heating load, model accuracy)

---

## 🏗️ Project Structure
```
GenFleet/
│
├── app.py                    # Flask server (UI + API)
├── chatbot.py                # EV analytics rule-based chatbot
├── data_processing.py        # Load, clean, preprocess telemetry
├── feature_engineering.py    # Rolling smoothing + lag features
├── model_train.py            # XGBoost model training
├── viz.py                    # Visualization functions (Agg backend)
├── helper.py                 # Utility helpers (sanitize, ensure_dir)
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
├── dataset/                  # Raw + combined telemetry files
│     └── Trip_data.csv
│
├── results/                  # Trip summaries / processed outputs
│     └── Trip_data_trip_summary.csv
│
├── templates/                # Flask HTML templates
│     ├── index.html          # Dashboard UI
│     └── chat.html           # Chat assistant UI
│
└── static/                   # Static assets served by Flask
      ├── style.css           # UI styling
      └── plots/              # Generated visualizations
            ├── velocity.png
            ├── power.png
            └── importance.png
```


## Authors

- [@voltrix-225](https://www.github.com/voltrix-225)

