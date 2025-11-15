"""
chatbot.py
A lightweight EV analytics assistant for the GenFleet dashboard.
This assistant answers questions related to:
- Trip summaries
- Energy usage
- SoC behavior
- Power consumption
- Model performance (MAE, R²)
- Heating/Aux load effects
- Driving patterns

It does NOT require external APIs and works fully offline.
"""

import random


def chat_response(msg: str) -> str:
    if not msg:
        return "I didn't catch that. Could you ask again?"

    msg = msg.lower().strip()

    # --------------------------
    # 1️⃣ EV ENERGY QUESTIONS
    # --------------------------
    if "energy" in msg or "kwh" in msg or "consumption" in msg:
        return (
            "EV energy consumption is estimated from battery voltage × current over time. "
            "Trips with higher speed, acceleration, or HVAC usage show higher Wh/km. "
            "Check the 'trip_summary.csv' file to compare per-trip energy usage."
        )

    # --------------------------
    # 2️⃣ RANGE / EFFICIENCY
    # --------------------------
    if "range" in msg or "efficiency" in msg or "km" in msg:
        return (
            "Range mainly depends on: average speed, temperature, SoC drop, and heating load. "
            "Cold weather and aggressive acceleration reduce efficiency. "
            "You can visualize this using the power prediction graph in the dashboard."
        )

    # --------------------------
    # 3️⃣ SOC / BATTERY QUESTIONS
    # --------------------------
    if "soc" in msg or "state of charge" in msg or "battery" in msg:
        return (
            "SoC (State of Charge) represents remaining battery %. "
            "Large SoC drops during short trips indicate high energy draw. "
            "Check 'soc_drop_pct' in your trip summary to compare driving patterns."
        )

    # --------------------------
    # 4️⃣ HEATING / AC / AUXILIARY LOADS
    # --------------------------
    if "heat" in msg or "heater" in msg or "ac" in msg or "hvac" in msg:
        return (
            "HVAC loads significantly affect EV range, especially in winter. "
            "In your dataset, 'Heating Power CAN [kW]' represents heating energy usage. "
            "Higher heating load → higher power demand → faster SoC drop."
        )

    # --------------------------
    # 5️⃣ MODEL PERFORMANCE
    # --------------------------
    if "model" in msg or "accuracy" in msg or "mae" in msg or "score" in msg or "r2" in msg:
        return (
            "Model performance is shown in the pipeline output. "
            "MAE gives average error in watts, while R² indicates how well the model "
            "explains real power behaviour. Values near 1.0 show strong predictive accuracy."
        )

    # --------------------------
    # 6️⃣ DRIVING BEHAVIOR QUESTIONS
    # --------------------------
    if "speed" in msg or "acceleration" in msg or "throttle" in msg:
        return (
            "Driving behaviour strongly influences energy usage. "
            "High acceleration, high torque, and stop-and-go conditions increase power draw. "
            "You can check smoothed velocity and acceleration curves in the visualization tab."
        )

    # --------------------------
    # 7️⃣ ABOUT VISUALIZATIONS
    # --------------------------
    if "plot" in msg or "graph" in msg or "visual" in msg:
        return (
            "The dashboard generates three graphs: "
            "1) Raw vs Smoothed Velocity, "
            "2) Actual vs Predicted Power, "
            "3) Feature Importance. "
            "These help you understand power behaviour and model accuracy."
        )

    # --------------------------
    # 8️⃣ GENERAL HELP COMMANDS
    # --------------------------
    if "help" in msg or "what can you do" in msg:
        return (
            "I can help you understand:\n"
            "- Energy consumption (Wh, kWh)\n"
            "- SoC behaviour and battery usage\n"
            "- Range and efficiency trends\n"
            "- Heating/Aux load impact\n"
            "- Model performance (MAE, R²)\n"
            "- Driving behaviour metrics\n"
            "Just ask me anything related to EV analytics!"
        )

    # --------------------------
    # 9️⃣ FUN RANDOM AI REPLIES
    # --------------------------
    fun_replies = [
        "I'm analyzing those electrons right now ⚡",
        "EVs are cool — but smart EV analytics are cooler 😎",
        "Did you check the power prediction chart? It’s beautiful.",
        "Try asking me about SoC or range!",
        "Your dataset has some interesting driving behaviour patterns!"
    ]

    return random.choice(fun_replies)
