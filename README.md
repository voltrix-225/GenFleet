#  GenFleet: Synthetic EV Fleet Intelligence

## Project Overview
**GenFleet** is an AI-driven framework designed to **simulate, generate, and predict electric vehicle (EV) fleet performance** under diverse environmental and operational conditions.  
The core aim is to bridge the data scarcity gap in EV telematics — where real-world data is limited or inconsistent — by using **synthetic data generation** combined with **machine learning prediction models**.

By leveraging both **real trip data** and **AI-generated synthetic data**, GenFleet aspires to:
- Model real-world driving behavior and environmental effects (temperature, elevation, speed, HVAC use).  
- Predict key metrics such as **range**, **state-of-charge (SoC) drop**, and **battery performance**.  
- Support **fleet operators and EV researchers** in designing efficient charging and routing strategies.  

Ultimately, GenFleet acts as a **foundation for large-scale EV data augmentation and analytics**, merging generative AI with traditional regression and forecasting models.

---

## Technical Direction
GenFleet operates on two parallel axes:

1. **Synthetic Data Generation** — using a seed dataset to build generative models that can produce new, realistic trip profiles under varying conditions.  
2. **Predictive Modeling** — training XGBoost and similar ML models to forecast SoC consumption, trip energy, and performance metrics using combined real + synthetic datasets.

Future stages will integrate these two components, allowing the generator to feed the predictor dynamically — creating a **self-evolving EV fleet simulator**.

---

##  Week 1 — Data Acquisition & Cleaning
**Goal:** Build a reliable, unified dataset from raw trip CSVs.

###  Tasks Completed:
- Ingested trip logs across summer (partial data) and winter (complete data) conditions.  
- Handled tab-separated and non-UTF encodings (`\t`, `latin1`) to prevent parsing errors.  
- Combined all trips into a single cleaned dataset **`Trip_data.csv`** with traceable file origins.  
- Removed duplicates and missing-value rows; added basic structure for later EDA and model input preparation.  

**Output:** `Trip_data.csv` — a standardized dataset ready for exploration, feature normalization, and generative modeling.


