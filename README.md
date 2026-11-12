# ⚡ Hourly Electricity Demand Forecasting (24-Hour Horizon)
Built an hourly electricity-demand forecasting pipeline (44K records, 2020–2024) that predicts 24 hours ahead - cleaned timestamps and sensor spikes, engineered cyclical, lag, and degree-hour features, and cut MAE by 56% versus a seasonal-naive baseline on a held-out year, with SHAP explanations.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-success.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-blueviolet.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Metrics-orange.svg)

## 📖 Overview

This repository contains an end-to-end time-series machine learning pipeline designed to forecast hourly electricity demand **24 hours ahead**. 

Built on ~44,000 hourly records (2020–2024), the project demonstrates rigorous data cleaning (handling missing values and sensor spikes), advanced feature engineering (cyclical encodings, lag features, degree-hours), and robust modeling using **LightGBM**. The pipeline includes time-series cross-validation, a held-out year for final testing, and deployment-ready artifacts.

**The model achieves a Mean Absolute Error (MAE) of ~102 MW and an MAPE of 2.16%, outperforming a seasonal-naïve baseline by over 56%.**

---

## 📊 Dataset

The dataset (`electricity demand dataset.csv`) contains ~44K hourly records with the following columns:
*   `Timestamp`, `hour`, `dayofweek`, `month`, `year`, `dayofyear` *(Calendar features)*
*   `Temperature` (°C)
*   `Humidity` (%)
*   `Demand` (MW) *(Target Variable)*

*Assumption:* The model is provided with the weather forecast for the hours being predicted. In production, feed it the weather *forecast* rather than actuals.

---

## 🛠️ Pipeline Architecture

The notebook follows a strict data science lifecycle to prevent data leakage and ensure robust performance:

### 1. Data Cleaning & Validation
*   **Timestamp Rebuilding:** Instead of relying on potentially missing calendar columns, the datetime index is rebuilt using the date string and the row's position within the 24-hour cycle. The calendar columns are used only as a cross-check.
*   **Missing Values:** Weather data is interpolated. Missing target values (`Demand`) are kept (to maintain the hourly sequence for lags) but flagged and excluded from training/scoring.
*   **Local Outlier Detection:** Instead of a global IQR (which fails on hourly profiles), each point is compared to the average of its immediate neighbors. Points deviating by >8 robust standard deviations are flagged as sensor errors and interpolated.

### 2. Exploratory Data Analysis (EDA)
*   **Time-Series Trends:** Shows clear seasonality (summer/winter peaks).
*   **Daily & Weekly Profiles:** Highlights distinct morning and evening peaks, and lower demand on weekends.
*   **Temperature Response:** Shows a clear non-linear, U-shaped relationship (high demand at low and high temperatures). 
*   **Correlations:** Cyclical encodings (`hour_sin`, `hour_cos`) avoid the mathematical trap of treating hour 23 as far away from hour 0.

### 3. Feature Engineering
Every feature is constructed to respect the 24-hour forecast horizon (no future leakage).
*   **Cyclical Encodings:** `sin`/`cos` transforms for hour, day of week, and day of year.
*   **Temperature-derived:** Heating Degree Hours (HDH) and Cooling Degree Hours (CDH) based on a 18°C–24°C comfort band.
*   **Lag Features:** Demand from 24h, 48h, and 168h (1 week) ago.
*   **Rolling Statistics:** 24h and 168h rolling means and standard deviations, shifted by 24 hours so only past data is used.

### 4. Modeling & Validation
*   **Baselines:** Seasonal-naïve (t-168h) and Yesterday (t-24h).
*   **LightGBM:** A gradient-boosted decision tree model.
*   **Time-Series CV:** Utilizes `TimeSeriesSplit` with a 24-hour gap between train and validation to prevent leakage. Early stopping is strictly based on a subset of the training data, not the validation fold.
*   **Honest Testing:** The final year (2024) is completely held out as a test set.

---

## 🖼️ Visual Insights

### Missing Values & Timestamp Cross-Check
![Timestamp Rebuild](image1_placeholder.png)
*Auditing missing values and ensuring the rebuilt datetime index perfectly aligns with the provided calendar columns before dropping them.*

### Demand Trends
![Hourly Demand](image2_placeholder.png)
*The raw target variable showing strong annual seasonality and daily periodicity.*

### Daily and Weekly Profiles
![Daily/Weekly Profiles](image3_placeholder.png)
*Average demand by hour of day and day of week, revealing morning/evening peaks and weekend drops.*

### Temperature Response
![Temperature Scatter](image4_placeholder.png)
*Non-linear relationship: Demand increases both in cold and hot weather.*

### Correlation Matrix
![Correlation Matrix](image5_placeholder.png)
*Heatmap showing the correlation between Demand, Temperature, Humidity, and cyclical time features.*

### Forecast vs. Actual
![Forecast vs Actual](image6_placeholder.png)
*First week of the test set (2024). The LightGBM forecast (orange) tightly tracks the actual demand (blue), heavily outperforming the seasonal-naïve baseline (green).*

---

## 📈 Results

The LightGBM model was cross-validated, retrained on the full training set, and evaluated on the held-out 2024 test set.

| Metric | Seasonal-Naïve Baseline | LightGBM Model | Improvement |
| :--- | :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | 234.82 MW | **102.28 MW** | **+56.4%** |
| **RMSE** (Root Mean Sq. Error) | 305.33 MW | **135.53 MW** | **+55.6%** |
| **MAPE** (Mean Abs. % Error) | 4.90% | **2.16%** | **+56.0%** |

> *The cross-validated median tree count (874) was used for the final model, preventing overfitting while ensuring early stopping is consistent.*

---

## 🚀 Deployment & Explainability

### Explainability
The pipeline includes built-in explainability using LightGBM's **gain-based feature importances** and **SHAP (SHapley Additive exPlanations)** values. This provides transparency into *why* the model made a specific prediction (e.g., highlighting how the weather and time of day drove the demand).

### Export Artifacts
The notebook produces two key files for downstream serving (e.g., via FastAPI or Streamlit):
1.  `forecast.csv`: Contains the test window's `datetime`, `actual`, `forecast`, and `baseline`.
2.  `demand_model.joblib`: A serialized artifact containing:
    *   The trained LightGBM model.
    *   The exact list of feature columns used.
    *   Configuration parameters (`horizon_hours`, `lags`, `comfort_low`, `comfort_high`, `history_needed_hours`).
    *   Test metrics (for model registry/versioning).

---

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/electricity-demand-forecasting.git
   cd electricity-demand-forecasting
