# Retail Demand Forecasting: ARIMA vs Prophet vs XGBoost

A end-to-end demand forecasting project benchmarking three approaches — classical time-series (ARIMA), additive regression (Prophet), and gradient boosting (XGBoost) — on weekly retail sales data modeled after the Walmart Store Sales dataset.

---

## Project Overview

| | |
|---|---|
| **Domain** | Retail / Demand Planning |
| **Dataset** | Walmart-style weekly sales (synthetic + Kaggle) |
| **Forecast Horizon** | 12-week holdout |
| **Models** | ARIMA, Prophet, XGBoost |
| **Evaluation Metrics** | MAPE, MAE, RMSE |
| **Experiment Tracking** | MLflow |

---

## Results

| Model | MAPE | MAE | RMSE |
|---|---|---|---|
| ARIMA(2,1,2) | 9.26% | $1,447 | $1,729 |
| XGBoost | 6.84% | $1,138 | $1,277 |
| **Prophet** | **3.99%** | **$652** | **$751** |

**Prophet** achieved the best accuracy (3.99% MAPE), driven by its built-in US holiday effects and promotional markdown regressor.

---

## Project Structure

```
demand_forecasting/
├── demand_forecasting.py   # Main modeling script
├── eda_plots.png           # EDA visualizations
├── model_comparison.png    # 12-week forecast comparison
├── feature_importance.png  # XGBoost feature importance
└── README.md
```

---

## Key Techniques

- **Feature Engineering**: Lag features (1-week, 4-week, 52-week), rolling averages, seasonal indicators, promotional markdown flag
- **Stationarity Testing**: ADF test to determine ARIMA differencing order
- **Promotional Uplift Modeling**: Markdown regressor added to Prophet for promotion impact estimation
- **Hierarchical-ready Design**: Pipeline structured to support store-level and department-level aggregations
- **Experiment Tracking**: MLflow logs MAPE/MAE/RMSE and model artifacts for all three runs

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib statsmodels prophet xgboost scikit-learn mlflow

# 2. Run the pipeline
python demand_forecasting.py

# 3. (Optional) View MLflow UI
mlflow ui
# Then open http://localhost:5000
```

To use the real Walmart dataset:
1. Download from https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
2. Replace the `generate_retail_sales()` call with `pd.read_csv("train.csv")`

---

## Skills Demonstrated

`Python` · `Time-Series Forecasting` · `ARIMA` · `Prophet` · `XGBoost` · `Feature Engineering` · `MAPE / MAE / RMSE` · `MLflow` · `EDA` · `Demand Planning` · `Pandas` · `NumPy` · `Matplotlib`
