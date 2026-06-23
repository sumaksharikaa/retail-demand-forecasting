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
| ARIMA(2,1,2) | 4.71% | $71,687 | $84,218 |
| Prophet | 3.02% | $46,033 | $64,044 |
| **XGBoost** | **2.61%** | **$39,864** | **$50,841** |

**XGBoost** achieved the best accuracy (2.61% MAPE), well below 
the industry benchmark of 5–10%, driven by lag features, rolling 
averages, and holiday indicators.
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

Dataset used:
1. Download from https://www.kaggle.com/datasets/mikhail1681/walmart-sales
2. File name: `Walmart_Sales.csv`
---

## Skills Demonstrated

`Python` · `Time-Series Forecasting` · `ARIMA` · `Prophet` · `XGBoost` · `Feature Engineering` · `MAPE / MAE / RMSE` · `MLflow` · `EDA` · `Demand Planning` · `Pandas` · `NumPy` · `Matplotlib`
