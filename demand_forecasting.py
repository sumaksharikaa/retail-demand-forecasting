"""
Retail Demand Forecasting Project
===================================
Dataset  : Walmart Store Sales (Kaggle) — M5-style item-level weekly sales
Author   : Sumaksharika Nainavarapu
Goal     : Compare ARIMA, Prophet, and XGBoost for item-level demand forecasting
Metrics  : MAPE, MAE, RMSE
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import mlflow
import mlflow.sklearn

# ── 1. SYNTHETIC DATASET (mirrors Walmart Sales structure) ────────────────────
#   In a real project: df = pd.read_csv("train.csv") from Kaggle
#   Dataset URL: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

np.random.seed(42)

def generate_retail_sales(n_weeks=143, base=15000):
    """Generate synthetic weekly sales with trend, seasonality, and promotions."""
    weeks = pd.date_range(start="2010-02-05", periods=n_weeks, freq="W")
    trend = np.linspace(0, 3000, n_weeks)
    seasonality = 3000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    holiday_bump = np.zeros(n_weeks)
    # Simulate holiday spikes (Thanksgiving ~week 42, Christmas ~week 47)
    for yr in range(3):
        holiday_bump[42 + yr*52 : 48 + yr*52] += 5000
    noise = np.random.normal(0, 800, n_weeks)
    is_markdown = (np.random.rand(n_weeks) > 0.7).astype(int)  # promo flag
    promo_lift = is_markdown * np.random.uniform(500, 2000, n_weeks)

    sales = base + trend + seasonality + holiday_bump + promo_lift + noise
    df = pd.DataFrame({
        "date": weeks,
        "weekly_sales": sales.clip(min=0),
        "is_markdown": is_markdown,
        "store": 1,
        "dept": 1
    })
    return df

# Load real Walmart data
raw = pd.read_csv("train.csv")
raw["Date"] = pd.to_datetime(raw["Date"])

# Filter to one store + one department (e.g., Store 1, Dept 1)
df = raw[(raw["Store"] == 1) & (raw["Dept"] == 1)].copy()
df = df.rename(columns={"Date": "date", "Weekly_Sales": "weekly_sales"})
df["is_markdown"] = df["IsHoliday"].astype(int)  # use holiday flag as promo proxy
df = df[["date", "weekly_sales", "is_markdown", "Store", "Dept"]]
df = df.rename(columns={"Store": "store", "Dept": "dept"})
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset shape: {df.shape}")
print(df.head())


# ── 2. EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Retail Demand Forecasting — EDA", fontsize=14, fontweight="bold")

# 2a. Raw sales trend
axes[0, 0].plot(df["date"], df["weekly_sales"], color="#2E75B6", linewidth=1.5)
axes[0, 0].set_title("Weekly Sales Over Time")
axes[0, 0].set_xlabel("Date"); axes[0, 0].set_ylabel("Sales ($)")
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# 2b. Sales distribution
axes[0, 1].hist(df["weekly_sales"], bins=30, color="#1F4E79", edgecolor="white", alpha=0.8)
axes[0, 1].set_title("Sales Distribution")
axes[0, 1].set_xlabel("Weekly Sales ($)")

# 2c. Promo vs non-promo
promo_sales = df.groupby("is_markdown")["weekly_sales"].mean()
axes[1, 0].bar(["No Promo", "Markdown/Promo"], promo_sales.values, color=["#5B9BD5", "#1F4E79"])
axes[1, 0].set_title("Avg Sales: Promo vs No Promo")
axes[1, 0].set_ylabel("Avg Weekly Sales ($)")
for i, v in enumerate(promo_sales.values):
    axes[1, 0].text(i, v + 100, f"${v:,.0f}", ha="center", fontsize=10)

# 2d. Monthly seasonality
df["month"] = df["date"].dt.month
monthly = df.groupby("month")["weekly_sales"].mean()
axes[1, 1].plot(monthly.index, monthly.values, marker="o", color="#2E75B6", linewidth=2)
axes[1, 1].set_title("Average Sales by Month (Seasonality)")
axes[1, 1].set_xlabel("Month"); axes[1, 1].set_ylabel("Avg Sales ($)")
axes[1, 1].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
print("EDA plots saved → eda_plots.png")
plt.close()


# ── 3. TRAIN / TEST SPLIT (last 12 weeks = test) ─────────────────────────────
HOLDOUT = 12
train = df.iloc[:-HOLDOUT].copy()
test  = df.iloc[-HOLDOUT:].copy()
print(f"\nTrain: {len(train)} weeks | Test: {len(test)} weeks")


# ── 4. EVALUATION METRICS ─────────────────────────────────────────────────────

def evaluate(actual, predicted, model_name):
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"\n{model_name}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    return {"model": model_name, "MAPE": round(mape, 2), "MAE": round(mae, 0), "RMSE": round(rmse, 0)}


# ── 5. MODEL 1 — ARIMA ────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL 1: ARIMA")
print("="*50)

# Stationarity check
adf_result = adfuller(train["weekly_sales"])
print(f"ADF p-value: {adf_result[1]:.4f} ({'Stationary' if adf_result[1] < 0.05 else 'Non-stationary — differencing needed'})")

# Fit ARIMA(2,1,2) — typical starting point for weekly retail
arima_model = ARIMA(train["weekly_sales"], order=(2, 1, 2))
arima_fit   = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=HOLDOUT)
arima_metrics  = evaluate(test["weekly_sales"].values, arima_forecast.values, "ARIMA(2,1,2)")


# ── 6. MODEL 2 — PROPHET ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL 2: PROPHET")
print("="*50)

prophet_train = train[["date", "weekly_sales"]].rename(columns={"date": "ds", "weekly_sales": "y"})
prophet_test  = test[["date"]].rename(columns={"date": "ds"})

# Add US holidays and regressor for markdown promotions
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    changepoint_prior_scale=0.05
)
m.add_country_holidays(country_name="US")
m.add_regressor("is_markdown")

prophet_train["is_markdown"] = train["is_markdown"].values
prophet_test["is_markdown"]  = test["is_markdown"].values

m.fit(prophet_train)
prophet_forecast = m.predict(prophet_test)
prophet_metrics  = evaluate(test["weekly_sales"].values, prophet_forecast["yhat"].values, "Prophet")


# ── 7. MODEL 3 — XGBOOST (ML-based) ─────────────────────────────────────────
print("\n" + "="*50)
print("MODEL 3: XGBoost")
print("="*50)

def create_features(df):
    df = df.copy()
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["year"]        = df["date"].dt.year
    df["lag_1"]       = df["weekly_sales"].shift(1)
    df["lag_4"]       = df["weekly_sales"].shift(4)
    df["lag_52"]      = df["weekly_sales"].shift(52)
    df["rolling_4"]   = df["weekly_sales"].shift(1).rolling(4).mean()
    df["rolling_12"]  = df["weekly_sales"].shift(1).rolling(12).mean()
    return df

full_features = create_features(df)
FEATURE_COLS = ["week", "month", "quarter", "year", "is_markdown",
                "lag_1", "lag_4", "lag_52", "rolling_4", "rolling_12"]

xgb_train = full_features.iloc[:-HOLDOUT].dropna()
xgb_test  = full_features.iloc[-HOLDOUT:].dropna()

X_train = xgb_train[FEATURE_COLS]
y_train = xgb_train["weekly_sales"]
X_test  = xgb_test[FEATURE_COLS]
y_test  = xgb_test["weekly_sales"]

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_preds   = xgb_model.predict(X_test)
xgb_metrics = evaluate(y_test.values, xgb_preds, "XGBoost")


# ── 8. MODEL COMPARISON PLOT ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test["date"], test["weekly_sales"], label="Actual", color="black", linewidth=2, zorder=5)
ax.plot(test["date"], arima_forecast.values, label=f"ARIMA (MAPE {arima_metrics['MAPE']}%)", linestyle="--", color="#C00000")
ax.plot(test["date"], prophet_forecast["yhat"].values, label=f"Prophet (MAPE {prophet_metrics['MAPE']}%)", linestyle="--", color="#2E75B6")
ax.plot(xgb_test["date"], xgb_preds, label=f"XGBoost (MAPE {xgb_metrics['MAPE']}%)", linestyle="--", color="#70AD47")
ax.set_title("12-Week Demand Forecast: ARIMA vs Prophet vs XGBoost", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nModel comparison plot saved → model_comparison.png")
plt.close()


# ── 9. FEATURE IMPORTANCE (XGBoost) ──────────────────────────────────────────

importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance["feature"], importance["importance"], color="#1F4E79")
ax.set_title("XGBoost Feature Importance", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Feature importance plot saved → feature_importance.png")
plt.close()


# ── 10. MLFLOW EXPERIMENT TRACKING ───────────────────────────────────────────

mlflow.set_experiment("retail_demand_forecasting")

for metrics in [arima_metrics, prophet_metrics, xgb_metrics]:
    with mlflow.start_run(run_name=metrics["model"]):
        mlflow.log_param("model_type", metrics["model"])
        mlflow.log_metric("MAPE", metrics["MAPE"])
        mlflow.log_metric("MAE",  metrics["MAE"])
        mlflow.log_metric("RMSE", metrics["RMSE"])
        mlflow.log_artifact("eda_plots.png")
        mlflow.log_artifact("model_comparison.png")
        if metrics["model"] == "XGBoost":
            import mlflow.xgboost
            mlflow.xgboost.log_model(xgb_model, "xgboost_model")

print("\nMLflow experiment runs logged.")


# ── 11. RESULTS SUMMARY ───────────────────────────────────────────────────────

results = pd.DataFrame([arima_metrics, prophet_metrics, xgb_metrics])
print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
print(results.to_string(index=False))
best = results.loc[results["MAPE"].idxmin(), "model"]
print(f"\nBest model by MAPE: {best}")
