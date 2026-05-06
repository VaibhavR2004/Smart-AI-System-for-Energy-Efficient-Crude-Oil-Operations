"""
Crude Oil Price Prediction — Auto-Fetch Demo
Automatically pulls last 52 weeks of WTI prices from EIA API.
User only needs to provide their EIA API key.

Run: python crude_oil_prediction_demo.py
"""

import json
import pickle
import warnings
import joblib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — only thing the user needs to set
# ─────────────────────────────────────────────

EIA_API_KEY = "bD7wmetji5UHx8lXb4a0yIJ3Vb2Ox92WQQasFgqR"   # get free key at: https://www.eia.gov/opendata/register.php
MODEL_DIR   = Path("models/economics")

# ─────────────────────────────────────────────
# 1. Auto-fetch last 60 weeks of WTI from EIA
# ─────────────────────────────────────────────

def fetch_wti_prices(api_key: str, weeks: int = 60) -> pd.Series:
    """
    Fetch weekly WTI crude oil prices from EIA API v2.
    Returns a sorted Series indexed by date.
    """
    print(f"⏳ Fetching last {weeks} weeks of WTI prices from EIA...")

    end_date   = datetime.today()
    start_date = end_date - timedelta(weeks=weeks + 4)   # buffer for gaps

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key":           api_key,
        "frequency":         "weekly",
        "data[0]":           "value",
        "facets[series][]":  "RWTC",
        "start":             start_date.strftime("%Y-%m-%d"),
        "end":               end_date.strftime("%Y-%m-%d"),
        "sort[0][column]":   "period",
        "sort[0][direction]":"asc",
        "length":            weeks + 10,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    records = data.get("response", {}).get("data", [])

    if not records:
        raise ValueError("EIA returned no data. Check your API key or date range.")

    prices = pd.Series(
        {r["period"]: float(r["value"]) for r in records}
    ).sort_index()

    print(f"✅ Fetched {len(prices)} weekly WTI prices  "
          f"({prices.index[0]} → {prices.index[-1]})")
    print(f"   Latest price: ${prices.iloc[-1]:.2f} / barrel\n")

    return prices


def fetch_wti_prices_fallback() -> pd.Series:
    """
    Fallback: uses hardcoded recent prices if EIA API key is not set.
    Replace these with your own recent data if needed.
    """
    print("⚠️  No EIA API key set — using built-in recent WTI prices (fallback).")
    print("   Get a free key at https://www.eia.gov/opendata/register.php\n")

    # Approximate weekly WTI closes — update periodically
    dates  = pd.date_range(end=datetime.today(), periods=60, freq="W").strftime("%Y-%m-%d")
    values = [
        72.3, 73.1, 74.5, 73.8, 75.2, 76.0, 75.5, 74.9, 76.3, 77.1,
        76.8, 75.4, 74.2, 73.5, 72.8, 71.9, 73.0, 74.4, 75.8, 76.5,
        77.2, 78.0, 77.5, 76.9, 75.6, 74.8, 73.9, 72.5, 71.8, 70.9,
        71.5, 72.8, 74.0, 75.3, 76.1, 77.4, 78.2, 79.0, 78.5, 77.8,
        76.4, 75.0, 74.3, 73.6, 72.9, 74.1, 75.5, 76.8, 77.6, 78.4,
        79.1, 80.0, 79.5, 78.8, 77.2, 76.0, 75.3, 74.7, 76.1, 77.5,
    ]
    return pd.Series(dict(zip(dates, values)))


# ─────────────────────────────────────────────
# 2. Load model artifacts
# ─────────────────────────────────────────────

def load_artifacts():
    def load(fname):
        path = MODEL_DIR / fname
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    lgbm            = load("price_lgbm.pkl")
    xgb             = load("price_xgb.pkl")
    ridge           = load("price_ridge.pkl")
    scaler          = load("scaler.pkl")
    feature_columns = load("feature_columns.pkl")

    with open(MODEL_DIR / "meta.json") as f:
        meta = json.load(f)

    print("✅ All model artifacts loaded")
    print(f"   Models  : LightGBM | XGBoost | Ridge")
    print(f"   Features: {len(feature_columns)} columns\n")
    return lgbm, xgb, ridge, scaler, feature_columns, meta


# ─────────────────────────────────────────────
# 3. Feature engineering (internal — auto)
# ─────────────────────────────────────────────

def build_features(prices: pd.Series, feature_columns: list) -> pd.DataFrame:
    """Builds all 28 features from a price Series. Called internally."""
    s = prices.values.astype(float)
    p = s[-1]

    row = {
        # lags
        "lag_1w":        s[-2],
        "lag_2w":        s[-3],
        "lag_3w":        s[-4],
        "lag_4w":        s[-5],
        "lag_8w":        s[-9],
        "lag_12w":       s[-13],
        "lag_26w":       s[-27],
        "lag_52w":       s[-53] if len(s) >= 53 else s[0],
        # rolling
        "roll_mean_4w":  s[-4:].mean(),
        "roll_std_4w":   s[-4:].std(),
        "roll_mean_8w":  s[-8:].mean(),
        "roll_std_8w":   s[-8:].std(),
        "roll_mean_13w": s[-13:].mean(),
        "roll_std_13w":  s[-13:].std(),
        "roll_mean_26w": s[-26:].mean(),
        "roll_std_26w":  s[-26:].std(),
        # momentum
        "mom_4w":        p - s[-5],
        "mom_13w":       p - s[-13],
        # pct change
        "pct_chg_1w":    (p - s[-2]) / s[-2] * 100,
        "pct_chg_4w":    (p - s[-5]) / s[-5] * 100,
        # EMA
        "ema_8w":        pd.Series(s).ewm(span=8,  adjust=False).mean().iloc[-1],
        "ema_26w":       pd.Series(s).ewm(span=26, adjust=False).mean().iloc[-1],
        "ema_cross":     pd.Series(s).ewm(span=8,  adjust=False).mean().iloc[-1]
                       - pd.Series(s).ewm(span=26, adjust=False).mean().iloc[-1],
        # Bollinger Band position
        "bb_pos":        _bb_pos(s),
        # calendar
        "week_of_year":  int(datetime.today().isocalendar()[1]),
        "month":         datetime.today().month,
        "quarter":       (datetime.today().month - 1) // 3 + 1,
        "year":          datetime.today().year,
    }

    return pd.DataFrame([row])[feature_columns]


def _bb_pos(s):
    bb_mid   = s[-20:].mean()
    bb_std   = s[-20:].std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    return (s[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5


# ─────────────────────────────────────────────
# 4. Main prediction function (user-facing)
# ─────────────────────────────────────────────

def predict_next_week_wti(model: str = "ensemble") -> dict:
    """
    Predict next week's WTI crude oil price.
    Automatically fetches historical data — no input needed from user.

    Args:
        model : "ensemble" | "lgbm" | "xgb" | "ridge"

    Returns:
        dict with prediction results.

    Example:
        result = predict_next_week_wti()
        print(f"Next week WTI: ${result['predicted_price']} ({result['direction']})")
    """
    # fetch prices
    if EIA_API_KEY and EIA_API_KEY != "YOUR_EIA_API_KEY":
        prices = fetch_wti_prices(EIA_API_KEY, weeks=60)
    else:
        prices = fetch_wti_prices_fallback()

    if len(prices) < 52:
        raise ValueError(f"Not enough data: got {len(prices)}, need 52+")

    # build features
    features_df = build_features(prices, feature_columns)
    X_scaled    = scaler.transform(features_df.values)

    # predict
    preds = {
        "lgbm":  float(lgbm.predict(features_df)[0]),
        "xgb":   float(xgb.predict(X_scaled)[0]),
        "ridge": float(ridge.predict(X_scaled)[0]),
    }
    preds["ensemble"] = float(np.mean(list(preds.values())))

    current  = float(prices.iloc[-1])
    chosen   = preds[model]
    spread   = max(preds["lgbm"], preds["xgb"], preds["ridge"]) \
             - min(preds["lgbm"], preds["xgb"], preds["ridge"])

    return {
        "as_of_date":      prices.index[-1],
        "current_price":   round(current, 2),
        "predicted_price": round(chosen, 2),
        "change":          round(chosen - current, 2),
        "direction":       "UP ▲" if chosen > current else "DOWN ▼",
        "model_used":      model,
        "model_spread":    round(spread, 2),
        "confidence":      "HIGH" if spread < 2 else "MEDIUM" if spread < 5 else "LOW",
        "all_models": {k: round(v, 2) for k, v in preds.items()},
        "model_metrics":   meta["metrics"],
    }


# ─────────────────────────────────────────────
# 5. Run
# ─────────────────────────────────────────────

lgbm, xgb, ridge, scaler, feature_columns, meta = load_artifacts()

result = predict_next_week_wti(model="ensemble")

print("═" * 54)
print("  WTI CRUDE OIL — NEXT WEEK PRICE PREDICTION")
print("═" * 54)
print(f"  As of date       : {result['as_of_date']}")
print(f"  Current price    : ${result['current_price']:.2f} / barrel")
print(f"  Predicted price  : ${result['predicted_price']:.2f} / barrel")
print(f"  Expected change  : {result['change']:+.2f}")
print(f"  Direction        : {result['direction']}")
print(f"  Confidence       : {result['confidence']}  (spread ${result['model_spread']:.2f})")
print()
print("  Per-model breakdown:")
for m, v in result["all_models"].items():
    metrics = result["model_metrics"].get(m, {})
    r2      = metrics.get("r2", "-")
    print(f"    {m:<12} ${v:.2f}    R²={r2}")
print("═" * 54)