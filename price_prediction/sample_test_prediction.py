"""
Crude Oil Price Prediction — Correct Inference Demo
Source : EIA API v2 — WTI Weekly (PET.RWTC.W)
Features: lag, rolling, momentum, EMA, Bollinger Band, calendar

Run: python crude_oil_prediction_demo.py
"""

import json
import pickle
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. Load artifacts
# ─────────────────────────────────────────────

MODEL_DIR = Path("models/economics")

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

    return lgbm, xgb, ridge, scaler, feature_columns, meta

lgbm, xgb, ridge, scaler, feature_columns, meta = load_artifacts()

print("✅ All artifacts loaded")
print(f"   Feature count : {len(feature_columns)}")
print(f"   Features      : {feature_columns}\n")
print("   Model metrics :")
for m, v in meta["metrics"].items():
    print(f"     {m:<12} RMSE={v['rmse']}  MAE={v['mae']}  R²={v['r2']}")
print()

# ─────────────────────────────────────────────
# 2. Feature engineering from raw price history
#    Pass a list of recent weekly WTI close prices
#    (most recent LAST), at least 52 weeks needed.
# ─────────────────────────────────────────────

def build_features(weekly_prices: list) -> pd.DataFrame:
    """
    Build the 28 model features from a list of weekly WTI prices.

    Args:
        weekly_prices : list of floats, chronological order,
                        most recent price LAST.
                        Minimum length: 52 weeks.

    Returns:
        pd.DataFrame with exactly one row and all 28 feature columns.
    """
    if len(weekly_prices) < 52:
        raise ValueError(
            f"Need at least 52 weekly prices, got {len(weekly_prices)}"
        )

    s = pd.Series(weekly_prices, dtype=float)
    p = s.iloc[-1]   # latest price

    # ── lag features ──────────────────────────
    lag_1w  = s.iloc[-2]
    lag_2w  = s.iloc[-3]
    lag_3w  = s.iloc[-4]
    lag_4w  = s.iloc[-5]
    lag_8w  = s.iloc[-9]
    lag_12w = s.iloc[-13]
    lag_26w = s.iloc[-27]
    lag_52w = s.iloc[-53] if len(s) >= 53 else s.iloc[0]

    # ── rolling statistics ─────────────────────
    roll_mean_4w  = s.iloc[-4:].mean()
    roll_std_4w   = s.iloc[-4:].std()
    roll_mean_8w  = s.iloc[-8:].mean()
    roll_std_8w   = s.iloc[-8:].std()
    roll_mean_13w = s.iloc[-13:].mean()
    roll_std_13w  = s.iloc[-13:].std()
    roll_mean_26w = s.iloc[-26:].mean()
    roll_std_26w  = s.iloc[-26:].std()

    # ── momentum ──────────────────────────────
    mom_4w  = p - lag_4w
    mom_13w = p - lag_12w

    # ── percent change ─────────────────────────
    pct_chg_1w = (p - lag_1w) / lag_1w * 100
    pct_chg_4w = (p - lag_4w) / lag_4w * 100

    # ── exponential moving averages ────────────
    ema_8w    = s.ewm(span=8,  adjust=False).mean().iloc[-1]
    ema_26w   = s.ewm(span=26, adjust=False).mean().iloc[-1]
    ema_cross = ema_8w - ema_26w      # positive = bullish crossover

    # ── Bollinger Band position ────────────────
    bb_mid   = s.iloc[-20:].mean()
    bb_std   = s.iloc[-20:].std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos   = (p - bb_lower) / (bb_upper - bb_lower) if (bb_upper != bb_lower) else 0.5

    # ── calendar features ──────────────────────
    today        = pd.Timestamp.today()
    week_of_year = today.isocalendar().week
    month        = today.month
    quarter      = today.quarter
    year         = today.year

    row = {
        "lag_1w":        lag_1w,
        "lag_2w":        lag_2w,
        "lag_3w":        lag_3w,
        "lag_4w":        lag_4w,
        "lag_8w":        lag_8w,
        "lag_12w":       lag_12w,
        "lag_26w":       lag_26w,
        "lag_52w":       lag_52w,
        "roll_mean_4w":  roll_mean_4w,
        "roll_std_4w":   roll_std_4w,
        "roll_mean_8w":  roll_mean_8w,
        "roll_std_8w":   roll_std_8w,
        "roll_mean_13w": roll_mean_13w,
        "roll_std_13w":  roll_std_13w,
        "roll_mean_26w": roll_mean_26w,
        "roll_std_26w":  roll_std_26w,
        "mom_4w":        mom_4w,
        "mom_13w":       mom_13w,
        "pct_chg_1w":    pct_chg_1w,
        "pct_chg_4w":    pct_chg_4w,
        "ema_8w":        ema_8w,
        "ema_26w":       ema_26w,
        "ema_cross":     ema_cross,
        "bb_pos":        bb_pos,
        "week_of_year":  int(week_of_year),
        "month":         month,
        "quarter":       quarter,
        "year":          year,
    }

    return pd.DataFrame([row])[feature_columns]   # correct column order


# ─────────────────────────────────────────────
# 3. Sample weekly WTI prices (last 60 weeks)
#    Replace with your actual EIA weekly prices,
#    chronological order — most recent price LAST.
# ─────────────────────────────────────────────

sample_weekly_wti_prices = [
    104.5, 106.2, 108.0, 105.7, 102.3, 101.9, 103.5, 107.1, 110.4, 108.8,
    106.5, 104.2, 102.8, 101.3, 100.7, 99.8, 101.2, 103.6, 105.9, 107.3,
    109.1, 110.0, 108.7, 107.4, 105.8, 104.6, 103.2, 101.9, 100.5, 99.7,
    101.0, 102.6, 104.3, 106.7, 108.2, 109.8, 111.5, 113.0, 110.9, 108.6,
    106.3, 104.8, 103.7, 102.5, 101.6, 103.0, 105.4, 107.8, 109.6, 111.2,
    112.5, 113.8, 111.9, 109.7, 107.5, 105.9, 104.6, 103.8, 105.2, 106.9
]  # ← replace with real EIA weekly WTI closes

# ─────────────────────────────────────────────
# 4. Build features & scale
# ─────────────────────────────────────────────

features_df = build_features(sample_weekly_wti_prices)

print("── Engineered Features ──────────────────────────────")
print(features_df.T.rename(columns={0: "value"}).round(4).to_string())
print()

X_scaled = scaler.transform(features_df.values)   # pass numpy array to avoid warning

# ─────────────────────────────────────────────
# 5. Predict with all three models
# ─────────────────────────────────────────────

pred_lgbm     = float(lgbm.predict(features_df)[0])    # lgbm uses feature names
pred_xgb      = float(xgb.predict(X_scaled)[0])
pred_ridge    = float(ridge.predict(X_scaled)[0])
pred_ensemble = np.mean([pred_lgbm, pred_xgb, pred_ridge])

current_price = sample_weekly_wti_prices[-1]
spread        = max(pred_lgbm, pred_xgb, pred_ridge) - min(pred_lgbm, pred_xgb, pred_ridge)
direction     = "▲ UP" if pred_ensemble > current_price else "▼ DOWN"

print("── WTI Crude Oil Price Prediction ───────────────────")
print(f"  Current price  : ${current_price:.2f} / barrel")
print(f"  LightGBM       : ${pred_lgbm:.2f} / barrel  (RMSE={meta['metrics']['lightgbm']['rmse']})")
print(f"  XGBoost        : ${pred_xgb:.2f} / barrel  (RMSE={meta['metrics']['xgboost']['rmse']})")
print(f"  Ridge          : ${pred_ridge:.2f} / barrel  (RMSE={meta['metrics']['ridge']['rmse']})")
print(f"  Ensemble avg   : ${pred_ensemble:.2f} / barrel  (RMSE={meta['metrics']['ensemble']['rmse']})")
print()
print(f"  Direction      : {direction}  ({pred_ensemble - current_price:+.2f})")
print(f"  Model spread   : ${spread:.2f}  {'⚠️  high disagreement' if spread > 3 else '✅ models agree'}")
print()

# ─────────────────────────────────────────────
# 6. Clean inference function for production use
# ─────────────────────────────────────────────

def predict_wti_next_week(weekly_prices: list, model: str = "ensemble") -> dict:
    """
    Predict next week's WTI crude oil price.

    Args:
        weekly_prices : list of weekly WTI closing prices,
                        chronological, most recent LAST (min 52 values).
        model         : "lgbm" | "xgb" | "ridge" | "ensemble"

    Returns:
        dict with predicted price, direction, and per-model breakdown.

    Example:
        result = predict_wti_next_week(prices, model="ensemble")
        print(f"Next week: ${result['predicted_price']} ({result['direction']})")
    """
    df      = build_features(weekly_prices)
    X       = scaler.transform(df.values)

    preds = {
        "lgbm":  float(lgbm.predict(df)[0]),      # lgbm with feature names
        "xgb":   float(xgb.predict(X)[0]),
        "ridge": float(ridge.predict(X)[0]),
    }
    preds["ensemble"] = float(np.mean(list(preds.values())))

    chosen  = preds[model]
    current = weekly_prices[-1]

    return {
        "current_price":   round(current, 2),
        "predicted_price": round(chosen, 2),
        "change":          round(chosen - current, 2),
        "direction":       "UP" if chosen > current else "DOWN",
        "model_used":      model,
        "all_models":      {k: round(v, 2) for k, v in preds.items()},
    }


# ── Demo ──
result = predict_wti_next_week(sample_weekly_wti_prices, model="ensemble")
print("── predict_wti_next_week() output ───────────────────")
for k, v in result.items():
    print(f"  {k:<20}: {v}")