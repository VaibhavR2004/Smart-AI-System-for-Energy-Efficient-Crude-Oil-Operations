"""
Oil & Gas AI System — Pillar 4: Production Economics
=====================================================
Model 10 — Price Prediction & Cost Optimization

Trains:
  1. XGBoost  — primary oil price regressor
  2. LightGBM — ensemble complement
  3. Ridge     — linear baseline / stability anchor
  4. LP Solver — scipy.optimize.linprog cost optimizer (no pkl needed)

Data sources:
  A. EIA API v2  — weekly U.S. crude spot prices (live)
  B. FRED API    — macro indicators (CPI, USD Index)
  C. Synthetic   — well-level production + cost features

Saved artifacts (models/economics/ directory):
  models/economics/price_xgb.pkl
  models/economics/price_lgbm.pkl
  models/economics/price_ridge.pkl
  models/economics/scaler.pkl
  models/economics/feature_columns.pkl
  models/economics/meta.json
"""

import os
import json
import warnings
import joblib
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import EIA_API_KEY, FRED_API_KEY
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR    = "models/economics"
os.makedirs(MODEL_DIR, exist_ok=True)

# FIX 1: API keys must be read from env vars properly.
#         The original code passed the key string as the env var NAME,
#         which always returned "DEMO_KEY".
# EIA_API_KEY  = os.getenv("EIA_API_KEY",  "DEMO_KEY")   # export EIA_API_KEY="your_key"
EIA_API_KEY  = "EIA_API_KEY"        # export EIA_API_KEY="your_key"
FRED_API_KEY = os.getenv("FRED_API_KEY", "DEMO_KEY")   # export FRED_API_KEY="your_key"

# To hardcode for testing (not recommended for production):


RANDOM_STATE = 42
N_SPLITS     = 5          # TimeSeriesSplit folds
EIA_WEEKS    = 800        # ~15 years of weekly data


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA FETCHING — EIA + FRED APIs
# ──────────────────────────────────────────────────────────────────────────────

def fetch_eia_crude_prices(start: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetch WTI crude oil weekly spot prices from EIA API v2.
    Series: RWTC (Cushing, OK WTI Spot Price, weekly)
    Falls back to synthetic data if API is unavailable.
    """
    print("  [EIA] Fetching WTI crude spot prices …")

    if EIA_API_KEY == "DEMO_KEY":
        print("  [EIA] No API key set — using synthetic price data.")
        return _synthetic_prices(start)

    # FIX 2: Removed undefined `weeks` variable. Now uses `start` param
    #         to set the date range, consistent with function signature.
    end_date   = datetime.today()
    start_date = datetime.strptime(start, "%Y-%m-%d")

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key":            EIA_API_KEY,
        "frequency":          "weekly",
        "data[0]":            "value",
        "facets[series][]":   "RWTC",
        "start":              start_date.strftime("%Y-%m-%d"),
        "end":                end_date.strftime("%Y-%m-%d"),
        "sort[0][column]":    "period",
        "sort[0][direction]": "asc",
        "length":             EIA_WEEKS,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept":     "application/json",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        df = pd.DataFrame(data)[["period", "value"]].rename(
            columns={"period": "date", "value": "wti_price"}
        )
        df["date"]      = pd.to_datetime(df["date"])
        df["wti_price"] = pd.to_numeric(df["wti_price"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        print(f"  [EIA] Fetched {len(df)} weekly price records  "
              f"({df['date'].min().date()} → {df['date'].max().date()})")
        return df
    except Exception as e:
        print(f"  [EIA] API error: {e}  → falling back to synthetic data.")
        return _synthetic_prices(start)


def fetch_fred_macro(start: str = "2010-01-01") -> pd.DataFrame | None:
    """
    Fetch macro indicators from FRED:
      CPIAUCSL — Consumer Price Index (inflation proxy)
      DTWEXBGS — Trade-weighted USD index

    Returns weekly DataFrame for merging, or None on failure.
    """
    print("  [FRED] Fetching macro indicators …")

    if FRED_API_KEY == "DEMO_KEY":
        print("  [FRED] No API key set — skipping macro features.")
        return None

    series_map = {
        "CPIAUCSL": "cpi",
        "DTWEXBGS": "usd_index",
    }
    frames = []
    for series_id, col_name in series_map.items():
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}"
            f"&observation_start={start}&file_type=json"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            obs = r.json()["observations"]
            df  = pd.DataFrame(obs)[["date", "value"]].rename(
                columns={"value": col_name}
            )
            df["date"]   = pd.to_datetime(df["date"])
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            df = df.dropna()
            frames.append(df.set_index("date"))
        except Exception as e:
            print(f"  [FRED] {series_id} error: {e}")

    if not frames:
        return None

    macro = pd.concat(frames, axis=1).reset_index()
    macro = macro.set_index("date").resample("W").ffill().reset_index()
    print(f"  [FRED] Macro data fetched: {macro.columns.tolist()}")
    return macro


def _synthetic_prices(start: str) -> pd.DataFrame:
    """Generate realistic synthetic WTI price series using GBM + regime shocks."""
    np.random.seed(RANDOM_STATE)
    dates = pd.date_range(start=start, end=datetime.today(), freq="W")
    n     = len(dates)

    mu, sigma = 0.0002, 0.025
    returns   = np.random.normal(mu, sigma, n)
    prices    = [55.0]
    for r in returns[1:]:
        prices.append(prices[-1] * np.exp(r))

    prices = np.array(prices)
    prices[int(n * 0.50): int(n * 0.55)] *= 0.55   # crash
    prices[int(n * 0.55): int(n * 0.65)] *= 1.30   # recovery
    prices = np.clip(prices, 10, 140)

    return pd.DataFrame({"date": dates, "wti_price": prices})


# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def build_features(price_df: pd.DataFrame,
                   macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Create lag features, rolling stats, EMA, Bollinger Bands, and macro merges.
    Target: wti_price shifted -1 (next week's price).
    """
    df = price_df.copy().sort_values("date").reset_index(drop=True)

    # ── Lag features
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f"lag_{lag}w"] = df["wti_price"].shift(lag)

    # ── Rolling statistics
    for window in [4, 8, 13, 26]:
        df[f"roll_mean_{window}w"] = df["wti_price"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}w"]  = df["wti_price"].shift(1).rolling(window).std()

    # ── Momentum & trend
    df["mom_4w"]     = df["wti_price"].shift(1) - df["wti_price"].shift(5)
    df["mom_13w"]    = df["wti_price"].shift(1) - df["wti_price"].shift(14)
    df["pct_chg_1w"] = df["wti_price"].pct_change(1).shift(1)
    df["pct_chg_4w"] = df["wti_price"].pct_change(4).shift(1)

    # ── EMA
    df["ema_8w"]    = df["wti_price"].shift(1).ewm(span=8).mean()
    df["ema_26w"]   = df["wti_price"].shift(1).ewm(span=26).mean()
    df["ema_cross"] = df["ema_8w"] - df["ema_26w"]

    # ── Bollinger Band position
    roll_20        = df["wti_price"].shift(1).rolling(20)
    bb_upper       = roll_20.mean() + 2 * roll_20.std()
    bb_lower       = roll_20.mean() - 2 * roll_20.std()
    df["bb_pos"]   = (df["wti_price"].shift(1) - bb_lower) / \
                     (bb_upper - bb_lower + 1e-9)
    # FIX 3: bb_upper/bb_lower are intermediate columns — kept out of DataFrame
    #         to prevent them leaking into feature_cols and confusing the scaler.

    # ── Calendar features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["year"]         = df["date"].dt.year

    # ── Target: next week's price
    df["target"] = df["wti_price"].shift(-1)

    # ── Merge macro (optional)
    if macro_df is not None:
        macro_df = macro_df.copy()
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        df = pd.merge_asof(
            df.sort_values("date"),
            macro_df.sort_values("date"),
            on="date", direction="backward"
        )
        for col in ["cpi", "usd_index"]:
            if col in df.columns:
                df[f"{col}_chg"] = df[col].pct_change(4).shift(1)

    df = df.dropna().reset_index(drop=True)
    print(f"  [Features] Shape after engineering: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. WELL-LEVEL PRODUCTION COST FEATURES (synthetic)
# ──────────────────────────────────────────────────────────────────────────────

def build_well_cost_features(n_wells: int = 500) -> pd.DataFrame:
    """Synthetic well-level cost dataset for LP optimizer demo."""
    np.random.seed(RANDOM_STATE + 1)
    df = pd.DataFrame({
        "well_id":                np.arange(n_wells),
        "production_bpd":         np.random.lognormal(5.5, 0.8, n_wells).clip(10, 5000),
        "lift_cost_per_bbl":      np.random.normal(8, 3, n_wells).clip(1, 25),
        "transport_cost_per_bbl": np.random.normal(4, 1.5, n_wells).clip(0.5, 12),
        "royalty_rate":           np.random.uniform(0.10, 0.25, n_wells),
        "opex_fixed_daily":       np.random.normal(3000, 800, n_wells).clip(500, 8000),
        "wti_price_at_time":      np.random.normal(72, 18, n_wells).clip(25, 120),
    })
    df["total_var_cost"]     = df["lift_cost_per_bbl"] + df["transport_cost_per_bbl"]
    df["royalty_cost"]       = df["wti_price_at_time"] * df["royalty_rate"]
    df["breakeven_price"]    = (
        df["total_var_cost"] + df["royalty_cost"]
        + df["opex_fixed_daily"] / df["production_bpd"]
    )
    df["margin_per_bbl"]     = df["wti_price_at_time"] - df["breakeven_price"]
    df["total_daily_margin"] = df["margin_per_bbl"] * df["production_bpd"]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"    {name:10s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=40,
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


def train_ridge(X_train, y_train):
    return Ridge(alpha=10.0).fit(X_train, y_train)


def train_all_models(df: pd.DataFrame, feature_cols: list):
    """
    TimeSeriesSplit cross-validation → trains on all but the last fold,
    evaluates on the last (most recent) fold.
    """
    X = df[feature_cols].values
    y = df["target"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tscv.split(X_scaled))

    # Use last fold (most recent data) for final evaluation
    tr_idx, val_idx = splits[-1]
    X_tr,  X_val   = X_scaled[tr_idx], X_scaled[val_idx]
    y_tr,  y_val   = y[tr_idx],        y[val_idx]

    print("\n[Training] TimeSeriesSplit cross-validation (final fold) …")

    print("  Training XGBoost …")
    xgb_model   = train_xgboost(X_tr, y_tr, X_val, y_val)

    print("  Training LightGBM …")
    # FIX 4: LightGBM needs feature names for inference consistency.
    #         Train on DataFrame slice so feature names are stored in the booster.
    X_tr_df  = pd.DataFrame(X_tr,  columns=feature_cols)
    X_val_df = pd.DataFrame(X_val, columns=feature_cols)
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgbm_model.fit(
        X_tr_df, y_tr,
        eval_set=[(X_val_df, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )

    print("  Training Ridge …")
    ridge_model = train_ridge(X_tr, y_tr)

    print("\n[Evaluation] Validation set metrics:")
    metrics = {
        "xgboost":  evaluate("XGBoost",  y_val, xgb_model.predict(X_val)),
        "lightgbm": evaluate("LightGBM", y_val, lgbm_model.predict(X_val_df)),
        "ridge":    evaluate("Ridge",    y_val, ridge_model.predict(X_val)),
    }

    y_ens = (
        xgb_model.predict(X_val)
        + lgbm_model.predict(X_val_df)
        + ridge_model.predict(X_val)
    ) / 3.0
    metrics["ensemble"] = evaluate("Ensemble", y_val, y_ens)

    return xgb_model, lgbm_model, ridge_model, scaler, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 5. LP COST OPTIMIZER
# ──────────────────────────────────────────────────────────────────────────────

def solve_lp_cost(
    well_costs:      np.ndarray,
    production_caps: np.ndarray,
    transport_costs: np.ndarray,
    demand_bpd:      float,
    price_per_bbl:   float,
) -> dict:
    """
    Linear Program: minimise total extraction + transport cost
    subject to:
      Σ x_i  = demand_bpd           (meet demand exactly)
      0 ≤ x_i ≤ production_caps_i   (well capacity)
    """
    n   = len(well_costs)
    c   = well_costs + transport_costs

    A_eq  = np.ones((1, n))
    b_eq  = [demand_bpd]
    bounds = [(0, cap) for cap in production_caps]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        alloc      = result.x
        total_cost = float(np.dot(c, alloc))   # FIX 5: result.fun = c·x already,
        revenue    = price_per_bbl * demand_bpd  # but explicit dot is clearer & safer
        profit     = revenue - total_cost
        return {
            "status":           "optimal",
            "allocation_bpd":   alloc.tolist(),
            "total_cost_usd":   round(total_cost, 2),
            "revenue_usd":      round(revenue, 2),
            "profit_usd":       round(profit, 2),
            "avg_cost_per_bbl": round(total_cost / demand_bpd, 4),
        }
    return {"status": "infeasible", "message": result.message}


# ──────────────────────────────────────────────────────────────────────────────
# 6. SAVE ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────────

def save_models(xgb_model, lgbm_model, ridge_model, scaler, feature_cols, metrics):
    paths = {
        "xgboost":  f"{MODEL_DIR}/price_xgb.pkl",
        "lightgbm": f"{MODEL_DIR}/price_lgbm.pkl",
        "ridge":    f"{MODEL_DIR}/price_ridge.pkl",
        "scaler":   f"{MODEL_DIR}/scaler.pkl",
        "features": f"{MODEL_DIR}/feature_columns.pkl",
    }
    joblib.dump(xgb_model,    paths["xgboost"])
    joblib.dump(lgbm_model,   paths["lightgbm"])
    joblib.dump(ridge_model,  paths["ridge"])
    joblib.dump(scaler,       paths["scaler"])
    joblib.dump(feature_cols, paths["features"])

    meta = {
        "trained_at":    datetime.now().isoformat(),
        "eia_source":    "EIA API v2 — PET.RWTC.W (WTI Weekly)",
        "fred_source":   "FRED — CPIAUCSL, DTWEXBGS",
        "models":        list(paths.keys()),
        "feature_count": len(feature_cols),
        "metrics":       metrics,
    }
    with open(f"{MODEL_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Save] Artifacts saved to '{MODEL_DIR}/'")
    for k, p in paths.items():
        print(f"   ✓ {p}")
    print(f"   ✓ {MODEL_DIR}/meta.json")


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Oil & Gas AI — Economics Model Training")
    print("  Pillar 4: Price Prediction + Cost Optimization")
    print("=" * 60)

    # ── Fetch data
    print("\n[Step 1] Fetching market data …")
    price_df = fetch_eia_crude_prices(start="2010-01-01")
    macro_df = fetch_fred_macro(start="2010-01-01")

    # ── Feature engineering
    print("\n[Step 2] Building features …")
    feat_df = build_features(price_df, macro_df)

    # FIX 6: Exclude ALL non-feature columns including bb_upper/bb_lower
    #         which were accidentally included in the original exclude set comment
    #         but bb_upper/bb_lower are no longer added to df (fixed in build_features).
    exclude   = {"date", "wti_price", "target"}
    feat_cols = [c for c in feat_df.columns if c not in exclude]
    print(f"  Feature columns ({len(feat_cols)}): {feat_cols}")

    # ── Train models
    print("\n[Step 3] Training models …")
    xgb_m, lgbm_m, ridge_m, scaler, metrics = train_all_models(feat_df, feat_cols)

    # ── LP demo
    print("\n[Step 4] LP Cost Optimizer demo …")
    well_df   = build_well_cost_features(n_wells=20)
    lp_result = solve_lp_cost(
        well_costs      = well_df["lift_cost_per_bbl"].values,
        production_caps = well_df["production_bpd"].values,
        transport_costs = well_df["transport_cost_per_bbl"].values,
        demand_bpd      = 5000.0,
        price_per_bbl   = float(price_df["wti_price"].iloc[-1]),
    )
    print(f"  LP status : {lp_result['status']}")
    if lp_result["status"] == "optimal":
        print(f"  Total cost: ${lp_result['total_cost_usd']:,.2f} / day")
        print(f"  Profit    : ${lp_result['profit_usd']:,.2f} / day")
        print(f"  Avg cost  : ${lp_result['avg_cost_per_bbl']:.2f} / bbl")

    # ── Save
    print("\n[Step 5] Saving artifacts …")
    save_models(xgb_m, lgbm_m, ridge_m, scaler, feat_cols, metrics)

    print("\n[Done] Economics model training complete.")
    print("  → Use crude_oil_prediction_demo.py for inference")


if __name__ == "__main__":
    main()