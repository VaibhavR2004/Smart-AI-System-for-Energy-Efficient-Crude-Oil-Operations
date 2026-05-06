"""
Oil & Gas AI System — Pillar 4: Production Economics
=====================================================
predictor.py — Inference Layer

Loads saved .pkl models and exposes two clean functions:

  predict_price(features: dict) -> dict
      Ensemble oil price prediction using XGBoost + LightGBM + Ridge.

  optimize_cost(wells: list[dict], demand_bpd: float, wti_price: float) -> dict
      LP-based cost optimizer (no pkl — runs scipy at request time).

Import this module into your FastAPI app:
  from economics.predictor import predict_price, optimize_cost
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (lazy singleton — loaded once on first call)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "models" / "economics"

_models  = {}
_scaler  = None
_feat_cols = None
_meta    = None


def _load_models():
    """Load all economics models into memory (called once)."""
    global _scaler, _feat_cols, _meta

    required = ["price_xgb.pkl", "price_lgbm.pkl", "price_ridge.pkl",
                "scaler.pkl", "feature_columns.pkl"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Economics models not found: {missing}\n"
            f"Run train_economics.py first."
        )

    _models["xgboost"]  = joblib.load(MODEL_DIR / "price_xgb.pkl")
    _models["lightgbm"] = joblib.load(MODEL_DIR / "price_lgbm.pkl")
    _models["ridge"]    = joblib.load(MODEL_DIR / "price_ridge.pkl")
    _scaler             = joblib.load(MODEL_DIR / "scaler.pkl")
    _feat_cols          = joblib.load(MODEL_DIR / "feature_columns.pkl")

    meta_path = MODEL_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _meta = json.load(f)

    print(f"[Economics] Models loaded from {MODEL_DIR}")
    if _meta:
        print(f"  Trained at : {_meta.get('trained_at', 'unknown')}")
        print(f"  Features   : {_meta.get('feature_count', '?')}")


def _ensure_loaded():
    if not _models:
        _load_models()


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRICE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_price(features: dict, weights: dict | None = None) -> dict:
    """
    Predict next-week WTI oil price using ensemble of trained models.

    Parameters
    ----------
    features : dict
        Key-value pairs matching the trained feature columns.
        Minimum required: lag_1w (last week's price).
        Any missing features are filled with 0 (model will degrade gracefully).

    weights : dict, optional
        Custom ensemble weights, e.g. {"xgboost": 0.5, "lightgbm": 0.3, "ridge": 0.2}
        Defaults to equal weighting.

    Returns
    -------
    dict with keys:
        predicted_price   — ensemble price forecast ($/bbl)
        model_predictions — individual model outputs
        confidence_range  — ±1 std of model outputs
        feature_count     — how many features were matched
        meta              — training metadata
    """
    _ensure_loaded()

    # Build feature vector in the correct column order
    row = {col: features.get(col, 0.0) for col in _feat_cols}
    X   = np.array([list(row.values())])
    X_s = _scaler.transform(X)

    preds = {
        "xgboost":  float(_models["xgboost"].predict(X_s)[0]),
        "lightgbm": float(_models["lightgbm"].predict(X_s)[0]),
        "ridge":    float(_models["ridge"].predict(X_s)[0]),
    }

    # Ensemble weighting
    if weights is None:
        weights = {"xgboost": 1/3, "lightgbm": 1/3, "ridge": 1/3}

    total_w = sum(weights.values())
    ensemble_price = sum(
        preds[m] * weights.get(m, 0) / total_w for m in preds
    )

    pred_vals = list(preds.values())
    confidence_std = float(np.std(pred_vals))

    matched = sum(1 for col in _feat_cols if col in features)

    return {
        "predicted_price":   round(ensemble_price, 4),
        "model_predictions": {k: round(v, 4) for k, v in preds.items()},
        "confidence_range": {
            "low":  round(ensemble_price - confidence_std, 4),
            "high": round(ensemble_price + confidence_std, 4),
        },
        "feature_count": {
            "matched":  matched,
            "expected": len(_feat_cols),
        },
        "meta": {
            "trained_at": _meta.get("trained_at") if _meta else None,
            "eia_source": _meta.get("eia_source") if _meta else None,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. COST OPTIMIZATION (LP — no pkl, runs at request time)
# ──────────────────────────────────────────────────────────────────────────────

def optimize_cost(
    wells:       list[dict],
    demand_bpd:  float,
    wti_price:   float,
) -> dict:
    """
    LP cost optimizer: allocate production across wells to meet demand
    at minimum total cost.

    Parameters
    ----------
    wells : list of dicts, each with keys:
        well_id              — identifier
        production_cap_bpd   — max daily capacity
        lift_cost_per_bbl    — variable extraction cost $/bbl
        transport_cost_per_bbl — pipeline/trucking cost $/bbl
        royalty_rate         — fraction of revenue (0.10–0.25)

    demand_bpd : float
        Total barrels/day to be produced.

    wti_price : float
        Current WTI spot price ($/bbl).  Used to compute royalty cost and profit.

    Returns
    -------
    dict with:
        status             — "optimal" | "infeasible"
        allocation         — per-well allocation list
        total_cost_usd     — total daily operating cost
        revenue_usd        — daily revenue at wti_price
        profit_usd         — revenue - cost
        avg_cost_per_bbl   — total_cost / demand_bpd
        utilization        — % of total capacity used
    """
    n = len(wells)
    if n == 0:
        return {"status": "error", "message": "No wells provided."}

    caps       = np.array([w.get("production_cap_bpd",    100.0)  for w in wells])
    lift       = np.array([w.get("lift_cost_per_bbl",       8.0)  for w in wells])
    transport  = np.array([w.get("transport_cost_per_bbl",  4.0)  for w in wells])
    royalty_r  = np.array([w.get("royalty_rate",            0.15) for w in wells])

    royalty_cost_per_bbl = wti_price * royalty_r
    c = lift + transport + royalty_cost_per_bbl  # total variable cost vector

    # Equality: Σ x_i = demand_bpd
    A_eq = np.ones((1, n))
    b_eq = [demand_bpd]

    # Bounds: 0 ≤ x_i ≤ cap_i
    bounds = [(0, cap) for cap in caps]

    if demand_bpd > caps.sum():
        return {
            "status":  "infeasible",
            "message": f"Total capacity ({caps.sum():.1f} bpd) < demand ({demand_bpd:.1f} bpd)",
        }

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        return {"status": "infeasible", "message": result.message}

    alloc       = result.x
    total_cost  = float(np.dot(c, alloc))
    revenue     = wti_price * demand_bpd
    profit      = revenue - total_cost
    utilization = demand_bpd / caps.sum() * 100

    allocation_detail = [
        {
            "well_id":          wells[i].get("well_id", i),
            "allocated_bpd":    round(float(alloc[i]), 2),
            "cost_per_bbl":     round(float(c[i]), 4),
            "daily_cost_usd":   round(float(c[i] * alloc[i]), 2),
        }
        for i in range(n)
        if alloc[i] > 0.01   # exclude zero-allocation wells
    ]

    return {
        "status":            "optimal",
        "allocation":        allocation_detail,
        "total_cost_usd":    round(total_cost, 2),
        "revenue_usd":       round(revenue, 2),
        "profit_usd":        round(profit, 2),
        "avg_cost_per_bbl":  round(total_cost / demand_bpd, 4),
        "utilization_pct":   round(utilization, 2),
        "wti_price_used":    wti_price,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. LIVE PRICE FETCH UTILITY (call before inference for freshness)
# ──────────────────────────────────────────────────────────────────────────────

def get_current_wti_price(api_key: str | None = None) -> float | None:
    """
    Fetch latest WTI spot price from EIA API.
    Returns price as float, or None if unavailable.

    Usage in FastAPI startup / background task:
        price = get_current_wti_price(api_key=EIA_API_KEY)
    """
    import os, requests
    key = api_key or os.getenv("EIA_API_KEY", "DEMO_KEY")
    if key == "DEMO_KEY":
        return None

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key":           key,
        "frequency":         "weekly",
        "data[0]":           "value",
        "facets[product][]": "EPCWTI",
        "facets[duoarea][]": "NUS",
        "sort[0][column]":   "period",
        "sort[0][direction]":"desc",
        "length":            1,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        val = r.json()["response"]["data"][0]["value"]
        return float(val)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("Testing predict_price …")
    try:
        result = predict_price({
            "lag_1w": 75.0, "lag_2w": 73.5, "lag_4w": 70.0,
            "lag_8w": 68.0, "lag_52w": 65.0,
            "roll_mean_4w": 72.0, "roll_std_4w": 3.0,
            "roll_mean_13w": 71.0, "ema_8w": 72.5, "ema_26w": 70.0,
            "ema_cross": 2.5, "mom_4w": 5.0, "mom_13w": 7.0,
            "pct_chg_1w": 0.02, "pct_chg_4w": 0.05,
            "bb_pos": 0.65, "month": 6, "quarter": 2,
            "week_of_year": 24, "year": 2025,
        })
        print(f"  Predicted WTI: ${result['predicted_price']:.2f}/bbl")
        print(f"  Range: ${result['confidence_range']['low']:.2f} – "
              f"${result['confidence_range']['high']:.2f}")
    except FileNotFoundError as e:
        print(f"  [Skip] {e}")

    print("\nTesting optimize_cost …")
    test_wells = [
        {"well_id": f"W{i}", "production_cap_bpd": 300,
         "lift_cost_per_bbl": 7 + i, "transport_cost_per_bbl": 3.5,
         "royalty_rate": 0.15}
        for i in range(10)
    ]
    lp = optimize_cost(test_wells, demand_bpd=1500, wti_price=78.50)
    print(f"  Status  : {lp['status']}")
    print(f"  Profit  : ${lp['profit_usd']:,.2f}/day")
    print(f"  Avg cost: ${lp['avg_cost_per_bbl']:.2f}/bbl")
