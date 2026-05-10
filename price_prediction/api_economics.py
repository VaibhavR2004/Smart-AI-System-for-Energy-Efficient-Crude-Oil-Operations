"""
Oil & Gas AI System — Pillar 4: Production Economics
=====================================================
api_economics.py — FastAPI Router

Plug into your main FastAPI app:

    from economics.api_economics import router as economics_router
    app.include_router(economics_router, prefix="/api/v1")

Endpoints:
    POST /predict/price     — Ensemble oil price forecast
    POST /optimize/cost     — LP production cost optimizer
    GET  /economics/health  — Model status + last training info
    GET  /economics/price/live — Fetch current WTI from EIA (live)
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from price_prediction.predictor import (
    predict_price,
    optimize_cost,
    get_current_wti_price
)
router = APIRouter(tags=["Economics — Price & Cost"])

EIA_API_KEY = os.getenv("EIA_API_KEY", "DEMO_KEY")


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class PriceFeatures(BaseModel):
    """
    Features for oil price prediction.
    Provide as many lag/rolling features as available.
    The model will fill missing ones with 0.
    """
    lag_1w:        float = Field(...,  description="WTI price 1 week ago ($/bbl)")
    lag_2w:        float = Field(0.0,  description="WTI price 2 weeks ago")
    lag_4w:        float = Field(0.0,  description="WTI price 4 weeks ago")
    lag_8w:        float = Field(0.0,  description="WTI price 8 weeks ago")
    lag_12w:       float = Field(0.0)
    lag_26w:       float = Field(0.0)
    lag_52w:       float = Field(0.0)
    roll_mean_4w:  float = Field(0.0,  description="4-week rolling average price")
    roll_std_4w:   float = Field(0.0)
    roll_mean_13w: float = Field(0.0)
    roll_std_13w:  float = Field(0.0)
    roll_mean_26w: float = Field(0.0)
    ema_8w:        float = Field(0.0)
    ema_26w:       float = Field(0.0)
    ema_cross:     float = Field(0.0,  description="EMA8 - EMA26 momentum signal")
    mom_4w:        float = Field(0.0)
    mom_13w:       float = Field(0.0)
    pct_chg_1w:    float = Field(0.0)
    pct_chg_4w:    float = Field(0.0)
    bb_pos:        float = Field(0.5,  description="Bollinger Band position (0=lower, 1=upper)")
    month:         int   = Field(1,    ge=1, le=12)
    quarter:       int   = Field(1,    ge=1, le=4)
    week_of_year:  int   = Field(1,    ge=1, le=53)
    year:          int   = Field(2025, ge=2000, le=2100)
    # Optional macro (if FRED data was included during training)
    cpi:           Optional[float] = Field(None)
    usd_index:     Optional[float] = Field(None)
    cpi_chg:       Optional[float] = Field(None)
    usd_index_chg: Optional[float] = Field(None)
    # Optional ensemble weights
    xgb_weight:    float = Field(0.34, ge=0, le=1)
    lgbm_weight:   float = Field(0.33, ge=0, le=1)
    ridge_weight:  float = Field(0.33, ge=0, le=1)


class WellInput(BaseModel):
    well_id:               str   = Field(...,  description="Unique well identifier")
    production_cap_bpd:    float = Field(...,  gt=0, description="Max daily production (bbl/day)")
    lift_cost_per_bbl:     float = Field(...,  gt=0, description="Variable extraction cost ($/bbl)")
    transport_cost_per_bbl:float = Field(...,  gt=0, description="Pipeline/transport cost ($/bbl)")
    royalty_rate:          float = Field(0.15, ge=0, le=0.5, description="Royalty fraction (0–0.5)")


class CostOptimizeRequest(BaseModel):
    wells:      list[WellInput] = Field(..., min_length=1)
    demand_bpd: float           = Field(..., gt=0, description="Total daily production target (bbl/day)")
    wti_price:  Optional[float] = Field(None, description="WTI spot price ($/bbl). If omitted, fetches live.")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/economics/health")
def economics_health():
    """Returns model status and metadata."""
    from pathlib import Path
    import json
    meta_path = Path(__file__).parent / "models" / "economics" / "meta.json"
    if not meta_path.exists():
        return {
            "status": "models_not_trained",
            "message": "Run train_economics.py to train and save models.",
        }
    with open(meta_path) as f:
        meta = json.load(f)
    return {"status": "ready", **meta}


@router.get("/economics/price/live")
def get_live_wti():
    """
    Fetch current WTI crude spot price from EIA API.
    Requires EIA_API_KEY environment variable.
    """
    price = get_current_wti_price(api_key=EIA_API_KEY)
    if price is None:
        raise HTTPException(
            status_code=503,
            detail="EIA API unavailable or API key not set. "
                   "Set EIA_API_KEY env var from https://www.eia.gov/opendata/"
        )
    return {"wti_price_usd_per_bbl": price, "source": "EIA API v2"}


@router.post("/predict/price")
def price_prediction(body: PriceFeatures):
    """
    Predict next-week WTI oil price using XGBoost + LightGBM + Ridge ensemble.

    Provide lag features based on recent price history.
    The model automatically weights predictions from all three models.
    """
    try:
        features = body.model_dump(exclude={"xgb_weight", "lgbm_weight", "ridge_weight"})
        # Remove None macro fields (only include if model was trained with them)
        features = {k: v for k, v in features.items() if v is not None}

        weights = {
            "xgboost":  body.xgb_weight,
            "lightgbm": body.lgbm_weight,
            "ridge":    body.ridge_weight,
        }

        result = predict_price(features, weights=weights)
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@router.post("/optimize/cost")
def cost_optimization(body: CostOptimizeRequest):
    """
    LP-based production cost optimizer.

    Allocates barrels/day across wells to meet a production target
    at minimum total cost (extraction + transport + royalties).

    If wti_price is omitted, fetches live price from EIA API.
    """
    # Resolve WTI price
    wti_price = body.wti_price
    if wti_price is None:
        wti_price = get_current_wti_price(api_key=EIA_API_KEY)
        if wti_price is None:
            raise HTTPException(
                status_code=422,
                detail="wti_price not provided and EIA API unavailable. "
                       "Please supply wti_price explicitly."
            )

    wells_list = [w.model_dump() for w in body.wells]

    try:
        result = optimize_cost(
            wells      = wells_list,
            demand_bpd = body.demand_bpd,
            wti_price  = wti_price,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimizer error: {e}")

    if result["status"] == "infeasible":
        raise HTTPException(status_code=422, detail=result.get("message", "Infeasible"))

    return result
