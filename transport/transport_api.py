"""
transport_api.py
================
FastAPI endpoint integration for the Transport Model (Pillar 3).
Plugs into the existing Oil & Gas AI API layer.

Endpoints:
    POST /route
        Body : { "lat": float, "lon": float, "region": str|null }
        Returns: nearest refinery info + pipeline route + distances

    GET  /route/regions
        Returns list of available regions

    GET  /route/health
        Returns cache status

Run standalone:
    uvicorn transport_api:app --host 0.0.0.0 --port 8003 --reload
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transport_model import transport, REGION_FILES

app = FastAPI(
    title="Oil & Gas Transport Model API",
    description="Nearest refinery lookup + pipeline route optimisation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    lat:    float = Field(..., description="Query latitude  (e.g. 29.76 for Houston TX)")
    lon:    float = Field(..., description="Query longitude (e.g. -95.37)")
    region: Optional[str] = Field(
        None,
        description="Force a specific region: canada | mexico | usa | saudi_arabia. "
                    "Omit for auto-detection from coordinates.",
    )


class Waypoint(BaseModel):
    lat:  float
    lon:  float
    name: str


class NearestRefinery(BaseModel):
    name:     str
    lat:      float
    lon:      float
    dist_km:  float
    region:   str


class RouteResponse(BaseModel):
    query:             dict
    nearest_refinery:  NearestRefinery
    route:             list[Waypoint]
    total_graph_km:    float
    direct_km:         float
    hops:              int


class NearestRequest(BaseModel):
    lat:    float
    lon:    float
    region: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse, tags=["Transport"])
async def find_route(req: RouteRequest):
    """
    Find the nearest refinery and optimal pipeline route from a given location.

    - **lat / lon**: your well or field location
    - **region**: optional — if omitted, the model auto-detects from coordinates

    Returns the nearest refinery name, distance, and a full waypoint route.
    """
    if req.region and req.region not in REGION_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown region '{req.region}'. "
                   f"Valid options: {list(REGION_FILES.keys())}",
        )
    try:
        result = transport.find_route(req.lat, req.lon, region=req.region)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.post("/route/nearest", response_model=NearestRefinery, tags=["Transport"])
async def nearest_refinery(req: NearestRequest):
    """
    Lightweight endpoint — returns only the nearest refinery (no route).
    Much faster when you only need distance + name.
    """
    if req.region and req.region not in REGION_FILES:
        raise HTTPException(status_code=422, detail=f"Unknown region '{req.region}'")
    try:
        return transport.find_nearest_refinery(req.lat, req.lon, region=req.region)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/route/regions", tags=["Transport"])
async def list_regions():
    """List all available regions and their GeoJSON filenames."""
    return {
        "regions":        list(REGION_FILES.keys()),
        "loaded":         transport.loaded_regions(),
        "file_map":       REGION_FILES,
    }


@app.get("/route/health", tags=["Transport"])
async def health():
    """Model cache status."""
    return {
        "status":         "ok",
        "loaded_regions": transport.loaded_regions(),
        "cached_count":   len(transport.loaded_regions()),
    }


@app.delete("/route/cache/{region}", tags=["Transport"])
async def unload_region(region: str):
    """Free memory for a region. It will be re-loaded on next request."""
    if region not in REGION_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
    transport.unload_region(region)
    return {"message": f"Region '{region}' unloaded from cache."}
