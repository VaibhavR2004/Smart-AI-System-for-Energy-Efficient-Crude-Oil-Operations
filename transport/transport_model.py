"""
Oil & Gas AI — Transport Model (Pillar 3)
=========================================
Optimized transport model that:
  - Lazily loads GeoJSON refinery clusters by region
  - Uses KDTree for O(log n) nearest-refinery lookup
  - Builds a NetworkX graph only for the loaded region
  - Finds nearest refinery and shortest pipeline path via Dijkstra
  - Returns distance + route for map display

Folder layout expected:
  data/
    clusters/
      CANADA_wells.geojson
      maxico_wells.geojson        (Mexico)
      SAUDI ARABIA_well.geojson
      usa_wells.geojson

GeoJSON feature schema (each Feature):
  geometry.type        : "Point"
  geometry.coordinates : [lon, lat]
  properties.name      : refinery / well name  (optional)
  properties.type      : "refinery" | "well"   (optional)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "clusters"

REGION_FILES: dict[str, str] = {
    "canada":       "CANADA_wells.geojson",
    "mexico":       "maxico_wells.geojson",
    "saudi_arabia": "SAUDI ARABIA_well.geojson",
    "usa":          "usa_wells.geojson",
}

EARTH_RADIUS_KM = 6_371.0

# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two (lat, lon) points."""
    r = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# GeoJSON loader  — loads one region file, keeps only refineries
# ---------------------------------------------------------------------------

def _load_geojson(path: Path) -> list[dict]:
    """
    Parse a GeoJSON FeatureCollection.
    Returns list of dicts: {name, lat, lon, type}.
    Filters to Point features only (skips LineStrings / MultiLineStrings).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    features = data.get("features", [])
    points: list[dict] = []

    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        points.append({
            "name": props.get("name") or props.get("NAME") or "Unknown",
            "lat":  lat,
            "lon":  lon,
            "type": (props.get("type") or props.get("TYPE") or "refinery").lower(),
        })

    return points


# ---------------------------------------------------------------------------
# RegionTransportModel  — one per loaded region
# ---------------------------------------------------------------------------

class RegionTransportModel:
    """
    Holds:
      - list of refinery/well points for one region
      - KDTree built on (lat, lon) in radians for fast lookup
      - Lightweight NetworkX graph connecting nearby facilities
    """

    def __init__(self, name: str, points: list[dict], max_edge_km: float = 500.0):
        self.region_name = name
        self.points = points              # all loaded points
        self._build_kdtree()
        self._build_graph(max_edge_km)

    # ------------------------------------------------------------------
    # KDTree  (built on radian coords — haversine-compatible)
    # ------------------------------------------------------------------

    def _build_kdtree(self) -> None:
        coords_rad = np.radians(
            [(p["lat"], p["lon"]) for p in self.points]
        )
        self._kdtree = KDTree(coords_rad)
        self._coords_rad = coords_rad

    # ------------------------------------------------------------------
    # Lightweight graph  (connect each node to its k nearest neighbours)
    # ------------------------------------------------------------------

    def _build_graph(self, max_edge_km: float, k: int = 8) -> None:
        n = len(self.points)
        G = nx.Graph()

        for i, p in enumerate(self.points):
            G.add_node(i, **p)

        if n < 2:
            self.graph = G
            return

        actual_k = min(k + 1, n)  # +1 because first hit is self
        dists_rad, idxs = self._kdtree.query(self._coords_rad, k=actual_k)

        for i in range(n):
            p1 = self.points[i]
            for j_pos in range(1, actual_k):   # skip self (j_pos=0)
                j = idxs[i][j_pos]
                p2 = self.points[j]
                dist_km = haversine_km(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
                if dist_km <= max_edge_km and not G.has_edge(i, j):
                    G.add_edge(i, j, weight=dist_km)

        self.graph = G

    # ------------------------------------------------------------------
    # Nearest refinery lookup
    # ------------------------------------------------------------------

    def nearest_refinery(
        self,
        lat: float,
        lon: float,
        n_candidates: int = 5,
    ) -> dict:
        """
        Return the nearest refinery (type == 'refinery') to (lat, lon).
        Falls back to any nearest point if no explicit refineries found.
        """
        query_rad = np.radians([lat, lon])
        actual_k = min(n_candidates * 3, len(self.points))
        _, idxs = self._kdtree.query(query_rad, k=actual_k)

        if np.isscalar(idxs):
            idxs = [int(idxs)]
        else:
            idxs = idxs.tolist()

        # Prefer labelled refineries
        refinery_candidates = [i for i in idxs if self.points[i]["type"] == "refinery"]
        candidates = refinery_candidates if refinery_candidates else idxs

        best_idx = min(
            candidates,
            key=lambda i: haversine_km(lat, lon, self.points[i]["lat"], self.points[i]["lon"]),
        )
        best = self.points[best_idx]
        dist_km = haversine_km(lat, lon, best["lat"], best["lon"])

        return {
            "index":    best_idx,
            "name":     best["name"],
            "lat":      best["lat"],
            "lon":      best["lon"],
            "dist_km":  round(dist_km, 2),
            "region":   self.region_name,
        }

    # ------------------------------------------------------------------
    # Graph routing  — Dijkstra from query point to nearest refinery
    # ------------------------------------------------------------------

    def find_route(
        self,
        src_lat: float,
        src_lon: float,
        dst_idx: Optional[int] = None,
    ) -> dict:
        """
        Find shortest graph path from (src_lat, src_lon) to the nearest
        refinery node.  Returns route waypoints + total graph distance.
        """
        if not self.graph.nodes:
            return {"error": "Graph is empty — no points loaded."}

        # Snap source to nearest node
        query_rad = np.radians([src_lat, src_lon])
        _, src_idx_arr = self._kdtree.query(query_rad, k=1)
        src_idx = int(src_idx_arr) if not np.isscalar(src_idx_arr) else int(src_idx_arr)

        # Determine destination
        if dst_idx is None:
            result = self.nearest_refinery(src_lat, src_lon)
            dst_idx = result["index"]

        if src_idx == dst_idx:
            p = self.points[src_idx]
            return {
                "route": [{"lat": p["lat"], "lon": p["lon"], "name": p["name"]}],
                "total_graph_km": 0.0,
                "direct_km": 0.0,
                "hops": 0,
            }

        # Check connectivity and attempt Dijkstra
        try:
            if nx.has_path(self.graph, src_idx, dst_idx):
                path_nodes = nx.dijkstra_path(self.graph, src_idx, dst_idx, weight="weight")
                total_km = nx.dijkstra_path_length(self.graph, src_idx, dst_idx, weight="weight")
            else:
                # Nodes not connected — fallback: direct line
                path_nodes = [src_idx, dst_idx]
                total_km = haversine_km(
                    self.points[src_idx]["lat"], self.points[src_idx]["lon"],
                    self.points[dst_idx]["lat"], self.points[dst_idx]["lon"],
                )
        except nx.NetworkXNoPath:
            path_nodes = [src_idx, dst_idx]
            total_km = haversine_km(
                self.points[src_idx]["lat"], self.points[src_idx]["lon"],
                self.points[dst_idx]["lat"], self.points[dst_idx]["lon"],
            )

        route = [
            {
                "lat":  self.points[n]["lat"],
                "lon":  self.points[n]["lon"],
                "name": self.points[n]["name"],
            }
            for n in path_nodes
        ]

        direct_km = haversine_km(src_lat, src_lon, self.points[dst_idx]["lat"], self.points[dst_idx]["lon"])

        return {
            "route":            route,
            "total_graph_km":   round(total_km, 2),
            "direct_km":        round(direct_km, 2),
            "hops":             len(path_nodes) - 1,
        }


# ---------------------------------------------------------------------------
# TransportModelManager  — lazy-loads regions on demand, caches them
# ---------------------------------------------------------------------------

class TransportModelManager:
    """
    Master manager.  Only loads a region's GeoJSON into memory when first
    requested.  Subsequent queries for the same region reuse the cached model.
    Memory stays low because unused regions are never loaded.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self._cache: dict[str, RegionTransportModel] = {}

    # ------------------------------------------------------------------
    # Internal: resolve which region file to load
    # ------------------------------------------------------------------

    def _resolve_region(self, lat: float, lon: float) -> str:
        """Geo-fence heuristic to auto-detect region from coordinates."""
        if 5.0 <= lat <= 83.0 and -142.0 <= lon <= -52.0:
            # Rough North America bounding box
            if lat >= 49.0:
                return "canada"
            elif lat <= 32.0 and lon <= -86.0:
                return "mexico"
            else:
                return "usa"
        elif 12.0 <= lat <= 38.0 and 34.0 <= lon <= 60.0:
            return "saudi_arabia"
        # fallback — load all regions and pick globally nearest
        return "all"

    def _load_region(self, region: str) -> RegionTransportModel:
        if region in self._cache:
            return self._cache[region]

        path = self.data_dir / REGION_FILES[region]
        if not path.exists():
            raise FileNotFoundError(f"GeoJSON not found: {path}")

        print(f"[TransportModel] Loading region '{region}' from {path.name} ...")
        points = _load_geojson(path)
        if not points:
            raise ValueError(f"No Point features found in {path.name}")

        model = RegionTransportModel(region, points)
        self._cache[region] = model
        print(f"[TransportModel] '{region}' loaded — {len(points)} points, "
              f"{model.graph.number_of_nodes()} nodes, "
              f"{model.graph.number_of_edges()} edges")
        return model

    def _load_all(self) -> list[RegionTransportModel]:
        models = []
        for region in REGION_FILES:
            path = self.data_dir / REGION_FILES[region]
            if path.exists():
                models.append(self._load_region(region))
        return models

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_nearest_refinery(
        self,
        lat: float,
        lon: float,
        region: Optional[str] = None,
    ) -> dict:
        """
        Given a (lat, lon) location, find the nearest refinery.

        Parameters
        ----------
        lat, lon : float  — query location
        region   : str    — one of 'canada','mexico','usa','saudi_arabia', or
                            None (auto-detect from coordinates)

        Returns
        -------
        dict with keys: name, lat, lon, dist_km, region
        """
        if region is None:
            region = self._resolve_region(lat, lon)

        if region == "all":
            models = self._load_all()
            results = [m.nearest_refinery(lat, lon) for m in models]
            return min(results, key=lambda r: r["dist_km"])

        model = self._load_region(region)
        return model.nearest_refinery(lat, lon)

    def find_route(
        self,
        src_lat: float,
        src_lon: float,
        region: Optional[str] = None,
    ) -> dict:
        """
        Find pipeline route from (src_lat, src_lon) to nearest refinery.

        Returns
        -------
        dict with keys:
          nearest_refinery : dict (name, lat, lon, dist_km)
          route            : list of {lat, lon, name} waypoints
          total_graph_km   : float
          direct_km        : float
          hops             : int
        """
        if region is None:
            region = self._resolve_region(src_lat, src_lon)

        if region == "all":
            models = self._load_all()
            best_ref = min(
                [m.nearest_refinery(src_lat, src_lon) for m in models],
                key=lambda r: r["dist_km"],
            )
            region = best_ref["region"]

        model = self._load_region(region)
        nearest = model.nearest_refinery(src_lat, src_lon)
        routing = model.find_route(src_lat, src_lon, dst_idx=nearest["index"])

        return {
            "query": {"lat": src_lat, "lon": src_lon},
            "nearest_refinery": nearest,
            **routing,
        }

    def unload_region(self, region: str) -> None:
        """Free memory for a region no longer needed."""
        self._cache.pop(region, None)
        print(f"[TransportModel] Region '{region}' unloaded from cache.")

    def loaded_regions(self) -> list[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Module-level singleton  (import and use directly)
# ---------------------------------------------------------------------------

transport = TransportModelManager()


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
        region = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        # Default demo: Houston TX area
        lat, lon, region = 29.76, -95.37, "usa"

    print(f"\n{'='*60}")
    print(f"  Query Location : ({lat}, {lon})")
    print(f"  Region         : {region or 'auto-detect'}")
    print(f"{'='*60}\n")

    result = transport.find_route(lat, lon, region=region)

    ref = result["nearest_refinery"]
    print(f"  Nearest Refinery : {ref['name']}")
    print(f"  Location         : ({ref['lat']:.4f}, {ref['lon']:.4f})")
    print(f"  Direct distance  : {result['direct_km']} km")
    print(f"  Graph route dist : {result['total_graph_km']} km")
    print(f"  Route hops       : {result['hops']}")
    print(f"\n  Waypoints:")
    for wp in result["route"][:10]:
        print(f"    → {wp['name']}  ({wp['lat']:.4f}, {wp['lon']:.4f})")
    if len(result["route"]) > 10:
        print(f"    ... ({len(result['route'])-10} more waypoints)")
