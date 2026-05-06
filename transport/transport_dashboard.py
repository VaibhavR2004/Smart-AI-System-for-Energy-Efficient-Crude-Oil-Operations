"""
transport_dashboard.py
======================
Interactive HTML map dashboard for the Oil & Gas Transport Model.

Usage:
    python transport_dashboard.py [lat] [lon] [region]

Opens an HTML file in your browser showing:
  - Your query location (blue marker)
  - Nearest refinery (red marker)
  - Pipeline route polyline
  - Distance annotation

Dependencies:
    pip install folium networkx scipy numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

import folium
from folium.plugins import AntPath, MeasureControl, MiniMap

# Ensure transport_model.py is importable
sys.path.insert(0, str(Path(__file__).parent))
from transport_model import transport, haversine_km


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS = {
    "query":    "#00D4FF",
    "refinery": "#FF4D4D",
    "waypoint": "#FFA500",
    "route":    "#FF6B00",
    "bg":       "#0d1117",
}


# ---------------------------------------------------------------------------
# Build Folium map
# ---------------------------------------------------------------------------

def build_map(
    src_lat: float,
    src_lon: float,
    region: str | None = None,
) -> folium.Map:

    result = transport.find_route(src_lat, src_lon, region=region)
    ref    = result["nearest_refinery"]
    route  = result["route"]
    waypoints = [(wp["lat"], wp["lon"]) for wp in route]

    # Centre map between query and refinery
    center_lat = (src_lat + ref["lat"]) / 2
    center_lon = (src_lon + ref["lon"]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Query location marker ──────────────────────────────────────────
    folium.Marker(
        location=[src_lat, src_lon],
        popup=folium.Popup(
            f"<b>📍 Query Location</b><br>Lat: {src_lat:.4f}<br>Lon: {src_lon:.4f}",
            max_width=220,
        ),
        tooltip="Your Location",
        icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
    ).add_to(m)

    folium.CircleMarker(
        location=[src_lat, src_lon],
        radius=14,
        color=COLORS["query"],
        fill=True,
        fill_color=COLORS["query"],
        fill_opacity=0.25,
        weight=2,
    ).add_to(m)

    # ── Nearest refinery marker ────────────────────────────────────────
    folium.Marker(
        location=[ref["lat"], ref["lon"]],
        popup=folium.Popup(
            f"<b>🏭 {ref['name']}</b><br>"
            f"Region: {ref['region'].replace('_',' ').title()}<br>"
            f"Direct: <b>{ref['dist_km']} km</b><br>"
            f"Route:  <b>{result['total_graph_km']} km</b>",
            max_width=260,
        ),
        tooltip=f"🏭 {ref['name']} ({ref['dist_km']} km)",
        icon=folium.Icon(color="red", icon="industry", prefix="fa"),
    ).add_to(m)

    folium.CircleMarker(
        location=[ref["lat"], ref["lon"]],
        radius=14,
        color=COLORS["refinery"],
        fill=True,
        fill_color=COLORS["refinery"],
        fill_opacity=0.25,
        weight=2,
    ).add_to(m)

    # ── Route polyline ─────────────────────────────────────────────────
    if len(waypoints) >= 2:
        # Animated dashed route
        AntPath(
            locations=waypoints,
            color=COLORS["route"],
            weight=4,
            opacity=0.9,
            delay=800,
            dash_array=[15, 25],
            pulse_color="#FFF",
            tooltip=f"Pipeline route — {result['total_graph_km']} km ({result['hops']} hops)",
        ).add_to(m)

        # Solid thinner line underneath for clarity
        folium.PolyLine(
            locations=waypoints,
            color=COLORS["route"],
            weight=2,
            opacity=0.4,
        ).add_to(m)

        # Intermediate waypoint dots (skip first and last)
        for wp in route[1:-1]:
            folium.CircleMarker(
                location=[wp["lat"], wp["lon"]],
                radius=4,
                color=COLORS["waypoint"],
                fill=True,
                fill_color=COLORS["waypoint"],
                fill_opacity=0.8,
                weight=1,
                tooltip=wp["name"],
            ).add_to(m)

    # ── Direct line (straight-line reference) ──────────────────────────
    folium.PolyLine(
        locations=[[src_lat, src_lon], [ref["lat"], ref["lon"]]],
        color="#888888",
        weight=1.5,
        opacity=0.5,
        dash_array="8 6",
        tooltip=f"Direct: {result['direct_km']} km",
    ).add_to(m)

    # ── Info box (top-right) ───────────────────────────────────────────
    info_html = f"""
    <div style="
        position: fixed; top: 12px; right: 12px; z-index: 9999;
        background: rgba(13,17,23,0.92);
        border: 1px solid #FF6B00;
        border-radius: 10px;
        padding: 14px 18px;
        font-family: 'Courier New', monospace;
        color: #e0e0e0;
        min-width: 240px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.6);
    ">
      <div style="font-size:13px; color:#FF6B00; font-weight:bold; margin-bottom:8px;">
        ⛽ OIL &amp; GAS — TRANSPORT MODEL
      </div>
      <table style="font-size:12px; border-collapse:collapse; width:100%">
        <tr><td style="color:#888; padding:2px 0">Region</td>
            <td style="text-align:right">{ref['region'].replace('_',' ').title()}</td></tr>
        <tr><td style="color:#888; padding:2px 0">Refinery</td>
            <td style="text-align:right; max-width:140px; overflow:hidden; text-overflow:ellipsis">{ref['name'][:20]}</td></tr>
        <tr><td style="color:#888; padding:2px 0">Direct dist.</td>
            <td style="text-align:right; color:#00D4FF">{result['direct_km']} km</td></tr>
        <tr><td style="color:#888; padding:2px 0">Route dist.</td>
            <td style="text-align:right; color:#FF6B00">{result['total_graph_km']} km</td></tr>
        <tr><td style="color:#888; padding:2px 0">Route hops</td>
            <td style="text-align:right">{result['hops']}</td></tr>
      </table>
      <div style="margin-top:10px; font-size:10px; color:#555">
        🔵 Query &nbsp; 🔴 Refinery &nbsp; 🟠 Route
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    # ── Controls ───────────────────────────────────────────────────────
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    MiniMap(position="bottomright", tile_layer="CartoDB dark_matter", zoom_level_offset=-6).add_to(m)
    folium.LayerControl().add_to(m)

    return m, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Oil & Gas Transport Model Dashboard")
    parser.add_argument("lat",    type=float, nargs="?", default=29.76,  help="Query latitude  (default: Houston TX)")
    parser.add_argument("lon",    type=float, nargs="?", default=-95.37, help="Query longitude (default: Houston TX)")
    parser.add_argument("region", type=str,   nargs="?", default=None,   help="Region: canada|mexico|usa|saudi_arabia (auto if omitted)")
    args = parser.parse_args()

    print(f"\n🔍 Query: ({args.lat}, {args.lon})  region={args.region or 'auto'}\n")

    m, result = build_map(args.lat, args.lon, region=args.region)

    ref = result["nearest_refinery"]
    print(f"  ✅ Nearest Refinery : {ref['name']}")
    print(f"     Location         : ({ref['lat']:.4f}, {ref['lon']:.4f})")
    print(f"     Direct distance  : {result['direct_km']} km")
    print(f"     Route distance   : {result['total_graph_km']} km")
    print(f"     Hops             : {result['hops']}\n")

    out = Path("transport_route_map.html")
    m.save(str(out))
    print(f"  🗺  Map saved → {out.resolve()}")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
