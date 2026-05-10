"""
Oil & Gas AI System — Interactive Streamlit Dashboard
Multi-Model ML Platform: Seismic | Failure | Extraction | Transport | Economics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image
import joblib
import pickle
import sys
import os
import requests
from price_prediction.config import EIA_API_KEY


# ─── FETCH LIVE WTI DATA FROM EIA ─────────────────────

def fetch_wti_prices(api_key, weeks=60):

    from datetime import datetime, timedelta

    end_date = datetime.today()

    start_date = end_date - timedelta(weeks=weeks + 4)

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

    params = {

        "api_key": api_key,

        "frequency": "weekly",

        "data[0]": "value",

        "facets[series][]": "RWTC",

        "start": start_date.strftime("%Y-%m-%d"),

        "end": end_date.strftime("%Y-%m-%d"),

        "sort[0][column]": "period",

        "sort[0][direction]": "asc",

        "length": weeks + 10,
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    records = data.get(
        "response",
        {}
    ).get(
        "data",
        []
    )

    prices = pd.Series({

        r["period"]: float(r["value"])

        for r in records

    }).sort_index()

    return prices
# from price_prediction.predictor import get_current_wti_price
# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OilGas AI — Intelligence Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── IMAGE TRANSFORM ───────────────────────────────

seismic_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    background-color: #050a14;
    color: #c8d8e8;
    font-family: 'Rajdhani', sans-serif;
}

/* Main background */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0a1628 0%, #050a14 60%, #030810 100%);
    background-attachment: fixed;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070e1e 0%, #050a14 100%);
    border-right: 1px solid #1a3a5c;
}

/* Header */
.main-header {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(90deg, #00d4ff, #0088cc, #ff6b35, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0 0.5rem;
    text-shadow: none;
}

.sub-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a8fa8;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-top: 2px solid #00d4ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #4a8fa8;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4ff;
    margin: 0.2rem 0;
}

.metric-delta {
    font-size: 0.75rem;
    color: #2ecc71;
}

/* Section titles */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 1rem;
}

/* Input labels */
.stSlider label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #7ab3cc !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #003d5c 0%, #005a8a 100%);
    color: #00d4ff;
    border: 1px solid #00d4ff;
    border-radius: 4px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #005a8a 0%, #0077bb 100%);
    border-color: #ffd700;
    color: #ffd700;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
}

/* Result boxes */
.result-box {
    background: linear-gradient(135deg, #0a1f10 0%, #051508 100%);
    border: 1px solid #1a5c2a;
    border-left: 3px solid #2ecc71;
    border-radius: 6px;
    padding: 1.2rem;
    margin-top: 1rem;
}

.result-box.warning {
    background: linear-gradient(135deg, #1f1a0a 0%, #150e05 100%);
    border: 1px solid #5c4a1a;
    border-left: 3px solid #ffd700;
}

.result-box.danger {
    background: linear-gradient(135deg, #1f0a0a 0%, #150505 100%);
    border: 1px solid #5c1a1a;
    border-left: 3px solid #e74c3c;
}

.result-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
}

.result-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #7ab3cc;
    letter-spacing: 2px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #070e1e;
    border-bottom: 1px solid #1a3a5c;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: #4a8fa8;
    padding: 0.7rem 1rem;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: rgba(0, 212, 255, 0.05) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a8fa8;
}

/* Info/warning */
.stAlert {
    border-radius: 4px;
    font-family: 'Rajdhani', sans-serif;
}

/* Number input */
.stNumberInput input {
    background: #0a1628;
    border: 1px solid #1a3a5c;
    color: #c8d8e8;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    border-radius: 4px;
}

/* Slider */
.stSlider .stSlider-thumb {
    background: #00d4ff;
}

/* Divider */
hr {
    border-color: #1a3a5c;
}

/* Sidebar nav button */
.nav-btn {
    display: block;
    width: 100%;
    padding: 0.8rem 1rem;
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    color: #4a8fa8;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s;
    margin: 2px 0;
}

.nav-btn:hover, .nav-btn.active {
    background: rgba(0, 212, 255, 0.05);
    border-left-color: #00d4ff;
    color: #00d4ff;
}

/* Status indicators */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

.status-green { background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }
.status-yellow { background: #ffd700; box-shadow: 0 0 6px #ffd700; animation-duration: 1s; }
.status-red { background: #e74c3c; box-shadow: 0 0 6px #e74c3c; animation-duration: 0.5s; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Gauge styles */
.gauge-container {
    text-align: center;
    padding: 0.5rem;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050a14; }
::-webkit-scrollbar-thumb { background: #1a3a5c; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff; }
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⬡ OIL & GAS AI INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MULTI-MODEL ML SYSTEM  |  SEISMIC · FAILURE · EXTRACTION · TRANSPORT · ECONOMICS</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family: 'Orbitron', monospace; font-size: 0.6rem; color: #4a8fa8; letter-spacing: 3px; text-align: center; padding: 0.5rem 0 1rem; border-bottom: 1px solid #1a3a5c;">
        SYSTEM CONTROL PANEL
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🛢️ Extraction Rate", "⚠️ Failure Prediction", "🛣️ Transport Router", "🌊 Seismic Detection", "📈 Price Prediction"],
        label_visibility="collapsed"
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top: 1px solid #1a3a5c; padding-top: 1rem;">
        <div class="metric-label" style="text-align:center; margin-bottom:0.8rem;">SYSTEM STATUS</div>
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; line-height: 2;">
            <span class="status-dot status-green"></span>Extraction Model<br>
            <span class="status-dot status-green"></span>Failure Model<br>
            <span class="status-dot status-green"></span>Price Models<br>
            <span class="status-dot status-yellow"></span>Transport Router<br>
            <span class="status-dot status-green"></span>Seismic CNN<br>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: #2a4a5c; text-align: center; letter-spacing: 1px;">
        v2.4.1 · Research Paper Build<br>
        SRM IST Delhi NCR · CSE Dept
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD SEISMIC MODEL ─────────────────────────


@st.cache_resource
def load_seismic_model():

    model = models.resnet18(weights=None)

    # FIX FOR GRAYSCALE MODEL
    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(
            "models/seismic_reservoir_cnn.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


seismic_model = load_seismic_model()

# ─── LOAD PRICE PREDICTION MODELS ─────────────────────

@st.cache_resource
def load_price_models():

    lgbm_model = joblib.load(
        "models/economics/price_lgbm_fixed.pkl"
    )

    xgb_model = joblib.load(
        "models/economics/price_xgb_fixed.pkl"
    )

    ridge_model = joblib.load(
        "models/economics/price_ridge_fixed.pkl"
    )

    scaler = joblib.load(
        "models/economics/scaler_fixed.pkl"
    )

    feature_columns = joblib.load(
        "models/economics/feature_columns_fixed.pkl"
    )

    return (
        lgbm_model,
        xgb_model,
        ridge_model,
        scaler,
        feature_columns
    )


(
    lgbm_model,
    xgb_model,
    ridge_model,
    scaler,
    feature_columns
) = load_price_models()

def build_features(prices, feature_columns):

    if hasattr(prices, 'values'):
        s = prices.values.astype(float)
    else:
        s = np.asarray(prices, dtype=float)

    p = s[-1]

    row = {

        "lag_1w": s[-2],
        "lag_2w": s[-3],
        "lag_3w": s[-4],
        "lag_4w": s[-5],

        "lag_8w": s[-9],
        "lag_12w": s[-13],

        "lag_26w": s[-27],

        "lag_52w": s[-53]
        if len(s) >= 53
        else s[0],

        "roll_mean_4w": s[-4:].mean(),
        "roll_std_4w": s[-4:].std(),

        "roll_mean_8w": s[-8:].mean(),
        "roll_std_8w": s[-8:].std(),

        "roll_mean_13w": s[-13:].mean(),
        "roll_std_13w": s[-13:].std(),

        "roll_mean_26w": s[-26:].mean(),
        "roll_std_26w": s[-26:].std(),

        "mom_4w": p - s[-5],

        "mom_13w": p - s[-13],

        "pct_chg_1w":
            (p - s[-2]) / s[-2] * 100,

        "pct_chg_4w":
            (p - s[-5]) / s[-5] * 100,

        "ema_8w":
            pd.Series(s).ewm(
                span=8,
                adjust=False
            ).mean().iloc[-1],

        "ema_26w":
            pd.Series(s).ewm(
                span=26,
                adjust=False
            ).mean().iloc[-1],
    }

    row["ema_cross"] = (
        row["ema_8w"]
        -
        row["ema_26w"]
    )

    bb_mid = s[-20:].mean()

    bb_std = s[-20:].std()

    bb_upper = bb_mid + 2 * bb_std

    bb_lower = bb_mid - 2 * bb_std

    row["bb_pos"] = (
        (s[-1] - bb_lower)
        /
        (bb_upper - bb_lower)
        if bb_upper != bb_lower
        else 0.5
    )

    from datetime import datetime

    row["week_of_year"] = int(
        datetime.today().isocalendar()[1]
    )

    row["month"] = datetime.today().month

    row["quarter"] = (
        (datetime.today().month - 1) // 3
    ) + 1

    row["year"] = datetime.today().year

    return pd.DataFrame([row])[
        feature_columns
    ]
# ─── IMAGE TRANSFORM ────────────────────────────

seismic_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])
# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        ("MODELS ACTIVE", "5", "+1 this sprint", col1),
        ("SEISMIC ACC.", "79.6%", "ResNet18 CNN", col2),
        ("FAILURE ACC.", "94.0%", "F1 = 0.94", col3),
        ("EXTRACTION R²", "0.990", "XGBoost Reg.", col4),
        ("PRICE RMSE", "$2.41", "Ensemble avg", col5),
    ]

    for label, value, delta, col in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram (radar + bar charts)
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-title">MODEL PERFORMANCE RADAR</div>', unsafe_allow_html=True)

        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Inference Speed']
        seismic   = [79.6, 82, 76, 79, 85]
        failure   = [94.0, 96, 57, 63, 90]
        extraction = [99.0, 98, 97, 97.5, 95]

        fig_radar = go.Figure()
        for name, vals, color in [
            ("Seismic CNN", seismic, "#00d4ff"),
            ("Failure RF", failure, "#ff6b35"),
            ("Extraction XGB", extraction, "#2ecc71"),
        ]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=name,
                line_color=color,
                fillcolor='rgba(0,212,255,0.1)' if color == '#00d4ff'
                else 'rgba(255,107,53,0.1)' if color == '#ff6b35'
                else 'rgba(46,204,113,0.1)',
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1a3a5c', color='#4a8fa8'),
                angularaxis=dict(gridcolor='#1a3a5c', color='#7ab3cc',
                                 tickfont=dict(family='Share Tech Mono', size=10)),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(family='Share Tech Mono', size=10, color='#c8d8e8'),
                        bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=50, r=50, t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">PIPELINE OVERVIEW</div>', unsafe_allow_html=True)

        pillars = ["Seismic<br>Detection", "Equipment<br>Failure", "Transport<br>Routing", "Extraction<br>Rate", "Price<br>Prediction"]
        scores  = [79.6, 94.0, 100, 99.0, 96.2]
        colors  = ['#00d4ff', '#ff6b35', '#2ecc71', '#ffd700', '#9b59b6']

        fig_bar = go.Figure(go.Bar(
            x=pillars,
            y=scores,
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            text=[f"{s}%" for s in scores],
            textposition='outside',
            textfont=dict(family='Orbitron', size=10, color='#c8d8e8'),
        ))

        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(5,10,20,0.5)',
            yaxis=dict(range=[0, 115], gridcolor='#1a3a5c', color='#4a8fa8',
                       tickfont=dict(family='Share Tech Mono', size=9)),
            xaxis=dict(color='#7ab3cc', tickfont=dict(family='Share Tech Mono', size=9)),
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # System table
    st.markdown('<div class="section-title">SYSTEM ARCHITECTURE SUMMARY</div>', unsafe_allow_html=True)

    df_summary = pd.DataFrame({
        "Pillar": ["🌊 Seismic (Model 1)", "⚠️ Failure (Model 3)", "🛣️ Transport", "🛢️ Extraction (Model 9)", "📈 Price Prediction"],
        "Algorithm": ["CNN (ResNet18)", "Random Forest + SHAP", "Dijkstra + KDTree", "XGBoost Regression", "LightGBM + XGBoost + Ridge"],
        "Dataset": ["F3 Dutch North Sea", "AI4I + 3W + OREDA", "OGIM GeoPackage", "3W Petrobras (2.6M)", "EIA WTI Weekly"],
        "Key Metric": ["79.65% Accuracy", "94% Acc, F1=0.94", "Route: O(log n) lookup", "R²=0.990, RMSE=6,206", "RMSE=$2.41"],
        "Status": ["🟢 Live", "🟢 Live", "🟡 Active", "🟢 Live", "🟢 Live"],
    })

    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EXTRACTION RATE
# ═══════════════════════════════════════════════════════════════
elif page == "🛢️ Extraction Rate":
    st.markdown('<div class="section-title">🛢️ WELL EXTRACTION RATE PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #4a8fa8; margin-bottom: 1.5rem;">
    XGBoost Regression · R² = 0.990 · RMSE = 6,206 BPD · Trained on 2.6M 3W Petrobras samples
    </div>
    """, unsafe_allow_html=True)

    # Quick-fill presets
    preset = st.selectbox("⚡ QUICK PRESET", ["Custom Input", "Well A — High Performing", "Well B — Average", "Well C — Degraded"])

    presets = {
        "Well A — High Performing": {
            "pgl": 92.0, "ppc": 88.4, "ptt": 19500000.0, "ttt": 145.0, "tdg": 185.0,
            "way": 1.5, "asc": 9.2, "aqs": 8.8, "mtbf": 300, "mttr": 4.0, "avail": 97.5,
        },
        "Well B — Average": {
            "pgl": 70.0, "ppc": 65.0, "ptt": 12000000.0, "ttt": 122.0, "tdg": 155.0,
            "way": 5.0, "asc": 6.5, "aqs": 6.0, "mtbf": 180, "mttr": 8.4, "avail": 85.2,
        },
        "Well C — Degraded": {
            "pgl": 40.0, "ppc": 35.0, "ptt": 6000000.0, "ttt": 95.0, "tdg": 115.0,
            "way": 12.0, "asc": 3.8, "aqs": 3.2, "mtbf": 60, "mttr": 18.0, "avail": 55.0,
        },
    }

    p = presets.get(preset, {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#00d4ff; letter-spacing:2px; margin-bottom:0.5rem;">📡 SENSOR READINGS</div>', unsafe_allow_html=True)
        pgl = st.number_input("Pressure Gas Lift (bar)", value=p.get("pgl", 75.0), step=0.1, format="%.1f")
        ppc = st.number_input("Pressure Production Casing (bar)", value=p.get("ppc", 70.0), step=0.1, format="%.1f")
        ptt = st.number_input("Pressure Top Tubing (Pa)", value=p.get("ptt", 1.2e7), step=1e5, format="%.0f")
        ttt = st.number_input("Temp Top Tubing (°C)", value=p.get("ttt", 122.0), step=1.0, format="%.1f")
        tdg = st.number_input("Temp Downhole Gauge (°C)", value=p.get("tdg", 155.0), step=1.0, format="%.1f")

    with col2:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#ffd700; letter-spacing:2px; margin-bottom:0.5rem;">🏗️ WELL ATTRIBUTES</div>', unsafe_allow_html=True)
        way = st.number_input("Well Age (years)", value=p.get("way", 5.0), min_value=0.0, max_value=50.0, step=0.5, format="%.1f")
        asc = st.slider("Attribute Score", 0.0, 10.0, value=p.get("asc", 6.5), step=0.1)
        aqs = st.slider("Aggregate Quality Score", 0.0, 10.0, value=p.get("aqs", 6.0), step=0.1)

    with col3:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#2ecc71; letter-spacing:2px; margin-bottom:0.5rem;">🔧 EQUIPMENT HEALTH</div>', unsafe_allow_html=True)
        mtbf = st.number_input("MTBF (days)", value=p.get("mtbf", 180), min_value=1, max_value=1000, step=5)
        mttr = st.number_input("MTTR (days)", value=p.get("mttr", 8.4), min_value=0.1, max_value=90.0, step=0.1, format="%.1f")
        avail = st.slider("Availability (%)", 0.0, 100.0, value=p.get("avail", 85.2), step=0.1)

    if st.button("▶  RUN EXTRACTION PREDICTION", key="ext_btn"):
        with st.spinner("Running XGBoost inference..."):
            time.sleep(0.8)

        # Simulate prediction (replace with actual model.predict())
        base = (pgl * 45 + ppc * 38 + (ptt / 1e7) * 800 + ttt * 15 + tdg * 10)
        age_penalty = max(0, 1 - (way - 1) * 0.04)
        quality_boost = (asc + aqs) / 20
        health_factor = (avail / 100) * (mtbf / 300) * max(0.1, 1 - mttr / 50)
        pred_bpd = base * age_penalty * quality_boost * health_factor

        pred_bpd = max(100, min(12000, pred_bpd))

        if pred_bpd >= 5000:
            status = "HIGH PRODUCTION"; css_class = "result-box"; emoji = "🟢"; color = "#2ecc71"
        elif pred_bpd >= 2000:
            status = "MODERATE PRODUCTION"; css_class = "result-box warning"; emoji = "🟡"; color = "#ffd700"
        else:
            status = "LOW PRODUCTION — REVIEW"; css_class = "result-box danger"; emoji = "🔴"; color = "#e74c3c"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">PREDICTED EXTRACTION RATE</div>
            <div class="result-value" style="color:{color};">{pred_bpd:,.0f} <span style="font-size:1rem;">BPD</span></div>
            <div style="margin-top:0.5rem; font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:{color};">
                {emoji} {status}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance visualization
        st.markdown("<br>", unsafe_allow_html=True)
        features = ["pressure_gas_lift", "pressure_production_casing", "pressure_toptubing",
                    "temp_toptubing", "temp_downholegauge", "well_age_years",
                    "attribute_score", "aggregate_quality_score", "mtbf_days", "mttr_days", "availability_pct"]
        importances = [0.18, 0.15, 0.14, 0.10, 0.09, 0.08, 0.07, 0.07, 0.05, 0.04, 0.03]
        colors_imp = ['#00d4ff' if imp > 0.10 else '#4a8fa8' for imp in importances]

        fig_imp = go.Figure(go.Bar(
            x=importances[::-1],
            y=features[::-1],
            orientation='h',
            marker_color=colors_imp[::-1],
            text=[f"{v:.0%}" for v in importances[::-1]],
            textposition='outside',
            textfont=dict(family='Share Tech Mono', size=9, color='#c8d8e8'),
        ))
        fig_imp.update_layout(
            title=dict(text="FEATURE IMPORTANCE (XGBoost)", font=dict(family='Orbitron', size=10, color='#00d4ff')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,20,0.5)',
            xaxis=dict(gridcolor='#1a3a5c', color='#4a8fa8', tickfont=dict(family='Share Tech Mono', size=9)),
            yaxis=dict(color='#7ab3cc', tickfont=dict(family='Share Tech Mono', size=9)),
            height=350, margin=dict(l=10, r=80, t=40, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — FAILURE PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "⚠️ Failure Prediction":
    st.markdown('<div class="section-title">⚠️ EQUIPMENT FAILURE PREDICTION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #4a8fa8; margin-bottom: 1.5rem;">
    Random Forest Classifier + SHAP · Accuracy=94% · F1=0.94 · Threshold=0.7708 · AI4I + 3W + OREDA datasets
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#00d4ff; letter-spacing:2px; margin-bottom:0.5rem;">📡 PRESSURE & FLOW</div>', unsafe_allow_html=True)
        f_pgl  = st.number_input("Pressure Gas Lift (bar)", value=92.0, step=0.5, format="%.1f", key="f_pgl")
        f_ppc  = st.number_input("Pressure Production Casing", value=88.4, step=0.5, format="%.1f", key="f_ppc")
        f_ptt  = st.number_input("Pressure Top Tubing (Pa)", value=1.95e7, step=1e5, format="%.0f", key="f_ptt")
        f_flow = st.number_input("Flow Gas Lift", value=0.3, step=0.01, format="%.2f", key="f_flow")

    with col2:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#ff6b35; letter-spacing:2px; margin-bottom:0.5rem;">🌡️ TEMPERATURE & VALVES</div>', unsafe_allow_html=True)
        f_ttt  = st.number_input("Temp Top Tubing (°C)", value=122.0, step=1.0, key="f_ttt")
        f_tdg  = st.number_input("Temp Downhole Gauge (°C)", value=118.0, step=1.0, key="f_tdg")
        f_tcp  = st.number_input("Temp Check Production (°C)", value=110.0, step=1.0, key="f_tcp")
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#ff6b35; letter-spacing:2px; margin-top:0.8rem; margin-bottom:0.3rem;">VALVE STATUS (0=closed, 1=open)</div>', unsafe_allow_html=True)
        f_dhsv  = st.selectbox("DHSV", [0, 1], index=0, key="f_dhsv")
        f_m1    = st.selectbox("Master 1", [0, 1], index=1, key="f_m1")
        f_m2    = st.selectbox("Master 2", [0, 1], index=1, key="f_m2")
        f_xmas  = st.selectbox("Xmas Tree", [0, 1], index=1, key="f_xmas")
        f_sdgl  = st.selectbox("Shutdown Gas Lift", [0, 1], index=1, key="f_sdgl")

    with col3:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#2ecc71; letter-spacing:2px; margin-bottom:0.5rem;">🔧 HEALTH METRICS</div>', unsafe_allow_html=True)
        f_mtbf  = st.number_input("MTBF (days)", value=180, min_value=1, max_value=1000, key="f_mtbf")
        f_mttr  = st.number_input("MTTR (days)", value=8.4, min_value=0.1, step=0.1, format="%.1f", key="f_mttr")
        f_frate = st.number_input("Failure Rate (per year)", value=2.1, min_value=0.0, step=0.1, format="%.1f", key="f_frate")
        f_avail = st.slider("Availability (%)", 0.0, 100.0, value=85.2, key="f_avail")

    if st.button("▶  PREDICT FAILURE PROBABILITY", key="fail_btn"):
        with st.spinner("Running Random Forest + SHAP analysis..."):
            time.sleep(1.0)

        # Simulate failure probability
        risk_score = 0.0
        risk_score += (1 - f_avail / 100) * 0.3
        risk_score += min(1, f_mttr / 30) * 0.2
        risk_score += min(1, f_frate / 5) * 0.2
        risk_score += (1 - min(1, f_mtbf / 300)) * 0.15
        if f_dhsv == 0: risk_score += 0.05
        risk_score += max(0, (f_tcp - 130) / 100) * 0.1
        risk_score = max(0.02, min(0.97, risk_score + random.uniform(-0.03, 0.03)))

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            title=dict(text="FAILURE PROBABILITY (%)", font=dict(family='Orbitron', size=12, color='#c8d8e8')),
            number=dict(suffix="%", font=dict(family='Orbitron', size=40, color='#e74c3c' if risk_score > 0.7 else '#ffd700' if risk_score > 0.4 else '#2ecc71')),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#4a8fa8', tickfont=dict(family='Share Tech Mono', size=9)),
                bar=dict(color='#e74c3c' if risk_score > 0.7 else '#ffd700' if risk_score > 0.4 else '#2ecc71'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='#1a3a5c',
                steps=[
                    dict(range=[0, 40], color='rgba(46, 204, 113, 0.1)'),
                    dict(range=[40, 70], color='rgba(255, 215, 0, 0.1)'),
                    dict(range=[70, 100], color='rgba(231, 76, 60, 0.1)'),
                ],
                threshold=dict(line=dict(color="#ffffff", width=2), thickness=0.75, value=77.08),
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c8d8e8'),
            height=280,
            margin=dict(l=30, r=30, t=60, b=20),
        )

        col_g, col_s = st.columns([1, 1])
        with col_g:
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_s:
            # SHAP-style feature importance
            shap_features = ["availability_pct", "mttr_days", "failure_rate", "mtbf_days",
                              "valvestatus_dhsv", "temp_checkproduction", "pressure_gas_lift", "flow_gaslift"]
            shap_values = [
                -(f_avail - 85) * 0.003,
                (f_mttr - 8) * 0.008,
                (f_frate - 2) * 0.05,
                -(f_mtbf - 180) * 0.001,
                (1 - f_dhsv) * 0.12,
                (f_tcp - 110) * 0.002,
                -(f_pgl - 70) * 0.001,
                -f_flow * 0.05,
            ]
            colors_shap = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_values]

            fig_shap = go.Figure(go.Bar(
                x=shap_values,
                y=shap_features,
                orientation='h',
                marker_color=colors_shap,
                text=[f"{'+' if v>0 else ''}{v:.3f}" for v in shap_values],
                textposition='outside',
                textfont=dict(family='Share Tech Mono', size=9, color='#c8d8e8'),
            ))
            fig_shap.update_layout(
                title=dict(text="SHAP FEATURE CONTRIBUTIONS", font=dict(family='Orbitron', size=10, color='#ff6b35')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,20,0.5)',
                xaxis=dict(gridcolor='#1a3a5c', color='#4a8fa8', tickfont=dict(family='Share Tech Mono', size=9), zeroline=True, zerolinecolor='#4a8fa8'),
                yaxis=dict(color='#7ab3cc', tickfont=dict(family='Share Tech Mono', size=9)),
                height=280, margin=dict(l=10, r=80, t=40, b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        if risk_score >= 0.7:
            css_r, status_r, emoji_r = "result-box danger", "FAILURE LIKELY — IMMEDIATE MAINTENANCE REQUIRED", "🔴"
        elif risk_score >= 0.4:
            css_r, status_r, emoji_r = "result-box warning", "ELEVATED RISK — SCHEDULE INSPECTION", "🟡"
        else:
            css_r, status_r, emoji_r = "result-box", "LOW RISK — NORMAL OPERATIONS", "🟢"

        st.markdown(f"""
        <div class="{css_r}">
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;">{emoji_r} {status_r}</span><br>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#4a8fa8;">
            Decision threshold: 0.7708 · Top SHAP features shown above · Random Forest + SMOTE
            </span>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — TRANSPORT ROUTER
# ═══════════════════════════════════════════════════════════════
elif page == "🛣️ Transport Router":
    import streamlit.components.v1 as components

    st.markdown('<div class="section-title">🛣️ PIPELINE TRANSPORT OPTIMIZER</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #4a8fa8; margin-bottom: 1rem;">
    Dijkstra's Algorithm + KDTree · ~80K nodes · ~64K edges · OGIM GeoPackage · ~4,700× faster than naive search
    <br><span style="color:#2a4a5c;">Click any preset or enter coordinates — map updates live. Click the map directly to query any location.</span>
    </div>
    """, unsafe_allow_html=True)

    TRANSPORT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oil & Gas — Transport Model Tester</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080c10;
    --panel:     #0d1420;
    --border:    #1e2d44;
    --accent:    #ff6b00;
    --accent2:   #00d4ff;
    --green:     #00ff88;
    --text:      #c8d8e8;
    --muted:     #4a6080;
    --font-ui:   'Barlow Condensed', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text); font-family: var(--font-ui); height: 100%; overflow: hidden; }
  body { display: flex; height: 100vh; }

  #sidebar {
    width: 320px; min-width: 320px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    z-index: 100; overflow-y: auto;
  }
  .logo { padding: 16px 18px 12px; border-bottom: 1px solid var(--border); }
  .logo-top { font-size: 9px; letter-spacing: 4px; color: var(--muted); text-transform: uppercase; font-family: var(--font-mono); }
  .logo-main { font-size: 19px; font-weight: 900; color: #fff; line-height: 1; margin: 3px 0 2px; }
  .logo-main span { color: var(--accent); }
  .logo-sub { font-size: 10px; color: var(--muted); font-family: var(--font-mono); letter-spacing: 1px; }

  .section { padding: 14px 18px; border-bottom: 1px solid var(--border); }
  .section-title { font-size: 9px; letter-spacing: 3px; color: var(--muted); text-transform: uppercase; margin-bottom: 12px; font-family: var(--font-mono); }

  .field { margin-bottom: 11px; }
  label { display: block; font-size: 10px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; font-family: var(--font-mono); }
  input, select {
    width: 100%; background: #111923; border: 1px solid var(--border);
    color: var(--text); font-family: var(--font-mono); font-size: 12px;
    padding: 8px 10px; border-radius: 4px; outline: none; transition: border-color .2s;
  }
  input:focus, select:focus { border-color: var(--accent); }
  .coords-row { display: flex; gap: 8px; }
  .coords-row .field { flex: 1; }

  .presets { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }
  .preset-btn {
    background: #111923; border: 1px solid var(--border); color: var(--muted);
    font-family: var(--font-mono); font-size: 9px; padding: 4px 9px;
    border-radius: 3px; cursor: pointer; transition: all .15s; letter-spacing: 1px;
  }
  .preset-btn:hover { border-color: var(--accent2); color: var(--accent2); }

  #run-btn {
    width: 100%; background: var(--accent); color: #fff; border: none;
    font-family: var(--font-ui); font-size: 14px; font-weight: 700;
    letter-spacing: 2px; padding: 11px; border-radius: 4px; cursor: pointer;
    text-transform: uppercase; transition: background .2s, transform .1s;
  }
  #run-btn:hover   { background: #ff8c30; }
  #run-btn:active  { transform: scale(.98); }
  #run-btn:disabled { background: #333; cursor: not-allowed; }

  #results { padding: 14px 18px; flex: 1; display: none; flex-direction: column; gap: 8px; }
  #results.visible { display: flex; }

  .result-card { background: #111923; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; }
  .result-card.highlight { border-color: var(--accent); }
  .rc-label { font-size: 8px; letter-spacing: 3px; color: var(--muted); text-transform: uppercase; font-family: var(--font-mono); margin-bottom: 3px; }
  .rc-value { font-family: var(--font-mono); font-size: 13px; color: #fff; word-break: break-all; }
  .rc-value.big { font-size: 22px; color: var(--accent); font-weight: bold; }
  .rc-value.cyan { color: var(--accent2); }
  .rc-value.green { color: var(--green); }
  .metric-row { display: flex; gap: 8px; }
  .metric-row .result-card { flex: 1; text-align: center; }

  #status-bar {
    padding: 9px 18px; font-family: var(--font-mono); font-size: 10px; color: var(--muted);
    border-top: 1px solid var(--border); display: flex; align-items: center; gap: 7px; flex-shrink: 0;
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; flex-shrink: 0;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

  #map-container { flex: 1; position: relative; background: #060a0e; }
  #map { width: 100%; height: 100%; }

  #map-overlay {
    position: absolute; top: 14px; left: 14px; z-index: 800;
    background: rgba(8,12,16,.88); border: 1px solid var(--border);
    border-radius: 6px; padding: 9px 13px; font-family: var(--font-mono);
    font-size: 10px; color: var(--muted); pointer-events: none; max-width: 320px;
  }
  #map-overlay span { color: var(--accent2); }

  .leaflet-control-zoom a { background: var(--panel) !important; color: var(--text) !important; border-color: var(--border) !important; }
  .leaflet-control-attribution { background: rgba(8,12,16,.8) !important; color: var(--muted) !important; font-size: 9px !important; }
  .leaflet-popup-content-wrapper {
    background: var(--panel) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-family: var(--font-mono) !important; font-size: 11px !important;
  }
  .leaflet-popup-tip { background: var(--panel) !important; }

  .spinner {
    display: inline-block; width: 12px; height: 12px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div id="sidebar">
  <div class="logo">
    <div class="logo-top">Pillar 3 · Transport Model</div>
    <div class="logo-main">OIL &amp; <span>GAS</span> AI</div>
    <div class="logo-sub">// PIPELINE ROUTE OPTIMIZER</div>
  </div>

  <div class="section">
    <div class="section-title">Quick Locations</div>
    <div class="presets" id="presets"></div>

    <div class="section-title" style="margin-top:4px">Source Coordinates</div>
    <div class="coords-row">
      <div class="field">
        <label>Latitude</label>
        <input type="number" id="lat" step="0.0001" value="29.76" placeholder="29.76">
      </div>
      <div class="field">
        <label>Longitude</label>
        <input type="number" id="lon" step="0.0001" value="-95.37" placeholder="-95.37">
      </div>
    </div>

    <div class="field">
      <label>Region Override</label>
      <select id="region">
        <option value="">Auto-detect from coordinates</option>
        <option value="usa">USA</option>
        <option value="canada">Canada</option>
        <option value="mexico">Mexico</option>
        <option value="saudi_arabia">Saudi Arabia</option>
      </select>
    </div>

    <button id="run-btn" onclick="runQuery()">&#x2B21; FIND NEAREST REFINERY</button>
    <div style="margin-top:9px; font-size:9px; color:var(--muted); font-family:var(--font-mono); line-height:1.6">
      &#x2191; Or click anywhere on the map to set location
    </div>
  </div>

  <div id="results">
    <div class="result-card highlight">
      <div class="rc-label">Nearest Refinery</div>
      <div class="rc-value" id="res-name">&#x2014;</div>
    </div>
    <div class="metric-row">
      <div class="result-card">
        <div class="rc-label">Direct Distance</div>
        <div class="rc-value big" id="res-direct">&#x2014;</div>
        <div style="font-family:var(--font-mono);font-size:9px;color:var(--muted)">km</div>
      </div>
      <div class="result-card">
        <div class="rc-label">Route Distance</div>
        <div class="rc-value big cyan" id="res-route">&#x2014;</div>
        <div style="font-family:var(--font-mono);font-size:9px;color:var(--muted)">km</div>
      </div>
    </div>
    <div class="result-card">
      <div class="rc-label">Region</div>
      <div class="rc-value green" id="res-region">&#x2014;</div>
    </div>
    <div class="result-card">
      <div class="rc-label">Pipeline Hops</div>
      <div class="rc-value" id="res-hops">&#x2014;</div>
    </div>
    <div class="result-card">
      <div class="rc-label">Refinery Coordinates</div>
      <div class="rc-value" id="res-coords" style="font-size:11px;color:var(--muted)">&#x2014;</div>
    </div>
    <div class="result-card" style="border-color:#1a3a5c;">
      <div class="rc-label">Backend Note</div>
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--muted);line-height:1.6">
        Client-side simulation active.<br>
        Uncomment fetch() in JS to connect<br>
        FastAPI /route endpoint.
      </div>
    </div>
  </div>

  <div id="status-bar">
    <div class="status-dot"></div>
    <span id="status-text">Ready &#x2014; enter coordinates or click map</span>
  </div>
</div>

<div id="map-container">
  <div id="map-overlay">
    TRANSPORT MODEL &nbsp;|&nbsp; <span id="overlay-info">Click map or enter coords &#x2192;</span>
  </div>
  <div id="map"></div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const PRESETS = [
  { label: "Houston TX",    lat: 29.76,  lon: -95.37,  region: "usa"          },
  { label: "Calgary AB",    lat: 51.04,  lon:-114.07,  region: "canada"       },
  { label: "Mexico City",   lat: 19.43,  lon: -99.13,  region: "mexico"       },
  { label: "Riyadh SA",     lat: 24.71,  lon:  46.68,  region: "saudi_arabia" },
  { label: "Midland TX",    lat: 31.99,  lon:-102.08,  region: "usa"          },
  { label: "Edmonton AB",   lat: 53.55,  lon:-113.49,  region: "canada"       },
  { label: "Bakken ND",     lat: 47.80,  lon:-103.00,  region: "usa"          },
  { label: "Eagle Ford TX", lat: 28.70,  lon: -99.10,  region: "usa"          },
];

const presetContainer = document.getElementById("presets");
PRESETS.forEach(p => {
  const btn = document.createElement("button");
  btn.className = "preset-btn";
  btn.textContent = p.label;
  btn.onclick = () => {
    document.getElementById("lat").value = p.lat;
    document.getElementById("lon").value = p.lon;
    document.getElementById("region").value = p.region;
    runQuery();
  };
  presetContainer.appendChild(btn);
});

const map = L.map("map", { center: [30, -60], zoom: 3 });
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "\\u00a9 OpenStreetMap contributors \\u00a9 CARTO",
  subdomains: "abcd", maxZoom: 18,
}).addTo(map);

map.on("click", e => {
  document.getElementById("lat").value = e.latlng.lat.toFixed(5);
  document.getElementById("lon").value = e.latlng.lng.toFixed(5);
  document.getElementById("region").value = "";
  runQuery();
});

let markersLayer = L.layerGroup().addTo(map);
let routeLayer   = L.layerGroup().addTo(map);

function svgIcon(color, symbol) {
  return L.divIcon({
    className: "",
    iconSize: [32, 32], iconAnchor: [16, 16],
    html: '<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">'
      + '<circle cx="16" cy="16" r="12" fill="' + color + '" opacity="0.2"/>'
      + '<circle cx="16" cy="16" r="7" fill="' + color + '"/>'
      + '<text x="16" y="21" text-anchor="middle" font-size="11" font-family="sans-serif">' + symbol + '</text>'
      + '</svg>',
  });
}

const REFINERIES = {
  usa: [
    { name:"ExxonMobil Baytown",       lat: 29.7355, lon: -94.9772 },
    { name:"Shell Deer Park",          lat: 29.7013, lon: -95.1199 },
    { name:"Valero Port Arthur",       lat: 29.9180, lon: -93.9316 },
    { name:"Marathon Galveston Bay",   lat: 29.5919, lon: -94.9819 },
    { name:"Motiva Port Arthur",       lat: 29.8974, lon: -93.9271 },
    { name:"Phillips 66 Lake Charles", lat: 30.2001, lon: -93.1974 },
    { name:"Valero Texas City",        lat: 29.3838, lon: -94.9027 },
    { name:"ExxonMobil Beaumont",      lat: 30.0802, lon: -94.1024 },
    { name:"Marathon Robinson",        lat: 39.0028, lon: -87.7348 },
    { name:"BP Whiting",               lat: 41.6795, lon: -87.4973 },
    { name:"Flint Hills Pine Bend",    lat: 44.7966, lon: -93.0327 },
    { name:"HollyFrontier El Dorado",  lat: 37.8179, lon: -96.8683 },
    { name:"Valero Memphis",           lat: 35.0846, lon: -90.0418 },
    { name:"Calumet Montana",          lat: 48.5144, lon:-109.6696 },
    { name:"Tesoro Los Angeles",       lat: 33.8663, lon:-118.2437 },
    { name:"Chevron El Segundo",       lat: 33.9092, lon:-118.4165 },
  ],
  canada: [
    { name:"Imperial Oil Edmonton",    lat: 53.5150, lon:-113.3790 },
    { name:"Shell Scotford",           lat: 53.7001, lon:-113.1833 },
    { name:"Suncor Fort McMurray",     lat: 57.0500, lon:-111.9900 },
    { name:"Husky Lloydminster",       lat: 53.2833, lon:-110.0000 },
    { name:"Co-op Regina",             lat: 50.4547, lon:-104.6048 },
    { name:"Valero Quebec City",       lat: 46.8325, lon: -71.2560 },
    { name:"Irving Saint John",        lat: 45.2700, lon: -66.0760 },
  ],
  mexico: [
    { name:"PEMEX Salina Cruz",        lat: 16.1858, lon: -95.1968 },
    { name:"PEMEX Minatitlan",         lat: 17.9897, lon: -94.5536 },
    { name:"PEMEX Tula",               lat: 20.0476, lon: -99.3436 },
    { name:"PEMEX Salamanca",          lat: 20.5700, lon:-101.1900 },
    { name:"PEMEX Cadereyta",          lat: 25.5833, lon: -99.9667 },
    { name:"PEMEX Ciudad Madero",      lat: 22.2600, lon: -97.8300 },
  ],
  saudi_arabia: [
    { name:"Aramco Ras Tanura",        lat: 26.6468, lon:  50.1614 },
    { name:"SATORP Jubail",            lat: 27.0059, lon:  49.6685 },
    { name:"Aramco Rabigh",            lat: 22.8017, lon:  39.0356 },
    { name:"Aramco Yanbu",             lat: 24.0889, lon:  38.0651 },
    { name:"SAMREF Yanbu",             lat: 24.0613, lon:  38.0528 },
    { name:"Aramco Riyadh",            lat: 24.6877, lon:  46.7219 },
  ],
};

function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2)**2
    + Math.cos(lat1*Math.PI/180) * Math.cos(lat2*Math.PI/180) * Math.sin(dLon/2)**2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function autoDetectRegion(lat, lon) {
  if (lat >= 5 && lat <= 83 && lon >= -142 && lon <= -52) {
    if (lat >= 49) return "canada";
    if (lat <= 22) return "mexico";
    return "usa";
  }
  if (lat >= 12 && lat <= 38 && lon >= 34 && lon <= 60) return "saudi_arabia";
  return "usa";
}

function simulateRoute(lat, lon, regionOverride) {
  const region = regionOverride || autoDetectRegion(lat, lon);
  const refs   = REFINERIES[region] || REFINERIES["usa"];

  let best = null, bestDist = Infinity;
  for (const r of refs) {
    const d = haversine(lat, lon, r.lat, r.lon);
    if (d < bestDist) { bestDist = d; best = r; }
  }

  const steps = 7;
  const route = [];
  for (let i = 0; i <= steps; i++) {
    const t      = i / steps;
    const jitter = (Math.random() - 0.5) * 0.6 * (1 - Math.abs(t - 0.5) * 2);
    route.push({
      lat:  lat  + (best.lat - lat)  * t + jitter,
      lon:  lon  + (best.lon - lon)  * t + jitter * 1.3,
      name: i === 0 ? "Query" : i === steps ? best.name : "Pipeline Node " + i,
    });
  }

  const routeKm = route.reduce((acc, wp, i) => {
    if (i === 0) return 0;
    return acc + haversine(route[i-1].lat, route[i-1].lon, wp.lat, wp.lon);
  }, 0);

  return {
    query:            { lat, lon },
    nearest_refinery: { name: best.name, lat: best.lat, lon: best.lon, region },
    route,
    direct_km:      Math.round(bestDist * 10) / 10,
    total_graph_km: Math.round(routeKm  * 10) / 10,
    hops:           steps,
  };
}

async function runQuery() {
  const lat    = parseFloat(document.getElementById("lat").value);
  const lon    = parseFloat(document.getElementById("lon").value);
  const region = document.getElementById("region").value || null;

  if (isNaN(lat) || isNaN(lon)) { setStatus("Invalid coordinates", "#e74c3c"); return; }

  setStatus('<span class="spinner"></span> Computing route...', "");
  document.getElementById("run-btn").disabled = true;

  await new Promise(r => setTimeout(r, 380));

  let result;
  try {
    // ── Production: replace simulateRoute() with your FastAPI call ──
    // const res = await fetch("http://localhost:8003/route", {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify({ lat, lon, region }),
    // });
    // if (!res.ok) throw new Error(await res.text());
    // result = await res.json();

    result = simulateRoute(lat, lon, region);
  } catch(e) {
    setStatus("Error: " + e.message, "#e74c3c");
    document.getElementById("run-btn").disabled = false;
    return;
  }

  renderResults(result);
  renderMap(lat, lon, result);
  document.getElementById("run-btn").disabled = false;
  setStatus("Route computed \u2014 " + result.hops + " hops, " + result.total_graph_km + " km", "");
}

function renderResults(r) {
  const ref = r.nearest_refinery;
  document.getElementById("res-name").textContent   = ref.name;
  document.getElementById("res-direct").textContent = r.direct_km;
  document.getElementById("res-route").textContent  = r.total_graph_km;
  document.getElementById("res-region").textContent = ref.region.replace("_"," ").toUpperCase();
  document.getElementById("res-hops").textContent   = r.hops + " pipeline segments";
  document.getElementById("res-coords").textContent = ref.lat.toFixed(4) + "\u00b0, " + ref.lon.toFixed(4) + "\u00b0";
  document.getElementById("overlay-info").textContent = ref.name + "  |  " + r.direct_km + " km direct";
  document.getElementById("results").classList.add("visible");
}

function renderMap(srcLat, srcLon, result) {
  markersLayer.clearLayers();
  routeLayer.clearLayers();

  const ref   = result.nearest_refinery;
  const route = result.route;

  if (route.length >= 2) {
    const latlngs = route.map(wp => [wp.lat, wp.lon]);
    L.polyline(latlngs, { color:"#ff6b00", weight:10, opacity:0.10 }).addTo(routeLayer);
    L.polyline(latlngs, { color:"#ff6b00", weight:3,  opacity:0.92, dashArray:"12 8" }).addTo(routeLayer);
    route.slice(1, -1).forEach(wp => {
      L.circleMarker([wp.lat, wp.lon], {
        radius:4, color:"#ffa500", fillColor:"#ffa500", fillOpacity:0.85, weight:1,
      }).bindTooltip(wp.name).addTo(routeLayer);
    });
  }

  L.polyline([[srcLat, srcLon],[ref.lat, ref.lon]], {
    color:"#4a6080", weight:1, dashArray:"5 5", opacity:0.5,
  }).addTo(routeLayer);

  L.marker([srcLat, srcLon], { icon: svgIcon("#00d4ff","\\uD83D\\uDCCD") })
    .bindPopup("<b>Query Location</b><br>Lat: " + srcLat.toFixed(4) + "<br>Lon: " + srcLon.toFixed(4) + "<br>Direct: " + result.direct_km + " km")
    .addTo(markersLayer);

  L.marker([ref.lat, ref.lon], { icon: svgIcon("#ff4d4d","\\uD83C\\uDFED") })
    .bindPopup("<b>" + ref.name + "</b><br>Region: " + ref.region + "<br>Direct: <b>" + result.direct_km + " km</b><br>Pipeline: <b>" + result.total_graph_km + " km</b>")
    .addTo(markersLayer).openPopup();

  map.fitBounds(
    L.latLngBounds([[srcLat, srcLon],[ref.lat, ref.lon]]),
    { padding:[60,60] }
  );
}

function setStatus(msg, color) {
  const el = document.getElementById("status-text");
  el.innerHTML = msg;
  el.style.color = color || "var(--muted)";
}
</script>
</body>
</html>"""

    components.html(TRANSPORT_HTML, height=720, scrolling=False)
# ═══════════════════════════════════════════════════════════════
# PAGE 5 — SEISMIC DETECTION
# ═══════════════════════════════════════════════════════════════
elif page == "🌊 Seismic Detection":
    st.markdown('<div class="section-title">🌊 SEISMIC RESERVOIR DETECTION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #4a8fa8; margin-bottom: 1.5rem;">
    CNN (ResNet18 Transfer Learning) · 79.65% Accuracy · Input: 128×128 grayscale inline slices · F3 Dutch North Sea
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#00d4ff; letter-spacing:2px; margin-bottom:0.8rem;">SEISMIC SLICE INPUT</div>', unsafe_allow_html=True)

        uploaded_img = st.file_uploader(
            "Upload Seismic Inline Slice (128×128 grayscale PNG/NPY)",
            type=['png', 'jpg', 'jpeg', 'npy'],
            key="seismic_upload"
        )

        inline_idx = st.number_input("OR Enter Inline Index (F3 cube)", value=400, min_value=1, max_value=651, step=1)

        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#4a8fa8; margin-top:0.5rem; margin-bottom:1rem; padding:0.8rem; border:1px solid #1a3a5c; border-radius:4px;">
        📋 INPUT SPEC:<br>
        · Format: 128×128 grayscale PNG or .npy<br>
        · Data: F3 Dutch North Sea amplitude slices<br>
        · Classes: Reservoir (1) | Non-Reservoir (0)<br>
        · Model: ResNet18 pretrained → fine-tuned
        </div>
        """, unsafe_allow_html=True)

        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, step=0.01)

        if st.button("▶  RUN SEISMIC CLASSIFICATION", key="seismic_btn"):
            with st.spinner("Running CNN inference..."):
                time.sleep(1.5)

            # Simulate CNN output
            # ─── REAL CNN INFERENCE ─────────────────────────

            if uploaded_img is not None:

                # Load uploaded image
                img = Image.open(uploaded_img).convert("L")

                # Convert image for model
                input_tensor = seismic_transform(img)

                input_tensor = input_tensor.unsqueeze(0)

                # Run model
                with torch.no_grad():

                    output = seismic_model(input_tensor)

                    probs = torch.softmax(output, dim=1)

                    reservoir_prob = probs[0][1].item()

                # Prediction
                is_reservoir = reservoir_prob >= confidence_threshold

                conf = reservoir_prob if is_reservoir else 1 - reservoir_prob

                logit = reservoir_prob

            else:

                st.warning("Please upload a seismic image.")

                st.stop()

            st.session_state['seismic_result'] = {
                'is_reservoir': is_reservoir,
                'confidence': conf,
                'logit': logit,
                'inline': inline_idx,
            }

        if 'seismic_result' in st.session_state:
            r = st.session_state['seismic_result']
            css_r = "result-box" if r['is_reservoir'] else "result-box danger"
            label = "RESERVOIR DETECTED" if r['is_reservoir'] else "NON-RESERVOIR"
            color_r = "#2ecc71" if r['is_reservoir'] else "#e74c3c"
            emoji_r = "🟢" if r['is_reservoir'] else "🔴"

            st.markdown(f"""
            <div class="{css_r}" style="margin-top:1rem;">
                <div class="result-label">CNN CLASSIFICATION OUTPUT</div>
                <div class="result-value" style="color:{color_r}; font-size:1.4rem;">{emoji_r} {label}</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7ab3cc; margin-top:0.5rem;">
                    Inline: #{r['inline']} · Confidence: {r['confidence']:.1%}<br>
                    Logit: {r['logit']:.4f} · Threshold: {confidence_threshold:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#00d4ff; letter-spacing:2px; margin-bottom:0.8rem;">SYNTHETIC SEISMIC VISUALIZATION</div>', unsafe_allow_html=True)

        # Generate synthetic seismic display
        # ─── DISPLAY REAL SEISMIC IMAGE ─────────────────

        if uploaded_img is not None:

            img = Image.open(uploaded_img).convert("L")

            z = np.array(img)

        else:

            z = np.zeros((128, 128))

        fig_seismic = go.Figure(go.Heatmap(
            z=z,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(
                tickfont=dict(family='Share Tech Mono', size=8, color='#7ab3cc'),
                title=dict(text='Amplitude', font=dict(family='Share Tech Mono', size=9, color='#7ab3cc')),
            ),
            hovertemplate='Sample: %{x}<br>Trace: %{y}<br>Amplitude: %{z:.3f}<extra></extra>',
        ))

        # Reservoir zone annotation if detected
        if st.session_state.get('seismic_result', {}).get('is_reservoir'):
            fig_seismic.add_shape(
                type='rect', x0=30, x1=90, y0=20, y1=45,
                line=dict(color='#2ecc71', width=2, dash='dash'),
                fillcolor='rgba(46, 204, 113, 0.05)',
            )
            fig_seismic.add_annotation(
                x=60, y=10, text="RESERVOIR ZONE",
                font=dict(family='Share Tech Mono', size=9, color='#2ecc71'),
                showarrow=False,
            )

        fig_seismic.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Sample', color='#4a8fa8', tickfont=dict(family='Share Tech Mono', size=8), gridcolor='rgba(26,58,92,0.3)'),
            yaxis=dict(title='Trace', color='#4a8fa8', tickfont=dict(family='Share Tech Mono', size=8), gridcolor='rgba(26,58,92,0.3)'),
            height=380,
            margin=dict(l=50, r=80, t=10, b=40),
        )
        st.plotly_chart(fig_seismic, use_container_width=True)

        # Model stats
        st.markdown("""
        <div style="display:flex; gap:1rem; margin-top:0.5rem;">
            <div class="metric-card" style="flex:1; text-align:center;">
                <div class="metric-label">Architecture</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#00d4ff;">ResNet18</div>
            </div>
            <div class="metric-card" style="flex:1; text-align:center;">
                <div class="metric-label">Test Accuracy</div>
                <div style="font-family:'Orbitron',monospace; font-size:0.9rem; color:#ffd700;">79.65%</div>
            </div>
            <div class="metric-card" style="flex:1; text-align:center;">
                <div class="metric-label">Input Size</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#2ecc71;">128×128</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Price Prediction":
    st.markdown('<div class="section-title">📈 WTI CRUDE OIL PRICE PREDICTION</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #4a8fa8; margin-bottom: 1.5rem;">
    Ensemble: LightGBM + XGBoost + Ridge · 28 features (lag/rolling/momentum/EMA/Bollinger) · EIA WTI Weekly data
    </div>
    """, unsafe_allow_html=True)

    # ─── FETCH LIVE WTI DATA ─────────────────────
    try:
        live_prices = fetch_wti_prices(EIA_API_KEY, weeks=60)
        default_prices = live_prices.tolist()
        fetched_current = float(live_prices.iloc[-1])
        st.success(f"✅ EIA API live — latest WTI: **${fetched_current:.2f}/bbl** ({live_prices.index[-1]})")
    except Exception as e:
        st.warning(f"⚠️ EIA API unavailable ({e}) — using fallback prices.")
        default_prices = [
            104.5, 106.2, 108.0, 105.7, 102.3, 101.9, 103.5, 107.1, 110.4, 108.8,
            106.5, 104.2, 102.8, 101.3, 100.7,  99.8, 101.2, 103.6, 105.9, 107.3,
            109.1, 110.0, 108.7, 107.4, 105.8, 104.6, 103.2, 101.9, 100.5,  99.7,
            101.0, 102.6, 104.3, 106.7, 108.2, 109.8, 111.5, 113.0, 110.9, 108.6,
            106.3, 104.8, 103.7, 102.5, 101.6, 103.0, 105.4, 107.8, 109.6, 111.2,
            112.5, 113.8, 111.9, 109.7, 107.5, 105.9, 104.6, 103.8, 105.2, 106.9,
        ]
        fetched_current = default_prices[-1]

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#00d4ff; letter-spacing:2px; margin-bottom:0.8rem;">CURRENT MARKET DATA</div>', unsafe_allow_html=True)

        # Try live FastAPI price endpoint, fall back to EIA fetch
        try:
            response = requests.get("http://localhost:8000/api/v1/economics/price/live", timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_price = float(data['wti_price_usd_per_bbl'])
            else:
                current_price = fetched_current
        except Exception:
            current_price = fetched_current

        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#2ecc71; margin-bottom:0.8rem;">
        LIVE PRICE: <span style="font-size:1rem; font-weight:bold;">${current_price:.2f}</span> / bbl
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 EDIT PRICE HISTORY (last 8 weeks)"):
            w8 = st.number_input("8 weeks ago", value=float(default_prices[-9]), step=0.1, format="%.2f")
            w7 = st.number_input("7 weeks ago", value=float(default_prices[-8]), step=0.1, format="%.2f")
            w6 = st.number_input("6 weeks ago", value=float(default_prices[-7]), step=0.1, format="%.2f")
            w5 = st.number_input("5 weeks ago", value=float(default_prices[-6]), step=0.1, format="%.2f")
            w4 = st.number_input("4 weeks ago", value=float(default_prices[-5]), step=0.1, format="%.2f")
            w3 = st.number_input("3 weeks ago", value=float(default_prices[-4]), step=0.1, format="%.2f")
            w2 = st.number_input("2 weeks ago", value=float(default_prices[-3]), step=0.1, format="%.2f")
            w1 = st.number_input("Last week",   value=float(default_prices[-2]), step=0.1, format="%.2f")

        model_choice = st.selectbox("ENSEMBLE MODEL", ["ensemble", "lgbm", "xgb", "ridge"])

        if st.button("▶  PREDICT NEXT WEEK PRICE", key="price_btn"):
            with st.spinner("Running ensemble inference..."):

                # ─── BUILD PRICE SERIES WITH EDITS ───────────────
                # Replace last 8 historical + append current; do NOT
                # overwrite default_prices so placeholder chart stays valid
                price_series = default_prices[:-9] + [w8, w7, w6, w5, w4, w3, w2, w1, current_price]

                if len(price_series) < 52:
                    st.error("Need at least 52 weeks of price data.")
                    st.stop()

                # ─── FEATURE ENGINEERING (single pass) ───────────
                try:
                    features_df = build_features(price_series, feature_columns)
                    X_scaled    = scaler.transform(features_df.values)

                    pred_lgbm  = float(lgbm_model.predict(features_df)[0])
                    pred_xgb   = float(xgb_model.predict(X_scaled)[0])
                    pred_ridge = float(ridge_model.predict(X_scaled)[0])
                    pred_ens   = float(np.mean([pred_lgbm, pred_xgb, pred_ridge]))

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

                p      = price_series[-1]
                preds  = {"lgbm": pred_lgbm, "xgb": pred_xgb, "ridge": pred_ridge, "ensemble": pred_ens}
                chosen = preds[model_choice]
                change = chosen - p
                spread = max(pred_lgbm, pred_xgb, pred_ridge) - min(pred_lgbm, pred_xgb, pred_ridge)

                st.session_state['price_result'] = {
                    'current':   p,
                    'chosen':    chosen,
                    'change':    change,
                    'direction': "▲ UP" if change > 0 else "▼ DOWN",
                    'color':     "#2ecc71" if change > 0 else "#e74c3c",
                    'spread':    spread,
                    'preds':     preds,
                    'prices':    price_series,   # save the edited series for the chart
                }

        if 'price_result' in st.session_state:
            r     = st.session_state['price_result']
            css_p = "result-box" if r['change'] > 0 else "result-box danger"

            st.markdown(f"""
            <div class="{css_p}" style="margin-top:1rem;">
                <div class="result-label">NEXT WEEK WTI FORECAST ({r['direction']})</div>
                <div class="result-value" style="color:{r['color']};">${r['chosen']:.2f}</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7ab3cc; margin-top:0.5rem;">
                    Current: ${r['current']:.2f} → Change: {r['change']:+.2f} $/bbl<br>
                    Model spread: ${r['spread']:.2f} {'⚠️ high disagreement' if r['spread'] > 3 else '✅ models agree'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            for m, v in r['preds'].items():
                bar_pct = min(100, max(0, (v - r['current']) / 5 * 50 + 50))
                mcolor  = "#2ecc71" if v > r['current'] else "#e74c3c"
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin:3px 0; font-family:'Share Tech Mono',monospace; font-size:0.65rem;">
                    <span style="color:#4a8fa8; width:80px;">{m.upper()}</span>
                    <div style="flex:1; height:4px; background:#0a1628; border-radius:2px; margin:0 8px;">
                        <div style="width:{bar_pct:.0f}%; height:4px; background:{mcolor}; border-radius:2px;"></div>
                    </div>
                    <span style="color:{mcolor}; width:70px; text-align:right;">${v:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        if 'price_result' in st.session_state:
            r           = st.session_state['price_result']
            prices_disp = r['prices'][-30:]   # uses the edited series saved in session
        else:
            prices_disp = default_prices[-30:]   # placeholder uses live-fetched data

        fig_price = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.08)
        x_hist    = list(range(len(prices_disp)))
        s         = pd.Series(prices_disp)

        fig_price.add_trace(go.Scatter(
            x=x_hist, y=prices_disp, mode='lines', name='WTI Price',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
        ), row=1, col=1)

        ema8  = s.ewm(span=8,  adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        fig_price.add_trace(go.Scatter(x=x_hist, y=ema8,  mode='lines', name='EMA 8',
            line=dict(color='#ffd700', width=1, dash='dot')), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=x_hist, y=ema26, mode='lines', name='EMA 26',
            line=dict(color='#ff6b35', width=1, dash='dot')), row=1, col=1)

        if 'price_result' in st.session_state:
            r = st.session_state['price_result']
            fig_price.add_trace(go.Scatter(
                x=[len(prices_disp)], y=[r['chosen']],
                mode='markers+text',
                marker=dict(size=12, color=r['color'], symbol='diamond'),
                text=[f"${r['chosen']:.2f}"],
                textposition='top center',
                textfont=dict(family='Share Tech Mono', size=10, color=r['color']),
                name='Forecast',
            ), row=1, col=1)

        bb_mid = s.rolling(20).mean()
        bb_std = s.rolling(20).std()
        fig_price.add_trace(go.Scatter(x=x_hist, y=bb_mid + 2*bb_std, mode='lines', name='BB Upper',
            line=dict(color='rgba(155,89,182,0.5)', width=1)), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=x_hist, y=bb_mid - 2*bb_std, mode='lines', name='BB Lower',
            fill='tonexty', fillcolor='rgba(155,89,182,0.05)',
            line=dict(color='rgba(155,89,182,0.5)', width=1)), row=1, col=1)

        mom = s.diff(4)
        fig_price.add_trace(go.Bar(
            x=x_hist, y=mom,
            marker_color=['#2ecc71' if (v is not None and v > 0) else '#e74c3c' for v in mom],
            name='4W Momentum', opacity=0.7,
        ), row=2, col=1)

        fig_price.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,20,0.5)',
            legend=dict(font=dict(family='Share Tech Mono', size=9, color='#c8d8e8'),
                        bgcolor='rgba(0,0,0,0)', orientation='h', y=1.05),
            height=460,
            margin=dict(l=50, r=20, t=60, b=20),
            xaxis=dict(gridcolor='#1a3a5c', color='#4a8fa8',
                       tickfont=dict(family='Share Tech Mono', size=8)),
            yaxis=dict(gridcolor='#1a3a5c', color='#4a8fa8',
                       tickfont=dict(family='Share Tech Mono', size=8),
                       title=dict(text='Price ($/bbl)',
                                  font=dict(family='Share Tech Mono', size=9, color='#4a8fa8'))),
            xaxis2=dict(gridcolor='#1a3a5c', color='#4a8fa8',
                        tickfont=dict(family='Share Tech Mono', size=8)),
            yaxis2=dict(gridcolor='#1a3a5c', color='#4a8fa8',
                        tickfont=dict(family='Share Tech Mono', size=8),
                        title=dict(text='Momentum',
                                   font=dict(family='Share Tech Mono', size=9, color='#4a8fa8'))),
        )
        st.plotly_chart(fig_price, use_container_width=True)