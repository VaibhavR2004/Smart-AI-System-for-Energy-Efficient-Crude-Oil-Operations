# Oil & Gas AI Intelligence Platform

> Multi-Model Machine Learning Platform for Upstream & Midstream Oil & Gas Operations

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-orange)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![License](https://img.shields.io/badge/License-Research_Project-lightgrey)

---

## Overview

The **Oil & Gas AI Intelligence Platform** is an end-to-end AI-powered analytics and optimization system designed for the oil and gas industry. The platform integrates multiple machine learning and optimization pipelines into a unified interactive dashboard for:

- 🌊 Seismic Reservoir Detection
- ⚠️ Equipment Failure Prediction
- 🛣️ Pipeline Transport Optimization
- 🛢️ Extraction Rate Forecasting
- 📈 Oil Price Prediction & Economics

The system combines deep learning, classical machine learning, graph optimization, explainable AI, and real-time visualization into a single operational platform.

---

# System Architecture

The platform is divided into 5 major AI pillars:

| Pillar | Objective | Model |
|---|---|---|
| Seismic Detection | Reservoir classification from seismic inline slices | CNN / ResNet18 |
| Failure Prediction | Equipment failure probability estimation | Random Forest + SHAP |
| Transport Optimization | Pipeline route optimization | Dijkstra + KDTree |
| Extraction Prediction | Predict extraction rate (BPD) | XGBoost Regression |
| Price Prediction | Forecast crude oil prices | LightGBM + XGBoost + Ridge |

---

# Features

## 🌊 Seismic Reservoir Detection
- CNN-based seismic classification
- ResNet18 transfer learning
- Processes grayscale inline slices
- Reservoir vs Non-reservoir classification
- PyTorch implementation

### Model Performance
- Accuracy: **79.65%**
- Input Size: **128×128**
- Optimizer: **Adam**
- Loss Function: **CrossEntropyLoss**

---

## ⚠️ Equipment Failure Prediction
- Predicts probability of oilfield equipment failure
- SHAP explainability integration
- Safety-critical inference pipeline
- Handles imbalanced datasets using:
  - SMOTE
  - Class balancing
  - Threshold optimization

### Model Performance
- Accuracy: **94%**
- F1 Score: **0.94**
- Threshold: **0.7708**

---

## 🛣️ Pipeline Transport Optimization
- Global pipeline network routing
- Uses OGIM geospatial pipeline dataset
- Fast nearest-node lookup with KDTree
- Real-world geographic route visualization
- Dijkstra shortest-path routing

### Performance
- ~80,000 nodes
- ~64,000 edges
- ~4,700× faster nearest-node lookup

---

## 🛢️ Extraction Rate Prediction
- Predicts well production output in barrels/day
- Trained on large-scale Petrobras 3W dataset
- XGBoost regression model

### Model Performance

| Metric | Value |
|---|---|
| RMSE | 6205.89 |
| MAE | 1013.04 |
| R² Score | 0.990 |

---

## 📈 Oil Price Prediction
- Ensemble forecasting system
- Uses:
  - LightGBM
  - XGBoost
  - Ridge Regression
- Live WTI price integration using EIA API
- Technical indicator feature engineering

---

# Tech Stack

## Languages
- Python

## Frameworks & Libraries
- Streamlit
- PyTorch
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Plotly
- Folium
- NetworkX
- GeoPandas
- Pandas
- NumPy

## Visualization
- Plotly Dashboards
- Folium Maps
- Interactive Gauges
- SHAP Explainability Charts

## APIs
- EIA API (WTI crude prices)

---

# Dashboard Features

The interactive Streamlit dashboard includes:

- Real-time KPI monitoring
- Failure probability gauges
- SHAP feature contribution charts
- Pipeline route visualization
- Extraction analytics
- Live WTI oil price trends
- Interactive seismic inference

---

# Project Structure

```bash
OilGasAI/
│
├── models/
│   ├── seismic/
│   ├── failure/
│   ├── extraction/
│   └── economics/
│
├── data/
│   ├── seismic/
│   ├── ogim/
│   ├── 3w/
│   └── oreada/
│
├── notebooks/
│   ├── seismic_model.ipynb
│   ├── transport_model.ipynb
│   └── train_failure_model.ipynb
│
├── app/
│   ├── dashboard.py
│   └── api/
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/oil-gas-ai-platform.git
cd oil-gas-ai-platform
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux / Mac
```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Dashboard

```bash
streamlit run dashboard.py
```

---

# API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict/seismic` | POST | Reservoir classification |
| `/predict/failure` | POST | Failure probability + SHAP |
| `/route` | POST | Optimal pipeline route |
| `/predict/extraction` | POST | Extraction rate prediction |
| `/predict/price` | POST | Oil price forecasting |
| `/optimize/cost` | POST | Production optimization |

---

# Datasets Used

| Dataset | Purpose |
|---|---|
| F3 Dutch North Sea | Seismic classification |
| AI4I Dataset | Failure prediction |
| Petrobras 3W | Sensor & extraction modeling |
| OREDA | Equipment reliability |
| OGIM GeoPackage | Pipeline routing |
| EIA WTI Data | Oil price prediction |

---

# Explainable AI (XAI)

The platform integrates **SHAP Explainability** for transparent failure prediction inference.

Benefits:
- Operator trust
- Safety validation
- Root-cause analysis
- Feature attribution

---

# Research Contributions

- Multi-model oil & gas intelligence platform
- Real-time routing optimization
- Explainable safety-critical AI
- Large-scale industrial sensor analytics
- Hybrid ML + graph optimization architecture

---

# Future Improvements

- Real-time IoT sensor ingestion
- Cloud-native deployment
- Kubernetes orchestration
- Reinforcement learning for production optimization
- Digital twin integration
- Predictive maintenance scheduling

---

# Author

**Vaibhav Kumar Rajput**  
SRM Institute of Science and Technology  
Department of Computer Science Engineering

---

# License

This project is developed for:
- Research
- Academic demonstration
- Industrial AI experimentation

---

# Acknowledgements

- Petrobras 3W Dataset
- OGIM Pipeline Dataset
- EIA Open Data API
- PyTorch Community
- Streamlit Framework

---

# Citation

```bibtex
@project{oilgasai2026,
  title={Oil & Gas AI Intelligence Platform},
  author={Vaibhav Kumar Rajput},
  year={2026},
  institution={SRM Institute of Science and Technology}
}
```
