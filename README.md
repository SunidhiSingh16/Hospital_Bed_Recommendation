# 🏥 Hospital Bed Demand Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-DD0031?style=flat)](https://xgboost.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat&logo=plotly)](https://plotly.com)
[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat&logo=render)](https://render.com)

A multi-model demand forecasting system for hospital bed capacity planning. Combines **Poisson arrival modelling** with **Gradient Boosting** and **XGBoost** regressors in a configurable weighted ensemble, deployed as an interactive Streamlit dashboard with real-time surge alerting.

> **Live Demo →** *(Deploy on Render — link after deployment)*

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                     │
│   Config · Analysis · Forecasting · Report              │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │   Data Pipeline        │
        │  generate_synthetic()  │
        │  add_dept_breakdown()  │
        │  prepare_ml_features() │
        └───────────┬────────────┘
                    │
       ┌────────────▼─────────────┐
       │    Forecasting Engine    │
       │                          │
       │  ┌─────────────────────┐ │
       │  │  Poisson Predictor  │ │  ← MLE of daily arrival rate λ
       │  │  scipy.stats.poisson│ │    80% prediction interval
       │  └─────────────────────┘ │
       │                          │
       │  ┌──────────┐ ┌────────┐ │
       │  │   GBM    │ │XGBoost │ │  ← Temporal feature regression
       │  │ sklearn  │ │ xgb    │ │    Train/test split 80/20
       │  └────┬─────┘ └───┬────┘ │
       │       │           │      │
       │  ┌────▼───────────▼────┐ │
       │  │  Ensemble (weighted) │ │  ← Poisson 25% | GBM 40% | XGB 35%
       │  └─────────────────────┘ │
       └──────────────────────────┘
                    │
       ┌────────────▼──────────────┐
       │  Outputs                  │
       │  · Next-day point pred.   │
       │  · 7-day forward forecast │
       │  · 95% confidence bands   │
       │  · Surge occupancy alert  │
       │  · Dept-level breakdown   │
       │  · Downloadable report    │
       └───────────────────────────┘
```

---

## Features

- **Dual forecasting paradigm** — statistical Poisson model (interpretable baseline) combined with ensemble ML for capturing non-linear temporal patterns
- **Weighted ensemble** — configurable Poisson + GBM + XGBoost combination with per-model importance attribution
- **7-day forward forecast** — extrapolated predictions with ±1.96σ confidence bands visualised against bed capacity
- **Real-time surge alerting** — threshold-based occupancy alerts (Normal < 70% < Elevated < 85% < Surge) with colour-coded status panels
- **Department-level decomposition** — ICU, Emergency, and General Ward admission streams via stochastic proportional allocation
- **Configurable scenario engine** — Normal, High-Variation, Pandemic, Summer-Surge, and fully custom parameter modes
- **Downloadable outputs** — Markdown executive report and CSV predictions per run

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit 1.31+ | Interactive dashboard, session state |
| Visualisation | Plotly 5.18+ | Interactive charts, stacked area, confidence bands |
| Statistical Model | SciPy (poisson) | Poisson MLE, PMF, prediction intervals |
| ML Models | scikit-learn GBM, XGBoost | Gradient boosted regression |
| Data | pandas, numpy | Feature engineering, rolling statistics |
| Deployment | Render (render.yaml) | Managed PaaS, free tier |

---

## Quickstart

```bash
# Clone
git clone <repository-url>
cd hospital-bed-recommendation

# Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
# → http://localhost:8501
```

---

## Deploy on Render

The repo includes `render.yaml` — import it into Render as a **Web Service**.

| Setting | Value |
|---------|-------|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment | Python |
| Plan | Free |

---

## Project Structure

```
hospital-bed-recommendation/
├── app.py                    # Streamlit dashboard (4 tabs)
├── requirements.txt
├── render.yaml               # Render deployment config
├── .streamlit/
│   └── config.toml           # Dark professional theme
├── src/
│   ├── data_loader.py        # Synthetic data generation + dept breakdown
│   ├── poisson_model.py      # Poisson MLE predictor
│   ├── ml_model.py           # GBM / XGBoost regressor wrapper
│   └── ensemble_model.py     # Weighted ensemble + forward forecaster
└── data/
    └── hospital_admissions.csv
```

---

## Model Details

### Poisson Arrival Model
- Fits λ (arrival rate) as MLE of trailing 30-day window: `λ̂ = mean(X)`
- Computes PMF over discrete admission range
- Returns highest-density 80% prediction interval

### Gradient Boosting / XGBoost
- Features: `day_of_week`, `month`, `is_weekend`, `rolling_mean_7d`, `rolling_std_7d`
- 80/20 chronological train/test split (no data leakage)
- Evaluated on MAE and RMSE; feature importance from `feature_importances_`

### Ensemble
- Weighted linear combination: `ŷ = 0.25·Poisson + 0.40·GBM + 0.35·XGBoost`
- Weights are configurable — tunable for stable vs. high-variance admission regimes
- Feature importance averaged across GBM and XGBoost sub-models

### 7-Day Forward Forecast
- Extrapolates over future calendar features with frozen rolling statistics
- Confidence bands: `ŷ ± 1.96 · σ_rolling` (approximate 95% interval)
- Plotted against user-defined bed capacity threshold

---

## What This Demonstrates

- End-to-end ML pipeline: data generation → feature engineering → multi-model training → evaluation → deployment
- Practical ensemble design with interpretable weighting
- Statistical + ML hybrid forecasting (Poisson as prior, tree models for residual patterns)
- Production-ready Streamlit deployment with session state management

---

## License

MIT License
