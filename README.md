## AirQuality Forecasting – Côte d'Azur Region, France

Real-time air quality forecasting using machine learning, FastAPI, and Streamlit.

---
## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Repository Structure](#repository-structure)
6. [Modelling Approach](#modelling-approach)
7. [API Documentation (FastAPI)](#api-documentation-fastapi)
8. [Streamlit Dashboard](#streamlit-dashboard)
9. [Living System: Daily Data, Forecasts and Drift Monitoring](#living-system-daily-data-forecasts-and-drift-monitoring)
10. [How to Run Locally](#how-to-run-locally)
   - [Using Docker (Recommended)](#option-1-using-docker-recommended)
   - [Local Development (Optional)](#option-2-local-development-optional)
11. [Screenshots](#screenshots)
12. [License](#license)

## Project Overview

Air pollution is a major environmental concern in the Côte d'Azur (PACA) region of France.
This project develops a machine-learning–based forecasting system to predict hourly pollutant concentrations for the next 24 hours, utilising historical air quality data.

- Real-time pollutant forecasting
- Interactive dashboards
- ML model predictions with lag & rolling temporal features
- Data exploration & visualization
- A reproducible ML pipeline
- A FastAPI inference backend
- API endpoints for integration
- Full containerization using Docker

## Key Features

### Machine Learning
- 24-hour ahead pollutant concentration forecasting
- Feature engineering:
  - Lag features (1h, 24h)
  - Rolling statistics
  - Time-based features (hour, weekday, month)
  - Categorical encoding (site, pollutant, influence, evaluation, implantation)
- Models used:
  - Random Forest
  - XGBoost (best-performing model saved)

### FastAPI Backend
- Lightweight inference-only API
- Loads trained model and encoders
- Endpoints for metadata and forecasting
- Docker-ready production configuration

### Streamlit Dashboard
- Interactive pollutant and site selection
- Historical air quality visualization
- 24-hour forecast visualization
- CSV download of predictions
- Real-time API connectivity

### Reproducible Pipeline
- Data preprocessing
- Encoder generation
- Model training and evaluation
- Artifacts stored under `models/`

## System Architecture
```
           ┌───────────────────┐
           │  Raw Air Quality   │
           │      Data (CSV)    │
           └─────────┬─────────┘
                     │
        ┌────────────▼────────────┐
        │   Data Preprocessing     │
        │ Feature Engineering +    │
        │ Label Encoders + Lags    │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │     ML Model Training    │
        │ RandomForest / XGBoost   │
        └────────────┬────────────┘
                     │ (best_model.pkl)
        ┌────────────▼────────────┐
        │       FastAPI API        │
        │    /meta, /forecast/24h  │
        └────────────┬────────────┘
                     │ (JSON)
        ┌────────────▼────────────┐
        │    Streamlit Dashboard   │
        │ Charts + Forecast + UI   │
        └──────────────────────────┘
```
## Tech Stack

```
  Component           -    Technology                               
  Backend API         -    FastAPI, Uvicorn                     
  Frontend UI         -    Streamlit, Plotly, Matplotlib        
  Machine Learning    -    XGBoost, Random Forest, Scikit-Learn 
  Deep Learning       -    Neural Networks, LSTM, GRU 
  Data processing     -    Pandas, NumPy                            
  Serialization       -    Joblib
  Experiment Tracking -    MLflow                                 
  Environment         -    conda
  Deployment          -    Docker, Docker Compose                             
```
## Repository Structure
```
AirQuality_AnalysisFR/
│
├── Api/
│   └── main.py                # FastAPI backend
│
├── data/
│   ├── raw/                   # Raw CSV files
│   ├── processed/             # Processed dataset (used in inference)
│   │   └── df_raw_cleaned.csv
│   └── sample/
│
├── models/
│   ├── best_model.pkl         # Trained ML model
│   └── encoders/              # Saved label encoders
│
├── notebooks/                 # EDA notebooks
│
├── airquality/                # Shared code: features, Geod'Air client, forecasting, drift
│
├── scripts/
│   ├── preprocess_data.py     # Data preprocessing
│   ├── train_model.py         # Model training
│   ├── build_reference.py     # Station coordinates + drift reference sample
│   ├── fetch_geodair.py       # Daily: pull live Geod'Air data
│   ├── make_forecasts.py      # Daily: 24h forecasts for all stations
│   └── monitor_drift.py       # Daily: Evidently drift + forecast skill
│
├── tests/                     # pytest suite
├── monitoring/                # Drift summaries (updated daily)
├── .github/workflows/         # CI, daily forecast cron, HF deploy
│
├── app.py                     # Streamlit dashboard
├── test_api.py                # API test script
├── site/index.html            # Static forecast page (rebuilt daily, published to Hugging Face)
├── requirements.txt           # Streamlit dashboard dependencies
├── requirements-api.txt       # FastAPI backend
├── requirements-pipeline.txt  # Daily jobs + tests
├── requirements-train.txt     # Everything, incl. DVC and MLflow for retraining
├── environment.yml            # Conda environment
├── Dockerfile.api             # FastAPI Dockerfile
├── Dockerfile.streamlit       # Streamlit Dockerfile
├── docker-compose.yml         # Multi-container orchestration
├── .gitignore
├── .dvcignore
└── README.md
```
## Modelling Approach

- Time-based train/test split
- Models evaluated using RMSE
- Best model selected automatically
- Final model saved as `best_model.pkl`
- Used only for inference in production

## API Documentation (FastAPI)
Base URL
```
http://127.0.0.1:8000    http://localhost:8000

```
### GET /
Health check.

Response:
```
{
  "message": "Air Quality Forecast API",
  "status": "ok"
}
```
### GET /meta
Returns available metadata:
```
{
  "pollutants": ["CO", "NO2", "O3", "PM10", "PM2.5"],
  "influences": ["Fond", "Industrielle", "Trafic"],
  "evaluations": ["mesures fixes", "mesures indicatives"],
  "implantations": ["Urbaine", "Rurale", "Périurbaine"],
  "sites_sample": ["FR02001", "FR02004", "FR02008"]
}

```

### POST /forecast/24h
```
{
  "datetime": "2025-11-19 19:00:00",
  "Latitude": 43.4020,
  "Longitude": 4.9819,
  "pollutant": "PM10",
  "influence": "Trafic",
  "evaluation": "mesures fixes",
  "implantation": "Urbaine",
  "site_code": "FR02008",
  "lag_1": 12.4,
  "lag_24": 8.1,
  "rolling_3": 10.3
}

```
### Response (24 items)
```
[
  {
    "forecast_time": "2025-11-20T20:00:00",
    "predicted_valeur": 14.12
  }
]
```
### GET /forecasts/latest
Forecasts made by the daily job from live Geod'Air data. Optional filters: `site_code`, `pollutant`.
```
{
  "issued_at": "2026-10-16 04:31:02",
  "forecasts": [
    {"site_code": "FR02008", "pollutant": "PM10", "base_time": "2026-10-16 02:00:00",
     "horizon_h": 1, "forecast_time": "2026-10-16 03:00:00",
     "predicted_valeur": 18.4, "last_valeur": 17.9}
  ]
}
```

### GET /monitoring/drift
Latest drift and forecast-skill summary (`monitoring/drift_summary.json`).

## Streamlit Dashboard

Features include:
- Live tab: measured values vs. the latest daily forecast, drift status and forecast skill
- Pollutant & Site selection
- Historical charts (7-day trend)
- 24-hour forecast chart
- Downloadable CSV
- API connectivity indicators
- Diagnostic outputs

## Living System: Daily Data, Forecasts and Drift Monitoring

A GitHub Actions cron job (`.github/workflows/daily-forecast.yml`, 04:30 UTC) keeps the system current:

```
Geod'Air (data.gouv.fr)  ──►  fetch_geodair.py   data/live/observations/YYYY-MM-DD.csv  (ATMO SUD, 90-day window)
                         ──►  make_forecasts.py  data/live/forecasts/latest.csv + daily archive
                         ──►  monitor_drift.py   monitoring/drift_summary.json, history.csv, Evidently HTML report
                         ──►  build_site.py      site/index.html  (static page: Plotly charts + map)
                         ──►  commit to main ──► deploy-hf.yml publishes the page to a Hugging Face static Space
```

- **Data source**: the LCSQA/INERIS real-time hourly files
  (`https://files.data.gouv.fr/ineris/lcsqa/concentrations-de-polluants-atmospheriques-reglementes/temps-reel/{YYYY}/FR_E2_{date}.csv`).
  The last 3 days are re-downloaded each run because recent files are still revised.
- **Forecasts**: every station/pollutant known to the encoders with enough recent data (about 90 series) gets a 24h recursive forecast from the existing model.
- **Drift**: Evidently `DataDriftPreset` compares the last 7 days with a 20k-row sample of the training data
  (`data/reference/drift_reference.csv`) on values, lag features, hour, weekday, pollutant, site and model predictions.
  When dataset drift is detected, the workflow opens (or comments on) a GitHub issue labelled `data-drift`. The full HTML report is attached to the run as an artifact.
- **Forecast skill**: archived forecasts are scored against what was later measured and compared with a persistence baseline (repeat the last value).

Run the same steps locally:
```
pip install -r requirements-pipeline.txt
python scripts/fetch_geodair.py --days 3
python scripts/make_forecasts.py
python scripts/monitor_drift.py
python scripts/build_site.py      # then open site/index.html in a browser
```
After retraining, rebuild the reference files with `python scripts/build_reference.py`.

### Tests and CI
`pytest` runs the suite in `tests/` (features, Geod'Air parsing, batch forecasting, drift, static page, API with the real model). `.github/workflows/ci.yml` runs it on every push and pull request.

### Deployment (Hugging Face static Space, free)
The public demo is a single static page, `site/index.html`, published to a Hugging Face **static** Space (free; Docker
and Gradio Spaces need PRO). All data is embedded in the page and the charts are drawn in the browser with Plotly.js,
so no server runs. `.github/workflows/deploy-hf.yml` uploads only that page plus the Space card
(`deploy/huggingface/README.md`), after every daily run and whenever the page changes on `main`. One-time setup:
1. Create a Hugging Face access token with **Write** access.
2. In the GitHub repo: **Settings → Secrets and variables → Actions**: add secret `HF_TOKEN`, and variable
   `HF_SPACE_ID` (e.g. `your-hf-username/airquality-paca`). The Space is created automatically.
3. Run **Actions → Daily forecast → Run workflow**.

The Streamlit dashboard and the FastAPI backend are for local use (`docker compose up`, or see below). Without the
API, the dashboard runs in standalone mode and computes forecasts itself with the same code the API uses.

## How to Run Locally
### Option 1: Using Docker (Recommended)

Prerequisites:

- Docker Desktop is installed and running
```
docker compose up --build
```
#### Access
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
#### To Stop
```
docker compose down
```
### Option 2: Local Development (Optional)

#### 1. Clone the repository
```
git clone https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR.git
cd AirQuality_AnalysisFR
```
#### 2. Create conda environment
```
conda env create -f environment.yml
conda activate airquality-ml
```
#### 3. Preprocess the raw data
```
python scripts\preprocess_data.py
```
#### 4. Train models (and log runs with MLflow)
```
python scripts\train_model.py
```
#### 5. View MLflow UI
```
mlflow ui --port 5000
http://127.0.0.1:5000
```
#### 6. Start FastAPI
```
python Api/main.py
uvicorn Api.main:app --host 127.0.0.1 --port 8000
```
#### 7. Start Streamlit
```
streamlit run app.py
```
## Screenshots

### Dashboard Preview
![History Plot](images/Screenshot%202025-12-11%20211855.png)

### Historical Data View
![History Plot](images/Screenshot%202025-12-11%20212147.png)

### 24hr Forecast Dashboard
![Forecast Plot](images/Screenshot%202025-12-11%20212028.png)

![Forecast Plot](images/Screenshot%202025-12-11%20222832.png)

## License
```
This project is released under the MIT License.
Feel free to use, modify, and distribute.
```

