##  AirQuality Forecasting – Côte d’Azur Region - France 

Real-time air quality forecasting using machine learning, FastAPI, and Streamlit.

---
##  Table of Contents

 1) Project Overview
 2) Features
 3) System Architecture
 4) Tech Stack
 5) Repository Structure
 6) Data Processing Pipeline
 7) Modelling Approach
 8) API Documentation (FastAPI)
 9) Streamlit Dashboard
10) How to Run the Application
     - Using Docker (Recommended)
     - Local Development (Optional) 
12) Screenshots
13) Future Improvements
14) License


##  Project Overview

Air pollution is a major environmental concern in the Côte d’Azur (PACA) region of France.
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

### Models used:

- Random Forest

- XGBoost (best-performing model saved)

- FastAPI Backend

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

- Artifacts stored under models/

 ##  System Architecture
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
##  Repository Structure
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
├── scripts/
│   ├── preprocess_data.py     # Data preprocessing
│   └── train_model.py         # Model training
│
├── app.py                     # Streamlit dashboard
├── test_api.py                # API test script
├── requirements.txt           # Python dependencies
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

- Final model saved as best_model.pkl

- Used only for inference in production

##  API Documentation (FastAPI)
Base URL
```
http://127.0.0.1:8000    http://localhost:8000

```
## GET /
Health check.
Response
```
{
  "message": "Air Quality Forecast API",
  "status": "ok"
}
```
## GET /meta
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

## POST /forecast/24h
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
## Response (24 items):
```
[
  {
    "forecast_time": "2025-11-20T20:00:00",
    "predicted_valeur": 14.12
  }
]
```
## Streamlit Dashboard

Features include:
- Pollutant & Site selection
- Historical charts (7-day trend)
- 24-hour forecast chart
- Downloadable CSV
- API connectivity indicators
 -Diagnostic outputs

## How to Run Locally
### Option 1: Using Docker (Recommended)

Prerequisites:

- Docker Desktop is installed and running
```
docker compose up --build
```
### Access:
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
### To Stop
```
docker compose down
```
### Option 2: Local Development (Optional)

## 1 Clone the repository
```
git clone https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR.git
cd AirQuality_AnalysisFR
```
## 2 Create conda environment
```
conda env create -f environment.yml
conda activate airquality-ml
```
## 3 Preprocess the Raw Data
```
python scripts\preprocess_data.py

```
## 4 Train Models (and log runs with MLflow)
```
python scripts\train_model.py

```
## 5 View MLflow UI
```
mlflow ui --port 5000
http://127.0.0.1:5000
```
## 6 Start FastAPI
```
python Api/main.py
uvicorn Api.main:app --host 127.0.0.1 --port 8000
```
## 7 Start Streamlit
```
streamlit run app.py
```
## Screenshots

 Dashboard Preview

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

  






