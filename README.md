## 🌍 AirQuality Forecasting – Côte d’Azur Region - France 

Real-time air quality forecasting using machine learning, FastAPI, and Streamlit.

---
## 📌 Table of Contents

 1) Project Overview
 2) Features
 3) System Architecture
 4) Tech Stack
 5) Repository Structure
 6) Data Processing Pipeline
 7) Modelling Approach
 8) API Documentation (FastAPI)
 9) Streamlit Dashboard
10) How to Run Locally
11) Screenshots
12) Future Improvements
13) License


## 📖 Project Overview

Air pollution is a major environmental concern in the Côte d’Azur (PACA) region of France.
This project provides:

- Real-time pollutant forecasting
- Interactive dashboards
- ML model predictions with lag & rolling temporal features
- Data exploration & visualization
- API endpoints for integration

## ⭐ Features
🧠 Machine Learning

- Forecasts next 24 hours of pollutant concentration

- Feature engineering:

- Lag features (1h, 24h)

- Rolling mean

- Time-based features (hour, month, weekday)

- Categorical encodings (site, pollutant, influence type)

This project uses historical air quality data and builds ML models (Random Forest, XGBoost) to forecast hourly pollutant concentration for the next 24 hours.

## ⚡ FastAPI Backend

- Lightweight inference API

- Encodes inputs using saved label encoders

- Returns 24-hour horizon predictions

## 📊 Streamlit Dashboard

- Historical charts

- Real-time REST API integration

- Downloadable CSV forecast

- Interactive UI & filtering

## 📁 Reproducible Pipeline

- Data processing

- Encoder generation

- Model training & evaluation

- Artifacts stored in models/

 ## 🏗️ System Architecture
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
        │    /meta, /forecast      │
        └────────────┬────────────┘
                     │ (JSON)
        ┌────────────▼────────────┐
        │    Streamlit Dashboard   │
        │ Charts + Forecast + UI   │
        └──────────────────────────┘
```
## 🔧 Tech Stack

```
  Component      -    Technology                               
  Backend API    -   **FastAPI**, Uvicorn                     
  Frontend UI    -   **Streamlit**, Plotly, Matplotlib        
  Machine Learning -  **XGBoost**, Random Forest, Scikit-Learn 
  Data           -   Pandas, NumPy                            
  Serialization  -    Joblib                                   
  Environment    -    conda                                    
```

## 📁 Repository Structure
```
AirQuality_AnalysisFR/
│
├── Api/
│   └── main.py                # FastAPI backend
│
├── data/
│   ├── raw/                   # Raw CSV files
│   ├── processed/             # Cleaned dataset + DVC tracked
│   │   └── df_raw_cleaned.csv
│   └── sample/
│
├── models/
│   ├── best_model.pkl         # Saved best ML model
│   └── encoders/              # Label encoders
│
├── notebooks/                 # Jupyter notebooks for EDA
│
├── app.py                     # Streamlit dashboard
├── test_api.py                # API connectivity script
├── environment.yml            # Conda environment file
├── .gitignore
└── README.md
```
## 📡 API Documentation (FastAPI)
Base URL
```
http://127.0.0.1:8000
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
  "pollutants": ["CO", "NO2", "PM10", ...],
  "influences": ["Trafic routier", "Fond", "Industriel"],
  "evaluations": ["Réglementaire", "Mesures indicatives"],
  "implantations": ["URBAIN", "RURAL"],
  "sites_sample": ["FR02001", "FR02004", ...]
}
```

## POST /forecast/24h
```
{
  "datetime": "2025-11-19 19:00:00",
  "Latitude": 43.4020,
  "Longitude": 4.9819,
  "pollutant": "PM10",
  "influence": "Trafic routier",
  "evaluation": "Réglementaire",
  "implantation": "URBAIN",
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
## 🎨 Streamlit Dashboard

Features include:
- Pollutant & Site selection
- Historical charts (7-day trend)
- 24-hour forecast chart
- Downloadable CSV
- API connectivity indicators
 -Diagnostic outputs

## ▶️ How to Run Locally

1️⃣ Clone the repository
```
git clone https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR.git
cd AirQuality_AnalysisFR
```
## 2️⃣ Create conda environment
```
conda env create -f environment.yml
conda activate airquality-ml
```
## 3️⃣ Start FastAPI
```
python Api/main.py
uvicorn Api.main:app --host 127.0.0.1 --port 8000
```
## 4️⃣ Start Streamlit
```
streamlit run app.py
```
## 📄 License
```
This project is released under the MIT License.
Feel free to use, modify, and distribute.
```

  






