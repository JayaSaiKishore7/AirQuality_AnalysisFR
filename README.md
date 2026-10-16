# Air Quality Forecasting: PACA, France

[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-FFD21E)](https://huggingface.co/spaces/JayaSaiKishore/airquality-paca)
[![CI](https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR/actions/workflows/ci.yml/badge.svg)](https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR/actions/workflows/ci.yml)
[![Daily forecast](https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR/actions/workflows/daily-forecast.yml/badge.svg)](https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR/actions/workflows/daily-forecast.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

24-hour forecasts of hourly pollutant concentrations (NO₂, O₃, PM10, PM2.5, CO) for the ATMO SUD monitoring
stations in the Provence-Alpes-Côte d'Azur (PACA) region, refreshed every day from live government data.

**▶ [Open the live demo](https://huggingface.co/spaces/JayaSaiKishore/airquality-paca)**

![Live forecast page on Hugging Face: station map and measured vs. forecast chart](images/live-forecast-page1.png)

## What it does

- **Pulls live data daily.** A GitHub Actions job downloads the latest hourly measurements from
  [Geod'Air](https://www.data.gouv.fr/datasets/5b98b648634f415309d52a50), the French national air quality database.
- **Forecasts the next 24 hours** for about 90 station/pollutant series with an XGBoost model.
- **Monitors itself.** [Evidently](https://www.evidentlyai.com/) checks whether recent data has drifted away from the training
  data, and old forecasts are scored against what was actually measured. Drift opens a GitHub issue automatically.
- **Publishes the results** as a static page (Plotly charts and map) on a free Hugging Face Space.

## Architecture

```mermaid
flowchart LR
    G["Geod'Air<br/>hourly measurements"] --> F

    subgraph GA["GitHub Actions: daily at 04:30 UTC"]
        F["fetch_geodair.py<br/>download + clean"] --> P["make_forecasts.py<br/>XGBoost, 24h ahead"]
        P --> M["monitor_drift.py<br/>Evidently drift +<br/>forecast skill"]
        M --> S["build_site.py<br/>static HTML page"]
    end

    S --> R[("Git repository<br/>data, forecasts,<br/>monitoring history")]
    S --> H["Hugging Face<br/>static Space"]
    M -. "drift detected" .-> I["GitHub issue"]

    subgraph LOCAL["Local / Docker Compose"]
        T["preprocess_data.py<br/>train_model.py + MLflow"] --> MOD[("models/")]
        MOD --> API["FastAPI<br/>/forecast/24h"]
        API --> UI["Streamlit dashboard"]
    end

    MOD --> P
```

Every push and pull request runs the test suite (`ci.yml`).

## Tech stack

| Area | Tools |
|---|---|
| Data & ML | pandas, NumPy, scikit-learn, XGBoost |
| Monitoring | Evidently |
| Serving | FastAPI, Streamlit |
| Visualisation | Plotly (Python and Plotly.js) |
| Experiments | MLflow (training runs), TensorFlow/Keras (LSTM/GRU comparison notebook) |
| Testing | pytest |
| Automation & deployment | GitHub Actions, Docker Compose, Hugging Face Spaces |

## Model

- **Target:** hourly concentration per station and pollutant.
- **Features:** station coordinates and type, pollutant, calendar features (hour, day, month, weekday, weekend),
  and recent history (value 1 h and 24 h earlier, mean of the previous 3 hours). Every feature uses only
  data from before the hour being predicted.
- **Per-pollutant scaling:** concentrations differ ~100× between pollutants (CO ≈ 0.2 mg/m³, O₃ ≈ 70 µg/m³).
  The target and history features are divided by each pollutant's median training level, so the model weighs
  every pollutant equally; predictions are converted back to real units.
- **Training:** Random Forest and XGBoost compared on a chronological 80/20 split. The best model by RMSE that is
  small enough to deploy is saved (XGBoost, 4.8 MB; the Random Forest scores slightly better but is 278 MB).
  Forecasts are made recursively, one hour at a time, for 24 hours.
- **Data:** ATMO SUD hourly measurements, January–November 2025.

### Results

Baseline: **persistence**, which repeats the last measured value. A useful model has to beat it.

**1 hour ahead**, held-out test set (14 Sep – 19 Nov 2025, 135,432 hourly values):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost (deployed)** | **3.69** | **6.45** | **0.931** |
| Random Forest | 3.66 | 6.36 | 0.933 |
| Persistence baseline | 3.99 | 7.20 | 0.915 |

**24 hours ahead**, live Geod'Air data (three forecasts issued 12–14 Oct 2026, 6,318 hourly values), mean absolute error:

| Pollutant | XGBoost | Persistence | Improvement |
|---|---|---|---|
| O₃ (µg/m³) | 14.4 | 25.2 | 43% |
| NO₂ (µg/m³) | 7.4 | 10.7 | 30% |
| PM10 (µg/m³) | 4.9 | 6.3 | 23% |
| CO (mg/m³) | 0.051 | 0.057 | 12% |
| PM2.5 (µg/m³) | 3.1 | 3.1 | no gain (−1%) |

The live comparison covers only three days so far; the daily pipeline keeps scoring every forecast against the
measurements and publishes the running error on the live page.

**Known limitations:**
- PM2.5 forecasts are no better than persistence yet.
- Live data is raw (unvalidated) while the training data is validated, which is one reason drift is detected.
- Only stations seen during training can be forecast.

An earlier version leaked the target: the 3-hour mean included the hour being predicted. Fixing it lowered the
1-hour test score (MAE 3.42, inflated by the leak, → 3.69), but training and forecasting now see the same inputs:
before the fix, a 24-hour backtest beat persistence by only 5–14%. A test in `tests/test_features.py` guards
against the leak coming back.

## Quick start

```bash
git clone https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR.git
cd AirQuality_AnalysisFR
docker compose up --build
```

Then open the dashboard at http://localhost:8501 and the API docs at http://localhost:8000/docs.

**Without Docker:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

pytest                             # run the tests
streamlit run app.py               # dashboard (works with or without the API)
uvicorn Api.main:app --port 8000   # API, see docs/API.md
```

**Run the daily pipeline locally:**

```bash
python scripts/fetch_geodair.py --days 3
python scripts/make_forecasts.py
python scripts/monitor_drift.py
python scripts/build_site.py       # then open site/index.html
```

**Retrain:**

```bash
python scripts/preprocess_data.py  # rebuilds data/processed/ from the monthly files in data/raw/
python scripts/train_model.py      # logs runs to MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db
python scripts/build_reference.py  # refreshes the drift reference
```

## Project structure

```
├── airquality/            Shared package: features, Geod'Air client, forecasting, drift, static page
├── Api/                   FastAPI backend
├── app.py                 Streamlit dashboard
├── scripts/               Pipeline, training and helper scripts
├── tests/                 pytest suite
├── data/
│   ├── raw/               Monthly Geod'Air exports, 2025
│   ├── processed/         Cleaned training data
│   ├── reference/         Station coordinates, drift reference sample
│   └── live/              Daily observations and forecasts (updated by the pipeline)
├── models/                Trained model and label encoders
├── monitoring/            Drift and forecast-skill history
├── site/                  Generated static page (deployed to Hugging Face)
├── notebooks/             Exploratory analysis and LSTM/GRU experiments
├── deploy/huggingface/    Hugging Face Space card
├── docs/API.md            API reference
└── .github/workflows/     CI, daily pipeline, deployment
```

| Requirements file | Used for |
|---|---|
| `requirements-api.txt` | FastAPI backend (Docker image) |
| `requirements-pipeline.txt` | Daily pipeline and tests (GitHub Actions) |
| `requirements.txt` | Streamlit dashboard |
| `requirements-dev.txt` | Everything above, plus MLflow and the notebooks |

## Screenshots

Local Streamlit dashboard:

| Historical data explorer | 24-hour forecast |
|---|---|
| ![Streamlit data explorer](images/dashboard-history-pm10.png) | ![Streamlit 24h forecast](images/dashboard-forecast-o3.png) |

## License

[MIT](LICENSE)
