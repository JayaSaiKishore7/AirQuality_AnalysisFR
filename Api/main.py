# Api/main.py
import json
import logging
import difflib
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# allow `python Api/main.py` as well as `uvicorn Api.main:app`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from airquality import config
from airquality.artifacts import load_artifacts
from airquality.features import DYNAMIC_COLS, STATIC_COLS, recursive_forecast

DATA_PATH = config.PROCESSED_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airquality-api")

app = FastAPI(title="Air Quality Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
encoders = {}

def encode(le, value: str) -> int:
    try:
        return int(le.transform([value])[0])
    except Exception:
        close = difflib.get_close_matches(value, le.classes_, n=1)
        if close:
            return int(le.transform([close[0]])[0])
        raise ValueError(f"Invalid value: {value}")

@app.on_event("startup")
def load_artifacts_on_startup():
    global model, encoders

    log.info("Loading model and encoders...")
    model, encoders = load_artifacts()
    log.info("All encoders loaded successfully")

class ForecastRequest(BaseModel):
    datetime: str
    Latitude: float
    Longitude: float
    pollutant: str
    influence: str
    evaluation: str
    implantation: str
    site_code: str
    lag_1: float
    lag_24: float
    rolling_3: float

class ForecastResult(BaseModel):
    forecast_time: str
    predicted_valeur: float

class LiveForecast(BaseModel):
    site_code: str
    pollutant: str
    base_time: str
    horizon_h: int
    forecast_time: str
    predicted_valeur: float
    last_valeur: float

class LiveForecastResponse(BaseModel):
    issued_at: Optional[str]
    forecasts: List[LiveForecast]

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@lru_cache(maxsize=4)
def _load_meta(path: str) -> dict:
    df = pd.read_csv(path, usecols=[
        config.COL_POLLUTANT, config.COL_SITE, config.COL_INFLUENCE,
        config.COL_EVALUATION, config.COL_IMPLANTATION,
    ])
    return {
        "pollutants": sorted(df[config.COL_POLLUTANT].dropna().unique().tolist()),
        "sites_sample": sorted(df[config.COL_SITE].dropna().unique().tolist())[:500],
        "influences": sorted(df[config.COL_INFLUENCE].dropna().unique().tolist()),
        "evaluations": sorted(df[config.COL_EVALUATION].dropna().unique().tolist()),
        "implantations": sorted(df[config.COL_IMPLANTATION].dropna().unique().tolist()),
    }

@app.get("/meta")
def meta():
    """
    META is sourced directly from the same dataset used in Streamlit.
    This guarantees site codes ALWAYS match.
    """
    return _load_meta(str(DATA_PATH))

@app.post("/forecast/24h", response_model=List[ForecastResult])
def forecast(req: ForecastRequest):
    try:
        start = datetime.fromisoformat(req.datetime)

        static = pd.DataFrame([{
            "Latitude": req.Latitude,
            "Longitude": req.Longitude,
            "pollutant_encoded": encode(encoders["pollutant"], req.pollutant),
            "influence_encoded": encode(encoders["influence"], req.influence),
            "evaluation_encoded": encode(encoders["evaluation"], req.evaluation),
            "implantation_encoded": encode(encoders["implantation"], req.implantation),
            "site_encoded": encode(encoders["site"], req.site_code),
        }])[STATIC_COLS]
        dynamic = pd.DataFrame([{
            "lag_1": req.lag_1,
            "lag_24": req.lag_24,
            "rolling_3": req.rolling_3,
        }])[DYNAMIC_COLS]

        fc = recursive_forecast(model, pd.Series([start]), static, dynamic)

        return [
            ForecastResult(forecast_time=t.isoformat(), predicted_valeur=float(y))
            for t, y in zip(pd.to_datetime(fc["forecast_time"]), fc["predicted_valeur"])
        ]

    except Exception as e:
        log.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasts/latest", response_model=LiveForecastResponse)
def latest_forecasts(site_code: Optional[str] = None, pollutant: Optional[str] = None):
    """Forecasts produced by the daily job from live Geod'Air data."""
    path = config.FORECASTS_DIR / "latest.csv"
    if not path.exists():
        return LiveForecastResponse(issued_at=None, forecasts=[])
    df = pd.read_csv(path, dtype={config.COL_SITE: str})
    if site_code:
        df = df[df[config.COL_SITE] == site_code]
    if pollutant:
        df = df[df[config.COL_POLLUTANT] == pollutant]
    df = df.rename(columns={config.COL_SITE: "site_code", config.COL_POLLUTANT: "pollutant"})
    issued_at = str(df["issued_at"].iloc[0]) if len(df) else None
    return LiveForecastResponse(
        issued_at=issued_at,
        forecasts=df.drop(columns="issued_at").to_dict(orient="records"),
    )

@app.get("/monitoring/drift")
def drift_status():
    """Latest drift + forecast-skill summary written by scripts/monitor_drift.py."""
    path = config.MONITORING_DIR / "drift_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No monitoring run yet")
    return json.loads(path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
