# Api/main.py
import os
import joblib
import logging
import unicodedata
import difflib
from typing import List
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

MODEL_PATH = "models/best_model.pkl"
ENCODER_DIR = "models/encoders"
DATA_PATH = "data/processed/df_raw_cleaned.csv"

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

def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()

def load_encoder(keywords):
    for f in os.listdir(ENCODER_DIR):
        for k in keywords:
            if norm(k) in norm(f):
                return joblib.load(os.path.join(ENCODER_DIR, f))
    raise RuntimeError(f"Encoder not found for {keywords}")

def encode(le, value: str) -> int:
    try:
        return int(le.transform([value])[0])
    except Exception:
        close = difflib.get_close_matches(value, le.classes_, n=1)
        if close:
            return int(le.transform([close[0]])[0])
        raise ValueError(f"Invalid value: {value}")

@app.on_event("startup")
def load_artifacts():
    global model, encoders

    log.info("Loading model...")
    model = joblib.load(MODEL_PATH)

    encoders["pollutant"] = load_encoder(["polluant"])
    encoders["influence"] = load_encoder(["influence"])
    encoders["evaluation"] = load_encoder(["evaluation"])
    encoders["implantation"] = load_encoder(["implantation"])
    encoders["site"] = load_encoder(["site"])

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

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/meta")
def meta():
    """
    META is sourced directly from the same dataset used in Streamlit.
    This guarantees site codes ALWAYS match.
    """
    df = pd.read_csv(DATA_PATH)

    return {
        "pollutants": sorted(df["Polluant"].dropna().unique().tolist()),
        "sites_sample": sorted(df["code site"].dropna().unique().tolist())[:500],
        "influences": sorted(df["type d'influence"].dropna().unique().tolist()),
        "evaluations": sorted(df["type d'évaluation"].dropna().unique().tolist()),
        "implantations": sorted(df["type d'implantation"].dropna().unique().tolist()),
    }

@app.post("/forecast/24h", response_model=List[ForecastResult])
def forecast(req: ForecastRequest):
    try:
        start = datetime.fromisoformat(req.datetime)

        static = {
            "Latitude": req.Latitude,
            "Longitude": req.Longitude,
            "pollutant_encoded": encode(encoders["pollutant"], req.pollutant),
            "influence_encoded": encode(encoders["influence"], req.influence),
            "evaluation_encoded": encode(encoders["evaluation"], req.evaluation),
            "implantation_encoded": encode(encoders["implantation"], req.implantation),
            "site_encoded": encode(encoders["site"], req.site_code),
        }

        dynamic = {
            "lag_1": req.lag_1,
            "lag_24": req.lag_24,
            "rolling_3": req.rolling_3,
        }

        results = []

        for i in range(1, 25):
            t = start + pd.Timedelta(hours=i)

            row = {
                **static,
                "hour": t.hour,
                "day": t.day,
                "month": t.month,
                "year": t.year,
                "weekday": t.weekday(),
                "weekend": int(t.weekday() >= 5),
                **dynamic,
            }

            order = [
                "Latitude", "Longitude",
                "hour", "day", "month", "year",
                "weekday", "weekend",
                "pollutant_encoded",
                "influence_encoded",
                "evaluation_encoded",
                "implantation_encoded",
                "site_encoded",
                "lag_1", "lag_24", "rolling_3",
            ]

            y = float(model.predict([[row[k] for k in order]])[0])

            results.append(
                ForecastResult(
                    forecast_time=t.isoformat(),
                    predicted_valeur=y
                )
            )

            dynamic["lag_1"] = y
            dynamic["rolling_3"] = (dynamic["rolling_3"] * 2 + y) / 3

        return results

    except Exception as e:
        log.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
