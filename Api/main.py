# Api/main.py
import os
import joblib
import json
import unicodedata
import difflib
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn

# Config
MODEL_PATH = os.path.join("models", "best_model.pkl")
ENCODER_DIR = os.path.join("models", "encoders")
REQUIRED_ENCODERS = ["pollutant", "influence", "evaluation", "implantation", "site"]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airquality-api")

# App + CORS (allow Streamlit origin)
app = FastAPI(title="Air Quality Forecast API (final)")
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utils
def norm(s: str) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII").lower()

def find_encoder_filename(keywords):
    if not os.path.isdir(ENCODER_DIR):
        return None
    files = os.listdir(ENCODER_DIR)
    norm_files = {f: norm(f) for f in files}
    # exact stem match
    for f, nf in norm_files.items():
        stem = os.path.splitext(f)[0]
        for kw in keywords:
            if norm(kw) == stem or norm(kw) == nf:
                return os.path.join(ENCODER_DIR, f)
    # substring
    for f, nf in norm_files.items():
        for kw in keywords:
            if norm(kw) in nf:
                return os.path.join(ENCODER_DIR, f)
    # fuzzy
    for f, nf in norm_files.items():
        for kw in keywords:
            if difflib.get_close_matches(norm(kw), [nf], n=1, cutoff=0.75):
                return os.path.join(ENCODER_DIR, f)
    return None

def safe_load_encoder_by_keywords(keywords, target_name):
    path = find_encoder_filename(keywords)
    if path is None:
        files = os.listdir(ENCODER_DIR) if os.path.isdir(ENCODER_DIR) else []
        sample = files[:30]
        raise RuntimeError(f"Could not find encoder for {target_name}. Available (sample): {sample}")
    try:
        le = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder from {path}: {e}")
    return le, path

def load_classes_json_if_exists(stem: str):
    jp = os.path.join(ENCODER_DIR, f"{stem}_classes.json")
    if os.path.exists(jp):
        try:
            with open(jp, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None

# Load model & encoders
log.info("Loading model and encoders...")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

encoders = {}
encoder_keywords_map = {
    "pollutant": ["pollutant", "polluant", "poll"],
    "influence": ["influence", "type d'influence", "trafic", "trafic routier"],
    "evaluation": ["evaluation", "évaluation", "réglementaire", "reglementaire"],
    "implantation": ["implantation", "type d'implantation", "urbain", "rural"],
    "site": ["site", "code site", "code_site", "site_code"],
}

for stem, kws in encoder_keywords_map.items():
    try:
        le, p = safe_load_encoder_by_keywords(kws, stem)
        encoders[stem] = le
        log.info("Loaded encoder %s from %s (classes sample: %s)", stem, p, list(getattr(le, "classes_", []))[:5])
    except Exception as e:
        log.error("Encoder load failed for %s: %s", stem, e)

missing = [r for r in REQUIRED_ENCODERS if r not in encoders]
if missing:
    raise RuntimeError(f"Missing encoders: {missing}. Files in encoder dir: {os.listdir(ENCODER_DIR) if os.path.isdir(ENCODER_DIR) else []}")

log.info("Encoders loaded: %s", list(encoders.keys()))

# Encoding fallback
def encode_with_fallback(le, raw_value: str, mapping: Optional[dict] = None) -> int:
    if raw_value is None:
        raise ValueError("Missing value")
    s = str(raw_value)
    try:
        return int(le.transform([s])[0])
    except Exception:
        pass
    if mapping:
        m = mapping.get(s.lower())
        if m:
            try:
                return int(le.transform([m])[0])
            except Exception:
                pass
    choices = list(le.classes_)
    lowered = {c.lower(): c for c in choices}
    if s.lower() in lowered:
        return int(le.transform([lowered[s.lower()]])[0])
    close = difflib.get_close_matches(s, choices, n=1, cutoff=0.6)
    if close:
        return int(le.transform([close[0]])[0])
    raise ValueError(f"Unknown label '{s}'. Examples: {choices[:10]}")

POLLUTANT_MAP = {"no2": "NO2", "pm10": "PM10", "pm2.5": "PM2.5", "pm25": "PM2.5", "o3": "O3"}
INFLUENCE_MAP = {"traffic": "Trafic routier", "trafic": "Trafic routier", "industrie": "Industriel"}
EVAL_MAP = {"reglementaire": "Réglementaire", "reg": "Réglementaire"}
IMPL_MAP = {"urbain": "URBAIN", "rural": "RURAL"}

# Pydantic models
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

# Routes
@app.get("/")
def root():
    return {"message": "Air Quality Forecast API", "status": "ok"}

@app.get("/meta")
def meta():
    out = {}
    for stem in REQUIRED_ENCODERS:
        classes = load_classes_json_if_exists(stem)
        if classes:
            out_key = stem + "s"
            out[out_key] = classes
        else:
            le = encoders.get(stem)
            out_key = stem + "s"
            out[out_key] = list(getattr(le, "classes_", [])) if le is not None else []
    return {
        "pollutants": out.get("pollutants", []),
        "influences": out.get("influences", []),
        "evaluations": out.get("evaluations", []),
        "implantations": out.get("implantations", []),
        "sites_sample": out.get("sites", [])[:500],
    }

@app.post("/forecast/24h", response_model=List[ForecastResult])
def forecast_24h(req: ForecastRequest):
    try:
        try:
            start_dt = datetime.fromisoformat(req.datetime)
        except Exception:
            start_dt = datetime.strptime(req.datetime, "%Y-%m-%d %H:%M:%S")

        pollutant_encoded = encode_with_fallback(encoders["pollutant"], req.pollutant, mapping=POLLUTANT_MAP)
        influence_encoded = encode_with_fallback(encoders["influence"], req.influence, mapping=INFLUENCE_MAP)
        evaluation_encoded = encode_with_fallback(encoders["evaluation"], req.evaluation, mapping=EVAL_MAP)
        implantation_encoded = encode_with_fallback(encoders["implantation"], req.implantation, mapping=IMPL_MAP)
        site_encoded = encode_with_fallback(encoders["site"], req.site_code, mapping=None)

        static = {
            "Latitude": req.Latitude,
            "Longitude": req.Longitude,
            "pollutant_encoded": pollutant_encoded,
            "influence_encoded": influence_encoded,
            "evaluation_encoded": evaluation_encoded,
            "implantation_encoded": implantation_encoded,
            "site_encoded": site_encoded,
        }
        dynamic = {"lag_1": req.lag_1, "lag_24": req.lag_24, "rolling_3": req.rolling_3}

        results = []
        for i in range(1, 25):
            next_time = start_dt + pd.Timedelta(hours=i)
            time_feats = {
                "hour": next_time.hour,
                "day": next_time.day,
                "month": next_time.month,
                "year": next_time.year,
                "weekday": next_time.weekday(),
                "weekend": 1 if next_time.weekday() in (5, 6) else 0,
            }
            feature_order = [
                "Latitude", "Longitude",
                "hour", "day", "month", "year",
                "weekday", "weekend",
                "pollutant_encoded", "influence_encoded", "evaluation_encoded", "implantation_encoded", "site_encoded",
                "lag_1", "lag_24", "rolling_3",
            ]
            feature_dict = {**static, **time_feats, **dynamic}
            row = [feature_dict[k] for k in feature_order]
            y_pred = float(model.predict([row])[0])
            results.append(ForecastResult(forecast_time=next_time.isoformat(), predicted_valeur=y_pred))
            dynamic["lag_1"] = y_pred
            dynamic["rolling_3"] = (dynamic["rolling_3"] * 2 + y_pred) / 3
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Forecast error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

if __name__ == "__main__":
    log.info("Starting API on 127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
