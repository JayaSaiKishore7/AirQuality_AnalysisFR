# Api/main.py
import os
import json
import joblib
import difflib
import unicodedata
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn

# -------------------------
# Config
# -------------------------
MODEL_PATH = os.path.join("models", "best_model.pkl")
ENCODER_DIR = os.path.join("models", "encoders")
REQUIRED_ENCODERS = ["pollutant", "influence", "evaluation", "implantation", "site"]

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airquality-api")

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="Air Quality Forecast API (Robust encoders + CORS)")

# Allow Streamlit & local origins (adjust/add your host if needed)
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

# -------------------------
# Utilities: normalization & discovery
# -------------------------
def norm(s: str) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII").lower()

def find_encoder_filename(keywords):
    """
    Search for an encoder file in ENCODER_DIR using normalized keywords.
    Preference order:
      1) exact normalized stem matches
      2) substring matches
      3) fuzzy match
    Returns full path or None.
    """
    if not os.path.isdir(ENCODER_DIR):
        return None

    files = os.listdir(ENCODER_DIR)
    norm_files = {f: norm(f) for f in files}
    candidates = []

    # 1) exact stem match
    for f, nf in norm_files.items():
        stem = os.path.splitext(f)[0]
        for kw in keywords:
            if norm(kw) == stem or norm(kw) == nf:
                candidates.append(f)
    if len(candidates) == 1:
        return os.path.join(ENCODER_DIR, candidates[0])
    if len(candidates) > 1:
        log.warning("Multiple exact encoder candidates for %s: %s", keywords, candidates)
        # pick shortest name as heuristic
        candidates_sorted = sorted(candidates, key=len)
        return os.path.join(ENCODER_DIR, candidates_sorted[0])

    # 2) substring
    candidates = []
    for f, nf in norm_files.items():
        for kw in keywords:
            if norm(kw) in nf:
                candidates.append(f)
    if len(candidates) == 1:
        return os.path.join(ENCODER_DIR, candidates[0])
    if len(candidates) > 1:
        candidates_sorted = sorted(candidates, key=len)
        log.info("Multiple substring candidates for %s: %s. Picking %s", keywords, candidates, candidates_sorted[0])
        return os.path.join(ENCODER_DIR, candidates_sorted[0])

    # 3) fuzzy
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
        raise RuntimeError(
            f"Could not find encoder matching {keywords} for '{target_name}' in {ENCODER_DIR}. "
            f"Available files (first 30): {sample}"
        )
    try:
        le = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder from {path}: {e}")
    return le, path

def load_classes_json_if_exists(stem: str) -> Optional[List[str]]:
    json_path = os.path.join(ENCODER_DIR, f"{stem}_classes.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None

# -------------------------
# Load model & encoders on startup
# -------------------------
log.info("Starting API - loading model and encoders...")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed loading model: {e}")

# auto-discover encoders
encoders = {}
encoder_paths = {}
encoder_keywords_map = {
    "pollutant": ["pollutant", "polluant", "poll"],
    "influence": ["influence", "type d'influence", "type dinfluence"],
    "evaluation": ["evaluation", "évaluation", "type d'évaluation", "type devaluation"],
    "implantation": ["implantation", "type d'implantation"],
    "site": ["site", "code site", "code_site"],
}

for stem, kws in encoder_keywords_map.items():
    try:
        le, p = safe_load_encoder_by_keywords(kws, stem)
        encoders[stem] = le
        encoder_paths[stem] = p
        log.info("Loaded encoder '%s' from %s (classes: %s)", stem, p, getattr(le, "classes_", [])[:5])
    except Exception as e:
        log.error("Encoder load failed for %s: %s", stem, e)

missing = [r for r in REQUIRED_ENCODERS if r not in encoders]
if missing:
    log.error("Missing encoders: %s. Available encoder files: %s", missing, os.listdir(ENCODER_DIR) if os.path.isdir(ENCODER_DIR) else [])
    raise RuntimeError(f"Missing encoders: {missing}")

log.info("All required encoders loaded: %s", list(encoders.keys()))

# -------------------------
# Encoding helper with fallbacks
# -------------------------
def encode_with_fallback(le, raw_value: str, mapping: Optional[dict] = None) -> int:
    if raw_value is None:
        raise ValueError("Missing value")
    raw_value_str = str(raw_value)

    # 1) exact
    try:
        return int(le.transform([raw_value_str])[0])
    except Exception:
        pass

    # 2) mapping
    if mapping:
        mapped = mapping.get(raw_value_str.lower())
        if mapped:
            try:
                return int(le.transform([mapped])[0])
            except Exception:
                pass

    # 3) case-insensitive
    choices = list(le.classes_)
    lowered = {c.lower(): c for c in choices}
    if raw_value_str.lower() in lowered:
        return int(le.transform([lowered[raw_value_str.lower()]])[0])

    # 4) fuzzy
    close = difflib.get_close_matches(raw_value_str, choices, n=1, cutoff=0.6)
    if close:
        return int(le.transform([close[0]])[0])

    # 5) helpful error listing some options
    examples = choices[:10]
    raise ValueError(f"Unknown label '{raw_value_str}'. Allowed examples: {examples}")

# optional mapping dicts (extend as needed)
INFLUENCE_MAP = {"traffic": "Trafic routier", "trafic": "Trafic routier", "industrie": "Industriel"}
POLLUTANT_MAP = {"no2": "NO2", "pm10": "PM10", "o3": "O3", "pm2.5": "PM2.5", "pm25": "PM2.5"}
EVALUATION_MAP = {"reglementaire": "Réglementaire", "reg": "Réglementaire"}
IMPLANTATION_MAP = {"urbain": "URBAIN", "rural": "RURAL"}

# -------------------------
# Pydantic models
# -------------------------
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
    # optional: recent_values: List[float]  # could be added for more accurate rolling buffer

class ForecastResult(BaseModel):
    forecast_time: str
    predicted_valeur: float

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Air Quality Forecast API", "status": "ok"}

@app.get("/meta")
def meta():
    """Return metadata (classes) for front-end dropdowns"""
    meta = {}
    for stem in REQUIRED_ENCODERS:
        classes_json = load_classes_json_if_exists(stem)
        if classes_json:
            meta[stem + "s"] = classes_json
            continue
        # fallback to encoder.classes_
        le = encoders.get(stem)
        if le is not None:
            meta[stem + "s"] = list(le.classes_)
        else:
            meta[stem + "s"] = []
    # limit some lists for performance where appropriate
    meta_out = {
        "pollutants": meta.get("pollutants", []),
        "influences": meta.get("influences", []),
        "evaluations": meta.get("evaluations", []),
        "implantations": meta.get("implantations", []),
        "sites_sample": meta.get("sites", [])[:500],
    }
    return meta_out

@app.post("/forecast/24h", response_model=List[ForecastResult])
def forecast_24h(req: ForecastRequest):
    """
    Multi-step 24h forecast starting from the last known point 'datetime' (the forecast will produce t+1..t+24).
    """
    try:
        # parse datetime robustly
        try:
            start_dt = datetime.fromisoformat(req.datetime)
        except Exception:
            start_dt = datetime.strptime(req.datetime, "%Y-%m-%d %H:%M:%S")

        # encode categories with fallbacks
        pollutant_encoded = encode_with_fallback(encoders["pollutant"], req.pollutant, mapping=POLLUTANT_MAP)
        influence_encoded = encode_with_fallback(encoders["influence"], req.influence, mapping=INFLUENCE_MAP)
        evaluation_encoded = encode_with_fallback(encoders["evaluation"], req.evaluation, mapping=EVALUATION_MAP)
        implantation_encoded = encode_with_fallback(encoders["implantation"], req.implantation, mapping=IMPLANTATION_MAP)
        site_encoded = encode_with_fallback(encoders["site"], req.site_code, mapping=None)

        # static features
        static = {
            "Latitude": req.Latitude,
            "Longitude": req.Longitude,
            "pollutant_encoded": pollutant_encoded,
            "influence_encoded": influence_encoded,
            "evaluation_encoded": evaluation_encoded,
            "implantation_encoded": implantation_encoded,
            "site_encoded": site_encoded,
        }

        # dynamic features initial state
        dynamic = {
            "lag_1": req.lag_1,
            "lag_24": req.lag_24,
            "rolling_3": req.rolling_3,
        }

        results = []
        # iterate hours 1..24
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

            # assemble feature vector in the same order used in training
            feature_order = [
                "Latitude", "Longitude",
                "hour", "day", "month", "year",
                "weekday", "weekend",
                "pollutant_encoded", "influence_encoded", "evaluation_encoded", "implantation_encoded", "site_encoded",
                "lag_1", "lag_24", "rolling_3",
            ]

            feature_dict = {**static, **time_feats, **dynamic}
            row = [feature_dict[k] for k in feature_order]

            # predict
            y_pred = float(model.predict([row])[0])

            results.append(ForecastResult(forecast_time=next_time.isoformat(), predicted_valeur=y_pred))

            # update dynamic features for next step
            dynamic["lag_1"] = y_pred
            # simple rolling update (approximation)
            dynamic["rolling_3"] = (dynamic["rolling_3"] * 2 + y_pred) / 3
            # lag_24 remains static for the first 24 steps

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Error during forecast")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")

# -------------------------
# Run server (when executed directly)
# -------------------------
if __name__ == "__main__":
    log.info("Uvicorn starting on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
