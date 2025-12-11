from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import joblib
import pandas as pd
import difflib
from typing import Optional, Dict, Any, List
import unicodedata

# -------------------------
# App setup
# -------------------------
app = FastAPI(title="Air Quality Forecast API (auto-find encoders)")

MODEL_PATH = os.path.join("models", "best_model.pkl")
ENCODER_DIR = os.path.join("models", "encoders")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

if not os.path.isdir(ENCODER_DIR):
    raise RuntimeError(f"Encoders directory not found: {ENCODER_DIR}")

# Feature order must match training
FEATURE_COLS = [
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

# -------------------------
# Utility: normalize strings (remove accents, lowercase)
# -------------------------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return s.lower()

# -------------------------
# Find encoder file by keywords (robust to accents / case)
# -------------------------
def find_encoder_filename(keywords):
    """
    keywords: iterable of strings to search for in filename (normalized)
    Returns: full path to the first matching file, or None
    """
    files = os.listdir(ENCODER_DIR)
    norm_files = {f: norm(f) for f in files}
    for f, nf in norm_files.items():
        for kw in keywords:
            if norm(kw) in nf:
                return os.path.join(ENCODER_DIR, f)
    # fallback: try fuzzy substring match on files
    for f, nf in norm_files.items():
        for kw in keywords:
            # if any close match
            if difflib.get_close_matches(norm(kw), [nf], n=1, cutoff=0.6):
                return os.path.join(ENCODER_DIR, f)
    return None

def safe_load_encoder_by_keywords(keywords, target_name):
    path = find_encoder_filename(keywords)
    if path is None:
        raise RuntimeError(f"Could not find encoder matching keywords {keywords} in {ENCODER_DIR} for {target_name}. Files: {os.listdir(ENCODER_DIR)}")
    try:
        le = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder from {path}: {e}")
    return le, path

# -------------------------
# Load model & encoders (auto-discover)
# -------------------------
model = joblib.load(MODEL_PATH)

pollutant_le, pollutant_path = safe_load_encoder_by_keywords(["poll", "pollutant", "polluant"], "pollutant_encoder")
influence_le, influence_path = safe_load_encoder_by_keywords(["influence"], "influence_encoder")
evaluation_le, evaluation_path = safe_load_encoder_by_keywords(["evaluation", "évaluation", "type d'evaluation", "type d'évaluation"], "evaluation_encoder")
implantation_le, implantation_path = safe_load_encoder_by_keywords(["implantation"], "implantation_encoder")
site_le, site_path = safe_load_encoder_by_keywords(["site", "code site", "code_site"], "site_encoder")

# Log which files were used (stdout)
print("Loaded encoder files:")
print(" pollutant:", pollutant_path)
print(" influence:", influence_path)
print(" evaluation:", evaluation_path)
print(" implantation:", implantation_path)
print(" site:", site_path)

# -------------------------
# Helper: encoding with fallback
# -------------------------
def encode_with_fallback(le, raw_value: str, mapping: Optional[Dict[str, str]] = None) -> int:
    """
    Attempt to encode a user-provided raw_value using LabelEncoder 'le'.
    1) exact transform
    2) mapping lookup (mapping keys are lower-cased)
    3) case-insensitive exact match against encoder classes
    4) fuzzy match against le.classes_
    Raises ValueError with helpful message if nothing found.
    """
    if raw_value is None:
        raise ValueError("Missing value")

    raw_value_str = str(raw_value)

    # 1) exact match
    try:
        return int(le.transform([raw_value_str])[0])
    except Exception:
        pass

    # 2) mapping (user-provided friendly tokens)
    if mapping:
        mapped = mapping.get(raw_value_str.lower())
        if mapped:
            try:
                return int(le.transform([mapped])[0])
            except Exception:
                pass

    # 3) case-insensitive exact match against encoder classes
    choices = list(le.classes_)
    lowered_choices = {c.lower(): c for c in choices}
    if raw_value_str.lower() in lowered_choices:
        choice = lowered_choices[raw_value_str.lower()]
        return int(le.transform([choice])[0])

    # 4) fuzzy match
    close = difflib.get_close_matches(raw_value_str, choices, n=1, cutoff=0.6)
    if close:
        return int(le.transform([close[0]])[0])

    # nothing worked -> raise clear error listing a few examples
    examples = choices[:10]
    raise ValueError(f"Unknown label '{raw_value_str}'. Allowed examples (first 10): {examples}")

# Optional mappings (expand if you like)
INFLUENCE_MAP = {
    "traffic": "Trafic routier",
    "trafic": "Trafic routier",
    "industrie": "Industriel",
}

POLLUTANT_MAP = {
    "no2": "NO2",
    "pm10": "PM10",
    "o3": "O3",
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
}

EVALUATION_MAP = {
    "reglementaire": "Réglementaire",
    "reg": "Réglementaire",
}

IMPLANTATION_MAP = {
    "urbain": "URBAIN",
    "rural": "RURAL",
}

# -------------------------
# Pydantic models
# -------------------------
class PredictionInput(BaseModel):
    Latitude: float
    Longitude: float
    hour: int
    day: int
    month: int
    year: int
    weekday: int
    weekend: int
    pollutant_encoded: int
    influence_encoded: int
    evaluation_encoded: int
    implantation_encoded: int
    site_encoded: int
    lag_1: float
    lag_24: float
    rolling_3: float

class RawPredictionInput(BaseModel):
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

class PredictionOutput(BaseModel):
    predicted_valeur: float

# --- NEW: Forecast Output Model ---
class ForecastResult(BaseModel):
    """Represents a single hour's prediction in the 24-hour forecast."""
    forecast_time: str
    predicted_valeur: float
# ----------------------------------

# -------------------------
# Core Multi-Step Forecasting Logic
# -------------------------
def forecast_next_24_raw(start_data: RawPredictionInput, model) -> List[ForecastResult]:
    """
    Multi-step 24h forecast starting from the last actual data point (start_data).
    This function handles all feature engineering iteratively.
    
    """
    
    # Pre-calculate time and encoded features from the start data
    try:
        current_time = datetime.fromisoformat(start_data.datetime)
    except Exception:
        current_time = datetime.strptime(start_data.datetime, "%Y-%m-%d %H:%M:%S")

    # --- Prepare Static Features ---
    try:
        static_features = {
            "Latitude": start_data.Latitude,
            "Longitude": start_data.Longitude,
            "pollutant_encoded": encode_with_fallback(pollutant_le, start_data.pollutant, mapping=POLLUTANT_MAP),
            "influence_encoded": encode_with_fallback(influence_le, start_data.influence, mapping=INFLUENCE_MAP),
            "evaluation_encoded": encode_with_fallback(evaluation_le, start_data.evaluation, mapping=EVALUATION_MAP),
            "implantation_encoded": encode_with_fallback(implantation_le, start_data.implantation, mapping=IMPLANTATION_MAP),
            "site_encoded": encode_with_fallback(site_le, start_data.site_code, mapping=None),
        }
    except ValueError as e:
         # This should have been caught in the route, but as a safeguard:
        raise HTTPException(status_code=400, detail=str(e))
        
    # --- Prepare Dynamic Features ---
    dynamic_features = {
        "lag_1": start_data.lag_1,
        "lag_24": start_data.lag_24, # Propagated through the 24 steps
        "rolling_3": start_data.rolling_3,
    }
    
    forecasts: List[ForecastResult] = []

    # Iterate for 24 steps (1-hour forecast each time)
    for i in range(1, 25):
        # Calculate the time for the *prediction* (t+i)
        next_time = current_time + pd.Timedelta(hours=i)

        # Calculate time features for the *current* forecast step (t+i)
        time_features = {
            "hour": next_time.hour,
            "day": next_time.day,
            "month": next_time.month,
            "year": next_time.year,
            "weekday": next_time.weekday(),
            "weekend": 1 if next_time.weekday() in (5, 6) else 0,
        }
        
        # Assemble the feature vector for the model
        feature_vector = {**static_features, **time_features, **dynamic_features}
        
        # Ensure the feature order matches training
        row = pd.DataFrame([[feature_vector[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
        
        # Predict the value at t+i
        y_pred = model.predict(row)[0]
        
        forecasts.append(
            ForecastResult(
                forecast_time=next_time.isoformat(),
                predicted_valeur=float(y_pred)
            )
        )
        
        # Update Dynamic Features for the next iteration (t+i+1)
        
        # New lag_1 (t+i+1's lag_1) is the prediction from the current step (t+i)
        dynamic_features["lag_1"] = float(y_pred)
        
        # Update rolling_3 (approximation: average the previous two values and the new prediction)
        # For simplicity and given the rolling_3 is already a lag-feature, we'll approximate 
        # the next rolling mean based on the new prediction and the last rolling mean.
        # A more complex rolling buffer tracking 3 values would be more accurate, 
        # but this simple update is often sufficient for short-term forecasts.
        dynamic_features["rolling_3"] = (dynamic_features["rolling_3"] * 2 + float(y_pred)) / 3

        # lag_24 remains static for the first 24 hours of forecast

    return forecasts


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Air Quality Forecast API (healthy - encoders auto-discovered)"}

@app.get("/meta")
def meta() -> Dict[str, Any]:
    return {
        "pollutants": list(pollutant_le.classes_),
        "influences": list(influence_le.classes_),
        "evaluations": list(evaluation_le.classes_),
        "implantations": list(implantation_le.classes_),
        "sites_sample": list(site_le.classes_)[:200],
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    row = pd.DataFrame([[getattr(input_data, col) for col in FEATURE_COLS]], columns=FEATURE_COLS)
    y_pred = model.predict(row)[0]
    return PredictionOutput(predicted_valeur=float(y_pred))

@app.post("/predict_raw", response_model=PredictionOutput)
def predict_raw(input_data: RawPredictionInput):
    # parse datetime
    try:
        dt = datetime.fromisoformat(input_data.datetime)
    except Exception:
        try:
            dt = datetime.strptime(input_data.datetime, "%Y-%m-%d %H:%M:%S")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid datetime format. Use 'YYYY-MM-DD HH:MM:SS' or ISO format.")

    hour = dt.hour
    day = dt.day
    month = dt.month
    year = dt.year
    weekday = dt.weekday()
    weekend = 1 if weekday in (5, 6) else 0

    # encode categories
    try:
        pollutant_encoded = encode_with_fallback(pollutant_le, input_data.pollutant, mapping=POLLUTANT_MAP)
        influence_encoded = encode_with_fallback(influence_le, input_data.influence, mapping=INFLUENCE_MAP)
        evaluation_encoded = encode_with_fallback(evaluation_le, input_data.evaluation, mapping=EVALUATION_MAP)
        implantation_encoded = encode_with_fallback(implantation_le, input_data.implantation, mapping=IMPLANTATION_MAP)
        site_encoded = encode_with_fallback(site_le, input_data.site_code, mapping=None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_dict = {
        "Latitude": input_data.Latitude,
        "Longitude": input_data.Longitude,
        "hour": hour,
        "day": day,
        "month": month,
        "year": year,
        "weekday": weekday,
        "weekend": weekend,
        "pollutant_encoded": pollutant_encoded,
        "influence_encoded": influence_encoded,
        "evaluation_encoded": evaluation_encoded,
        "implantation_encoded": implantation_encoded,
        "site_encoded": site_encoded,
        "lag_1": input_data.lag_1,
        "lag_24": input_data.lag_24,
        "rolling_3": input_data.rolling_3,
    }

    row = pd.DataFrame([[feature_dict[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    y_pred = model.predict(row)[0]
    return PredictionOutput(predicted_valeur=float(y_pred))

# --- NEW ROUTE ---

@app.post(
    "/forecast/24h",
    response_model=List[ForecastResult],
    summary="Get 24-Hour Multi-Step Forecast (Raw Input)"
)
def get_24h_forecast(start_data: RawPredictionInput):
    """
    Generates a 24-hour multi-step forecast starting from the provided latest data point.
    """
    
    # 1. Input Validation (Checks categories and datetime before starting the loop)
    try:
        # Check datetime format is valid
        try:
            datetime.fromisoformat(start_data.datetime)
        except Exception:
            datetime.strptime(start_data.datetime, "%Y-%m-%d %H:%M:%S")

        # Check all categories can be encoded (run the encoding logic once)
        encode_with_fallback(pollutant_le, start_data.pollutant, mapping=POLLUTANT_MAP)
        encode_with_fallback(influence_le, start_data.influence, mapping=INFLUENCE_MAP)
        encode_with_fallback(evaluation_le, start_data.evaluation, mapping=EVALUATION_MAP)
        encode_with_fallback(implantation_le, start_data.implantation, mapping=IMPLANTATION_MAP)
        encode_with_fallback(site_le, start_data.site_code, mapping=None)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid datetime format or data type.")

    # 2. Run the forecast logic
    try:
        forecast_data = forecast_next_24_raw(start_data, model)
        return forecast_data
    except Exception as e:
        # Catch unexpected errors during the iterative prediction
        print(f"Prediction error during 24h forecast: {e}")
        # Return a generic 500 error for production environments
        raise HTTPException(status_code=500, detail="Internal prediction error during multi-step forecast.")
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)