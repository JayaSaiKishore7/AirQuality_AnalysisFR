from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import joblib
import pandas as pd


# App setup
app = FastAPI(title="Air Quality Forecast API")

# Model path (best model saved by train_model.py)
MODEL_PATH = os.path.join("models", "best_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Feature order must match training code (train_model.py)
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


# Load label encoders (saved by preprocess_data.py)
ENCODER_DIR = os.path.join("models", "encoders")

pollutant_le = joblib.load(os.path.join(ENCODER_DIR, "polluant_encoder.pkl"))
influence_le = joblib.load(os.path.join(ENCODER_DIR, "type d'influence_encoder.pkl"))
evaluation_le = joblib.load(os.path.join(ENCODER_DIR, "type d'évaluation_encoder.pkl"))
implantation_le = joblib.load(os.path.join(ENCODER_DIR, "type d'implantation_encoder.pkl"))
site_le = joblib.load(os.path.join(ENCODER_DIR, "code site_encoder.pkl"))

# Request / response schemas
class PredictionInput(BaseModel):
    """Low-level: expects already encoded numeric features."""
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
    """High-level: raw values; API will encode + build features."""
    datetime: str           # e.g. "2025-12-10 14:00:00"
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


# Routes
@app.get("/")
def root():
    return {"message": "Air Quality Forecast API is running."}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Low-level prediction: caller sends already preprocessed features.
    """
    row = pd.DataFrame(
        [[getattr(input_data, col) for col in FEATURE_COLS]],
        columns=FEATURE_COLS,
    )
    y_pred = model.predict(row)[0]
    return PredictionOutput(predicted_valeur=float(y_pred))


@app.post("/predict_raw", response_model=PredictionOutput)
def predict_raw(input_data: RawPredictionInput):
    """
    High-level prediction:
    - Parse datetime -> hour/day/month/year/weekday/weekend
    - Encode pollutant, influence, evaluation, implantation, site_code
    - Use lag_1, lag_24, rolling_3 as given
    """


    try:
        dt = datetime.fromisoformat(input_data.datetime)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid datetime format. Use 'YYYY-MM-DD HH:MM:SS'.",
        )

    hour = dt.hour
    day = dt.day
    month = dt.month
    year = dt.year
    weekday = dt.weekday()
    weekend = 1 if weekday in (5, 6) else 0

    #  Encode categorical values
    try:
        pollutant_encoded = int(pollutant_le.transform([input_data.pollutant])[0])
        influence_encoded = int(influence_le.transform([input_data.influence])[0])
        evaluation_encoded = int(evaluation_le.transform([input_data.evaluation])[0])
        implantation_encoded = int(implantation_le.transform([input_data.implantation])[0])
        site_encoded = int(site_le.transform([input_data.site_code])[0])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Encoding error: {e}. Make sure the values exist in training data.",
        )

    # Build feature dict in the exact FEATURE_COLS order
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
