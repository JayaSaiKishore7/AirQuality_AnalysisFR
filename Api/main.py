from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import joblib
import pandas as pd
import uvicorn
from typing import List

# ==================== APP SETUP ====================
app = FastAPI(title="Air Quality Forecast API")

# ==================== FILE PATHS ====================
MODEL_PATH = os.path.join("models", "best_model.pkl")
ENCODER_DIR = os.path.join("models", "encoders")

# Check if files exist
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Model not found: {MODEL_PATH}")

if not os.path.exists(ENCODER_DIR):
    raise RuntimeError(f"❌ Encoders directory not found: {ENCODER_DIR}")

# ==================== LOAD MODEL ====================
print("🚀 Loading machine learning model...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# ==================== LOAD ENCODERS ====================
print("\n🔠 Loading encoders...")
encoders = {}

# List all encoder files
encoder_files = os.listdir(ENCODER_DIR)
print(f"Found {len(encoder_files)} encoder files")

# Load each encoder
for file in encoder_files:
    if file.endswith('.pkl'):
        path = os.path.join(ENCODER_DIR, file)
        try:
            # Extract encoder name from filename
            name = file.replace('_encoder.pkl', '').replace('.pkl', '')
            encoders[name] = joblib.load(path)
            print(f"   ✅ Loaded: {name}")
        except Exception as e:
            print(f"   ❌ Failed to load {file}: {e}")

# Make sure we have the required encoders
required = ['pollutant', 'influence', 'evaluation', 'implantation', 'site']
missing = [r for r in required if r not in encoders]
if missing:
    print(f"❌ Missing encoders: {missing}")
    print(f"Available: {list(encoders.keys())}")
    raise RuntimeError(f"Missing encoders: {missing}")

print("✅ All encoders loaded")

# ==================== ENCODER HELPER ====================
def encode_value(encoder_name: str, value: str):
    """Helper to encode a value"""
    if encoder_name not in encoders:
        raise ValueError(f"Unknown encoder: {encoder_name}")
    
    encoder = encoders[encoder_name]
    classes = list(encoder.classes_)
    
    # Try exact match
    if value in classes:
        return encoder.transform([value])[0]
    
    # Try case-insensitive match
    for cls in classes:
        if cls.lower() == value.lower():
            return encoder.transform([cls])[0]
    
    # Return first class as fallback (or raise error)
    print(f"⚠️  Warning: '{value}' not found in {encoder_name} encoder. Using '{classes[0]}'")
    return encoder.transform([classes[0]])[0]

# ==================== PYDANTIC MODELS ====================
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

# ==================== API ROUTES ====================
@app.get("/")
def root():
    return {
        "message": "Air Quality Forecast API",
        "status": "healthy",
        "version": "1.0"
    }

@app.get("/meta")
def get_metadata():
    """Get available metadata"""
    return {
        "pollutants": list(encoders["pollutant"].classes_)[:50],
        "influences": list(encoders["influence"].classes_)[:20],
        "evaluations": list(encoders["evaluation"].classes_)[:20],
        "implantations": list(encoders["implantation"].classes_)[:20],
        "sites_sample": list(encoders["site"].classes_)[:100],
    }

@app.post("/forecast/24h", response_model=List[ForecastResult])
def forecast_24h(request: ForecastRequest):
    """Generate 24-hour forecast"""
    try:
        # Parse datetime
        try:
            dt = datetime.fromisoformat(request.datetime)
        except:
            dt = datetime.strptime(request.datetime, "%Y-%m-%d %H:%M:%S")
        
        # Encode categorical features
        pollutant_encoded = encode_value("pollutant", request.pollutant)
        influence_encoded = encode_value("influence", request.influence)
        evaluation_encoded = encode_value("evaluation", request.evaluation)
        implantation_encoded = encode_value("implantation", request.implantation)
        site_encoded = encode_value("site", request.site_code)
        
        # Generate forecasts
        forecasts = []
        current_lag_1 = request.lag_1
        current_lag_24 = request.lag_24
        current_rolling = request.rolling_3
        
        for hour in range(1, 25):
            # Calculate forecast time
            forecast_time = dt + pd.Timedelta(hours=hour)
            
            # Prepare features
            features = [
                request.Latitude,
                request.Longitude,
                forecast_time.hour,
                forecast_time.day,
                forecast_time.month,
                forecast_time.year,
                forecast_time.weekday(),
                1 if forecast_time.weekday() in (5, 6) else 0,
                pollutant_encoded,
                influence_encoded,
                evaluation_encoded,
                implantation_encoded,
                site_encoded,
                current_lag_1,
                current_lag_24,
                current_rolling,
            ]
            
            # Make prediction
            prediction = model.predict([features])[0]
            
            # Store result
            forecasts.append(ForecastResult(
                forecast_time=forecast_time.isoformat(),
                predicted_valeur=float(prediction)
            ))
            
            # Update lags for next hour
            current_lag_1 = float(prediction)
            current_rolling = (current_rolling * 2 + float(prediction)) / 3
        
        return forecasts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

# ==================== START SERVER ====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🌐 Starting FastAPI server...")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)