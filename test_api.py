# test_api.py
import requests
import json
from datetime import datetime
import os
import pandas as pd

API = "http://127.0.0.1:8000"

def test_root():
    try:
        r = requests.get(f"{API}/", timeout=3)
        print("/ →", r.status_code, r.text)
    except Exception as e:
        print("Root failed:", e)

def test_meta():
    try:
        r = requests.get(f"{API}/meta", timeout=5)
        print("/meta →", r.status_code)
        if r.ok:
            meta = r.json()
            print("Meta keys:", list(meta.keys()))
            print("Pollutants sample:", meta.get("pollutants", [])[:5])
    except Exception as e:
        print("/meta failed:", e)

def test_forecast_with_sample():
    # try to build a payload from processed CSV if available
    processed_path = os.path.join("data", "processed", "df_raw_cleaned.csv")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        # pick a row with non-null lag features
        row = df.dropna(subset=["lag_1", "lag_24", "rolling_3"]).iloc[-1]
        payload = {
            "datetime": row["date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "pollutant": row.get("Polluant", ""),
            "influence": row.get("type d'influence", ""),
            "evaluation": row.get("type d'évaluation", ""),
            "implantation": row.get("type d'implantation", ""),
            "site_code": row.get("code site", ""),
            "lag_1": float(row["lag_1"]),
            "lag_24": float(row["lag_24"]),
            "rolling_3": float(row["rolling_3"]),
        }
    else:
        # fallback synthetic payload (likely to be encoded via fallback logic)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "datetime": now,
            "Latitude": 43.295,
            "Longitude": 5.375,
            "pollutant": "NO2",
            "influence": "Trafic routier",
            "evaluation": "Réglementaire",
            "implantation": "URBAIN",
            "site_code": "01001A",  # change to a valid code if needed
            "lag_1": 15.0,
            "lag_24": 14.0,
            "rolling_3": 14.5,
        }

    print("Forecast payload:", json.dumps(payload, indent=2, ensure_ascii=False))
    try:
        r = requests.post(f"{API}/forecast/24h", json=payload, timeout=20)
        print("forecast ->", r.status_code)
        if r.ok:
            data = r.json()
            print("Received forecast length:", len(data))
            print("First item:", data[0])
    except Exception as e:
        print("Forecast call failed:", e)

if __name__ == "__main__":
    print("Testing API connectivity...")
    test_root()
    test_meta()
    test_forecast_with_sample()
