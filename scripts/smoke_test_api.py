"""Manual smoke test against a running API (e.g. after `docker compose up`).

    python scripts/smoke_test_api.py

Automated tests live in tests/ and run with `pytest`.
"""
import requests, json, os, sys
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from airquality.features import history_window  # noqa: E402

API = "http://127.0.0.1:8000"

def test_root():
    try:
        r = requests.get(f"{API}/", timeout=3)
        print("/ ->", r.status_code, r.text)
    except Exception as e:
        print("Root failed:", e)

def test_meta():
    try:
        r = requests.get(f"{API}/meta", timeout=5)
        print("/meta ->", r.status_code)
        if r.ok:
            meta = r.json()
            print("Meta keys:", list(meta.keys()))
            print("Pollutants sample:", meta.get("pollutants", [])[:5])
    except Exception as e:
        print("/meta failed:", e)

def test_forecast():
    processed_path = os.path.join("data", "processed", "df_raw_cleaned.csv")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        row = df.iloc[-1]
        series = df[(df["code site"] == row["code site"]) & (df["Polluant"] == row["Polluant"])]
        history = history_window(series.set_index("date")["valeur"])
        payload = {
            "datetime": row["date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "pollutant": row.get("Polluant", ""),
            "influence": row.get("type d'influence", ""),
            "evaluation": row.get("type d'évaluation", ""),
            "implantation": row.get("type d'implantation", ""),
            "site_code": row.get("code site", ""),
            "recent_values": [float(v) for v in history],
        }
    else:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "datetime": now,
            "Latitude": 43.3,
            "Longitude": 5.37,
            "pollutant": "NO2",
            "influence": "Trafic",
            "evaluation": "mesures fixes",
            "implantation": "Urbaine",
            "site_code": "FR02001",
            "recent_values": [10.0] * 24,
        }
    print("Payload:", json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        r = requests.post(f"{API}/forecast/24h", json=payload, timeout=20)
        print("forecast ->", r.status_code)
        if r.ok:
            data = r.json()
            print("Received forecast items:", len(data))
            print("First item:", data[0])
        else:
            print("Forecast error:", r.text[:400])
    except Exception as e:
        print("Forecast request failed:", e)

if __name__ == "__main__":
    test_root()
    test_meta()
    test_forecast()
