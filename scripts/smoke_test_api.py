"""Smoke test against a running API (e.g. after `docker compose up` or in CI).

    python scripts/smoke_test_api.py            # API_BASE defaults to http://127.0.0.1:8000

Exits non-zero if any check fails, so it can gate a release.
Automated unit tests live in tests/ and run with `pytest`.
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from airquality import config  # noqa: E402
from airquality.features import history_window  # noqa: E402

API = os.getenv("API_BASE", "http://127.0.0.1:8000")


def check_health() -> None:
    r = requests.get(f"{API}/health", timeout=5)
    r.raise_for_status()
    assert r.json().get("status") == "healthy", r.text
    print("/health ok")


def check_meta() -> None:
    r = requests.get(f"{API}/meta", timeout=30)
    r.raise_for_status()
    meta = r.json()
    assert meta.get("pollutants"), "no pollutants in /meta"
    print("/meta ok:", meta["pollutants"][:5])


def forecast_payload() -> dict:
    if config.PROCESSED_PATH.exists():
        df = pd.read_csv(config.PROCESSED_PATH, parse_dates=["date"], dtype={"code site": str})
        row = df.iloc[-1]
        series = df[(df["code site"] == row["code site"]) & (df["Polluant"] == row["Polluant"])]
        history = history_window(series.set_index("date")["valeur"])
        return {
            "datetime": row["date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "pollutant": row["Polluant"],
            "influence": row["type d'influence"],
            "evaluation": row["type d'évaluation"],
            "implantation": row["type d'implantation"],
            "site_code": row["code site"],
            "recent_values": [float(v) for v in history],
        }
    return {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Latitude": 43.3,
        "Longitude": 5.37,
        "pollutant": "NO2",
        "influence": "Trafic",
        "evaluation": "mesures fixes",
        "implantation": "Urbaine",
        "site_code": "FR02001",
        "recent_values": [10.0] * 24,
    }


def check_forecast() -> None:
    payload = forecast_payload()
    r = requests.post(f"{API}/forecast/24h", json=payload, timeout=60)
    if not r.ok:
        print("Payload:", json.dumps(payload, ensure_ascii=False, indent=2))
    r.raise_for_status()
    data = r.json()
    assert len(data) == 24, f"expected 24 forecast hours, got {len(data)}"
    print("/forecast/24h ok, first item:", data[0])


def main() -> int:
    failed = []
    for check in (check_health, check_meta, check_forecast):
        try:
            check()
        except Exception as e:  # report every failing check, then exit non-zero
            print(f"FAIL {check.__name__}: {e}")
            failed.append(check.__name__)
    if failed:
        print(f"{len(failed)} smoke check(s) failed against {API}")
        return 1
    print(f"All smoke checks passed against {API}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
