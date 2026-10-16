"""API tests against the real committed model and encoders."""
import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from airquality import config
from Api import main

PAYLOAD = {
    "datetime": "2026-10-15 12:00:00",
    "Latitude": 43.401952,
    "Longitude": 4.981919,
    "pollutant": "PM10",
    "influence": "Industrielle",
    "evaluation": "mesures fixes",
    "implantation": "Urbaine",
    "site_code": "FR02008",
    # last 24 hourly values ending at `datetime`: a smooth daily cycle around 20 µg/m³
    "recent_values": [round(20 + 5 * math.sin(2 * math.pi * h / 24), 2) for h in range(24)],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").status_code == 200


def test_model_matches_feature_order():
    model, encoders = main.model, main.encoders
    assert model.n_features_in_ == 16
    assert set(encoders) == {"pollutant", "influence", "evaluation", "implantation", "site"}


def test_meta(client, tmp_path, monkeypatch):
    path = tmp_path / "processed.csv"
    pd.DataFrame({
        config.COL_POLLUTANT: ["NO2", "PM10", "NO2"],
        config.COL_SITE: ["FR02008", "FR03043", "FR02008"],
        config.COL_INFLUENCE: ["Fond", "Trafic", "Fond"],
        config.COL_EVALUATION: ["mesures fixes"] * 3,
        config.COL_IMPLANTATION: ["Urbaine"] * 3,
    }).to_csv(path, index=False)
    monkeypatch.setattr(main, "DATA_PATH", path)

    meta = client.get("/meta").json()
    assert meta["pollutants"] == ["NO2", "PM10"]
    assert meta["sites_sample"] == ["FR02008", "FR03043"]
    assert meta["evaluations"] == ["mesures fixes"]


def test_forecast_24h(client):
    r = client.post("/forecast/24h", json=PAYLOAD)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data) == 24
    assert data[0]["forecast_time"] == "2026-10-15T13:00:00"
    assert data[-1]["forecast_time"] == "2026-10-16T12:00:00"
    values = [d["predicted_valeur"] for d in data]
    assert all(0 <= v < 500 for v in values)


def test_forecast_rejects_bad_input(client):
    assert client.post("/forecast/24h", json={**PAYLOAD, "pollutant": "zzz"}).status_code == 500
    assert client.post("/forecast/24h", json={**PAYLOAD, "datetime": "not a date"}).status_code == 500
    assert client.post("/forecast/24h", json={"pollutant": "PM10"}).status_code == 422
    # exactly 24 hourly values are required
    assert client.post("/forecast/24h", json={**PAYLOAD, "recent_values": [20.0] * 23}).status_code == 422


def test_latest_forecasts(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FORECASTS_DIR", tmp_path)
    assert client.get("/forecasts/latest").json() == {"issued_at": None, "forecasts": []}

    pd.DataFrame({
        "issued_at": ["2026-10-16 05:00:00"] * 2,
        config.COL_SITE: ["FR02008", "FR03043"],
        config.COL_POLLUTANT: ["PM10", "NO2"],
        "base_time": ["2026-10-16 03:00:00"] * 2,
        "horizon_h": [1, 1],
        "forecast_time": ["2026-10-16 04:00:00"] * 2,
        "predicted_valeur": [21.5, 30.1],
        "last_valeur": [20.0, 29.0],
    }).to_csv(tmp_path / "latest.csv", index=False)

    body = client.get("/forecasts/latest", params={"site_code": "FR03043"}).json()
    assert body["issued_at"] == "2026-10-16 05:00:00"
    assert [f["pollutant"] for f in body["forecasts"]] == ["NO2"]


def test_drift_status(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MONITORING_DIR", tmp_path)
    assert client.get("/monitoring/drift").status_code == 404
    (tmp_path / "drift_summary.json").write_text(json.dumps({"drift": {"dataset_drift": False}}))
    assert client.get("/monitoring/drift").json()["drift"]["dataset_drift"] is False
