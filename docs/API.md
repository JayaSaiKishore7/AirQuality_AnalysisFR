# API reference

FastAPI backend (`Api/main.py`) for local use. Start it with `docker compose up` or:

```bash
uvicorn Api.main:app --host 127.0.0.1 --port 8000
```

Base URL: `http://127.0.0.1:8000`. Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Liveness check |
| GET | `/health` | Health check (used by Docker Compose) |
| GET | `/meta` | Pollutants, sites and categories known to the model |
| POST | `/forecast/24h` | 24-hour forecast for one station and pollutant |
| GET | `/forecasts/latest` | Forecasts produced by the daily pipeline from live data |
| GET | `/monitoring/drift` | Latest drift and forecast-skill summary |

---

## GET `/`

```json
{"status": "ok"}
```

## GET `/health`

```json
{"status": "healthy"}
```

## GET `/meta`

Categories present in the training data (`data/processed/df_raw_cleaned.csv`).

```json
{
  "pollutants": ["CO", "NO2", "O3", "PM10", "PM2.5"],
  "sites_sample": ["FR02001", "FR02004", "FR02008"],
  "influences": ["Fond", "Industrielle", "Trafic"],
  "evaluations": ["mesures fixes", "mesures indicatives"],
  "implantations": ["Périurbaine", "Rurale nationale", "Rurale près des villes", "Rurale régionale", "Urbaine"]
}
```

## POST `/forecast/24h`

Recursive 24-step forecast. The lag features of each hour (value 1 h and 24 h earlier, mean of
the previous 3 hours) are read from `recent_values`, extended with the model's own predictions.

Request:

```json
{
  "datetime": "2026-10-15 12:00:00",
  "Latitude": 43.401952,
  "Longitude": 4.981919,
  "pollutant": "PM10",
  "influence": "Industrielle",
  "evaluation": "mesures fixes",
  "implantation": "Urbaine",
  "site_code": "FR02008",
  "recent_values": [18.2, 17.9, 17.5, 16.8, 16.0, 15.9, 16.4, 18.1, 21.3, 24.0, 25.2, 24.8,
                    23.9, 22.7, 21.5, 20.8, 20.1, 19.6, 19.9, 20.6, 21.2, 20.7, 20.1, 19.8]
}
```

| Field | Meaning |
|---|---|
| `datetime` | Time of the last known measurement (ISO 8601) |
| `Latitude`, `Longitude` | Station coordinates |
| `pollutant`, `influence`, `evaluation`, `implantation`, `site_code` | Categories as in `/meta`. Unknown spellings are matched to the closest known value. |
| `recent_values` | Exactly 24 hourly measurements, oldest first; the last one is the value at `datetime`. Fill short gaps before sending (the dashboard interpolates gaps of up to 3 hours). |

Response: 24 items, one per hour after `datetime`.

```json
[
  {"forecast_time": "2026-10-15T13:00:00", "predicted_valeur": 19.84},
  {"forecast_time": "2026-10-15T14:00:00", "predicted_valeur": 19.52}
]
```

Errors: `422` if a field is missing, has the wrong type or `recent_values` does not have 24 values;
`500` if a category cannot be encoded or the date is invalid.

## GET `/forecasts/latest`

Forecasts written by the daily GitHub Actions job (`data/live/forecasts/latest.csv`).
Optional query parameters: `site_code`, `pollutant`.

```
GET /forecasts/latest?site_code=FR02008&pollutant=PM10
```

```json
{
  "issued_at": "2026-10-16 04:31:02",
  "forecasts": [
    {
      "site_code": "FR02008",
      "pollutant": "PM10",
      "base_time": "2026-10-16 02:00:00",
      "horizon_h": 1,
      "forecast_time": "2026-10-16 03:00:00",
      "predicted_valeur": 18.4,
      "last_valeur": 17.9
    }
  ]
}
```

Returns `{"issued_at": null, "forecasts": []}` before the first daily run.

## GET `/monitoring/drift`

Contents of `monitoring/drift_summary.json`, written by `scripts/monitor_drift.py`. Returns `404` before the first run.

```json
{
  "run_date": "2026-10-16",
  "window": {"start": "2026-10-09", "end": "2026-10-16"},
  "skill": null,
  "drift": {
    "dataset_drift": true,
    "drift_share": 0.67,
    "n_drifted": 6,
    "n_columns": 9,
    "columns": {
      "valeur": {"method": "Wasserstein distance (normed)", "score": 0.22, "threshold": 0.1, "drifted": true}
    }
  }
}
```

`skill` is `null` until archived forecasts can be compared with real measurements. It then contains
`scored_points`, `mae` (forecast mean absolute error), `persistence_mae` (error of repeating the last
value, the baseline to beat) and the same two errors per pollutant in `by_pollutant`.
