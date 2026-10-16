"""Check whether recent live data still looks like the training data.

- Data + prediction drift (Evidently) between data/reference/drift_reference.csv
  and the last N days of live observations.
- Forecast skill: archived forecasts scored against what was actually measured,
  compared with a persistence baseline (last known value).

Outputs (monitoring/):
  drift_summary.json   latest result, read by the API and dashboard
  history.csv          one row per run
  drift_report.html    full Evidently report (not committed; uploaded by CI)

Sets `drift=true|false` in $GITHUB_OUTPUT when running in GitHub Actions.

    python scripts/monitor_drift.py --window-days 7
"""
import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airquality import config, geodair  # noqa: E402
from airquality.artifacts import load_artifacts, load_stations  # noqa: E402
from airquality.drift import run_drift, with_predictions  # noqa: E402
from airquality.forecasting import build_live_features, score_past_forecasts  # noqa: E402

MIN_CURRENT_ROWS = 200


def forecast_skill(obs: pd.DataFrame, since: date) -> dict | None:
    files = [p for p in config.FORECASTS_DIR.glob("*.csv") if p.stem != "latest"]
    files = [p for p in files if date.fromisoformat(p.stem) >= since]
    if not files:
        return None
    fc = pd.concat([pd.read_csv(p, dtype={config.COL_SITE: str}) for p in files], ignore_index=True)
    scored = score_past_forecasts(fc, obs)
    if scored.empty:
        return None
    err = (scored["predicted_valeur"] - scored["actual_valeur"]).abs()
    base = (scored["last_valeur"] - scored["actual_valeur"]).abs()
    by_pollutant = (scored.assign(err=err, base=base)
                    .groupby(config.COL_POLLUTANT)[["err", "base"]].mean().round(3)
                    .rename(columns={"err": "mae", "base": "persistence_mae"})
                    .to_dict(orient="index"))
    return {
        "scored_points": int(len(scored)),
        "mae": round(float(err.mean()), 3),
        "persistence_mae": round(float(base.mean()), 3),
        "by_pollutant": by_pollutant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window-days", type=int, default=7)
    args = parser.parse_args()

    today = date.today()
    since = today - timedelta(days=args.window_days)
    obs = geodair.load_observations(since=since - timedelta(days=1))  # +1 day so lag_24 exists at the start
    if obs.empty:
        sys.exit("No live observations found. Run scripts/fetch_geodair.py first.")

    model, encoders = load_artifacts()
    feats = build_live_features(obs, encoders, load_stations())
    feats = feats[feats[config.COL_DATE].dt.date >= since]
    current = with_predictions(model, feats)
    reference = pd.read_csv(config.DRIFT_REFERENCE_PATH, dtype={config.COL_SITE: str})

    config.MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_date": today.isoformat(),
        "window": {"start": since.isoformat(), "end": today.isoformat()},
        "skill": forecast_skill(obs, since),
    }
    if len(current) < MIN_CURRENT_ROWS:
        print(f"Only {len(current)} live rows; skipping drift test (need {MIN_CURRENT_ROWS})")
        summary["drift"] = None
    else:
        result = run_drift(reference, current, html_path=config.MONITORING_DIR / "drift_report.html")
        summary["drift"] = result.to_dict()
        drifted = [c for c, v in result.columns.items() if v["drifted"]]
        print(f"Drift share {result.drift_share:.2f} ({result.n_drifted}/{result.n_columns}) "
              f"dataset_drift={result.dataset_drift} drifted={drifted}")

    (config.MONITORING_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    drift = summary["drift"] or {}
    skill = summary["skill"] or {}
    row = pd.DataFrame([{
        "run_date": today.isoformat(),
        "current_rows": drift.get("current_rows", len(current)),
        "drift_share": drift.get("drift_share"),
        "dataset_drift": drift.get("dataset_drift"),
        "mae": skill.get("mae"),
        "persistence_mae": skill.get("persistence_mae"),
    }])
    history_path = config.MONITORING_DIR / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        row = pd.concat([history[history["run_date"] != today.isoformat()], row], ignore_index=True)
    row.to_csv(history_path, index=False)

    if skill:
        print(f"Forecast MAE {skill['mae']} vs persistence {skill['persistence_mae']} "
              f"over {skill['scored_points']} points")

    if gh_out := os.getenv("GITHUB_OUTPUT"):
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"drift={'true' if drift.get('dataset_drift') else 'false'}\n")


if __name__ == "__main__":
    main()
