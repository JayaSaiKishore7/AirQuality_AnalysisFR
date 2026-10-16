"""Make 24h forecasts for every PACA station/pollutant from the latest live data.

Writes data/live/forecasts/latest.csv and an archive copy per issue date,
which monitor_drift.py later scores against the real measurements.

    python scripts/make_forecasts.py
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airquality import config, geodair  # noqa: E402
from airquality.artifacts import load_artifacts, load_stations  # noqa: E402
from airquality.forecasting import build_live_features, forecast_all, latest_states  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-days", type=int, default=3, help="live days used to build lag features")
    parser.add_argument("--keep-days", type=int, default=90, help="retention for archived forecasts")
    args = parser.parse_args()

    obs = geodair.load_observations(since=date.today() - timedelta(days=args.history_days))
    if obs.empty:
        sys.exit("No live observations found. Run scripts/fetch_geodair.py first.")

    model, encoders = load_artifacts()
    feats = build_live_features(obs, encoders, load_stations())
    states = latest_states(feats)
    issued_at = pd.Timestamp.now(tz="UTC").floor("s").tz_localize(None)
    fc = forecast_all(model, states, issued_at)

    config.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_csv(config.FORECASTS_DIR / "latest.csv", index=False)
    fc.to_csv(config.FORECASTS_DIR / f"{issued_at.date().isoformat()}.csv", index=False)
    geodair.prune(config.FORECASTS_DIR, keep_days=args.keep_days)

    n_series = fc.groupby([config.COL_SITE, config.COL_POLLUTANT]).ngroups if len(fc) else 0
    print(f"Forecast {n_series} series x 24h ({len(fc)} rows) issued {issued_at}")
    if n_series == 0:
        sys.exit("No series had enough recent data to forecast")


if __name__ == "__main__":
    main()
