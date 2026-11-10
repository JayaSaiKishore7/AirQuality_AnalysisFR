"""Build the small reference files the live pipeline needs from the training data.

- data/reference/stations.csv          station code -> coordinates
- data/reference/drift_reference.csv   sample of training rows (+ model predictions),
                                       the same number per calendar month, so the drift
                                       check can compare a live window with the same season

Re-run after retraining:  python scripts/build_reference.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airquality import config  # noqa: E402
from airquality.artifacts import load_model  # noqa: E402
from airquality.drift import with_predictions  # noqa: E402
from airquality.features import FEATURE_ORDER  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-month", type=int, default=4000, help="reference rows per calendar month")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    usecols = FEATURE_ORDER + [config.COL_VALUE, config.COL_POLLUTANT, config.COL_SITE]
    df = pd.read_csv(config.PROCESSED_PATH, usecols=usecols, dtype={config.COL_SITE: str})
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    stations = (df.groupby(config.COL_SITE)[["Latitude", "Longitude"]].first()
                .reset_index().sort_values(config.COL_SITE))
    stations.to_csv(config.STATIONS_PATH, index=False)
    print(f"Wrote {len(stations)} stations -> {config.STATIONS_PATH}")

    n = min(args.per_month, int(df.groupby("month").size().min()))
    sample = df.groupby("month", group_keys=False).sample(n=n, random_state=args.seed)
    ref = with_predictions(load_model(), sample, extra_cols=("month",))
    ref.to_csv(config.DRIFT_REFERENCE_PATH, index=False)
    print(f"Wrote {len(ref)} reference rows ({n} per month) -> {config.DRIFT_REFERENCE_PATH}")


if __name__ == "__main__":
    main()
