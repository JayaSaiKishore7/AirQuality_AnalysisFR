import numpy as np
import pandas as pd
import pytest

from airquality import config
from airquality.drift import DRIFT_COLS, run_drift


def frame(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.gamma(4, 5, n) + shift
    return pd.DataFrame({
        config.COL_VALUE: v,
        "lag_1": v + rng.normal(0, 1, n),
        "lag_24": v + rng.normal(0, 3, n),
        "rolling_3": v + rng.normal(0, 0.5, n),
        "hour": rng.integers(0, 24, n),
        "weekday": rng.integers(0, 7, n),
        "prediction": v + rng.normal(0, 2, n),
        config.COL_POLLUTANT: rng.choice(["NO2", "O3", "PM10"], n),
        config.COL_SITE: rng.choice(["FR02008", "FR03043"], n),
    })


def test_no_drift_on_same_distribution():
    result = run_drift(frame(3000, seed=1), frame(1500, seed=2))
    assert not result.dataset_drift
    assert result.n_columns == len(DRIFT_COLS)


def test_drift_detected_on_shifted_values(tmp_path):
    html = tmp_path / "report.html"
    result = run_drift(frame(3000, seed=1), frame(1500, shift=25, seed=2), html_path=html)
    assert result.dataset_drift
    for col in [config.COL_VALUE, "lag_1", "lag_24", "rolling_3", "prediction"]:
        assert result.columns[col]["drifted"], col
    assert not result.columns["hour"]["drifted"]
    assert html.exists() and html.stat().st_size > 0


def test_drift_reference_file_matches_schema():
    if not config.DRIFT_REFERENCE_PATH.exists():
        pytest.skip("reference not built")
    ref = pd.read_csv(config.DRIFT_REFERENCE_PATH, nrows=5)
    assert list(ref.columns) == DRIFT_COLS
