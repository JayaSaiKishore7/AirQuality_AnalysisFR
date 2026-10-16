import numpy as np
import pandas as pd

from airquality import config
from airquality.features import (
    DYNAMIC_COLS, FEATURE_ORDER, STATIC_COLS, add_lag_features, recursive_forecast, time_features,
)


def test_time_features_weekend_flag():
    ts = pd.Series(pd.to_datetime(["2026-10-16 05:00", "2026-10-17 23:00"]))  # Friday, Saturday
    tf = time_features(ts)
    assert tf["weekday"].tolist() == [4, 5]
    assert tf["weekend"].tolist() == [0, 1]
    assert tf["hour"].tolist() == [5, 23]


def test_feature_order_covers_static_and_dynamic():
    assert set(STATIC_COLS + DYNAMIC_COLS) < set(FEATURE_ORDER)
    assert len(FEATURE_ORDER) == 16


def test_recursive_forecast_updates_lags(lag_model):
    static = pd.DataFrame([dict.fromkeys(STATIC_COLS, 0.0)] * 2)
    dynamic = pd.DataFrame({"lag_1": [10.0, 50.0], "lag_24": [9.0, 49.0], "rolling_3": [9.0, 48.0]})
    start = pd.Series(pd.to_datetime(["2026-10-16 00:00", "2026-10-16 06:00"]))

    fc = recursive_forecast(lag_model, start, static, dynamic, horizon=24)

    assert len(fc) == 48
    s0 = fc[fc["series"] == 0].sort_values("horizon_h")
    # each step feeds its prediction back as lag_1
    assert s0["predicted_valeur"].tolist() == [11.0 + h for h in range(24)]
    assert s0["forecast_time"].iloc[0] == pd.Timestamp("2026-10-16 01:00")
    assert s0["forecast_time"].iloc[-1] == pd.Timestamp("2026-10-17 00:00")
    assert fc.loc[fc["series"] == 1, "predicted_valeur"].iloc[0] == 51.0


def test_add_lag_features_matches_definitions(observations):
    feats = add_lag_features(observations)
    g = feats[feats[config.COL_SITE] == "FR02008"].set_index(config.COL_DATE)[config.COL_VALUE]
    row = feats[(feats[config.COL_SITE] == "FR02008")].iloc[30]
    t = row[config.COL_DATE]
    assert row["lag_1"] == g[t - pd.Timedelta(hours=1)]
    assert row["lag_24"] == g[t - pd.Timedelta(hours=24)]
    expected = np.mean([g[t - pd.Timedelta(hours=k)] for k in range(3)])
    assert np.isclose(row["rolling_3"], expected)


def test_add_lag_features_respects_time_gaps(observations):
    # remove one hour: the next hour must not take an older value as lag_1
    gap = pd.Timestamp("2026-10-11 05:00")
    obs = observations[observations[config.COL_DATE] != gap]
    feats = add_lag_features(obs)
    after = feats[(feats[config.COL_SITE] == "FR02008") & (feats[config.COL_DATE] == gap + pd.Timedelta(hours=1))]
    assert after["lag_1"].isna().all()
    assert not (feats[config.COL_DATE] == gap).any()
