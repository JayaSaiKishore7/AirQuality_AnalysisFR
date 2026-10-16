import numpy as np
import pandas as pd

from airquality import config
from airquality.features import (
    DYNAMIC_COLS, FEATURE_ORDER, HISTORY_HOURS, STATIC_COLS, add_lag_features, history_window,
    recursive_forecast, time_features,
)


class ColumnModel:
    """Stand-in model that returns one feature column, to check what each step sees."""

    def __init__(self, column):
        self.i = FEATURE_ORDER.index(column)

    def predict(self, X):
        return np.asarray(X)[:, self.i]


def test_time_features_weekend_flag():
    ts = pd.Series(pd.to_datetime(["2026-10-16 05:00", "2026-10-17 23:00"]))  # Friday, Saturday
    tf = time_features(ts)
    assert tf["weekday"].tolist() == [4, 5]
    assert tf["weekend"].tolist() == [0, 1]
    assert tf["hour"].tolist() == [5, 23]


def test_feature_order_covers_static_and_dynamic():
    assert set(STATIC_COLS + DYNAMIC_COLS) < set(FEATURE_ORDER)
    assert len(FEATURE_ORDER) == 16


def forecast_with(model, history, horizon=24):
    static = pd.DataFrame([dict.fromkeys(STATIC_COLS, 0.0)] * len(history))
    start = pd.Series(pd.to_datetime(["2026-10-16 00:00"] * len(history)))
    return recursive_forecast(model, start, static, np.asarray(history, dtype=float), horizon=horizon)


def test_recursive_forecast_feeds_predictions_back(lag_model):
    history = [np.arange(24.0), np.arange(100.0, 124.0)]
    fc = forecast_with(lag_model, history)

    assert len(fc) == 48
    s0 = fc[fc["series"] == 0].sort_values("horizon_h")
    # lag_1 of the first step is the last measured value (23), then each prediction is fed back
    assert s0["predicted_valeur"].tolist() == [24.0 + h for h in range(24)]
    assert s0["forecast_time"].iloc[0] == pd.Timestamp("2026-10-16 01:00")
    assert s0["forecast_time"].iloc[-1] == pd.Timestamp("2026-10-17 00:00")
    assert fc.loc[fc["series"] == 1, "predicted_valeur"].iloc[0] == 124.0


def test_recursive_forecast_lag_24_reads_the_measured_history():
    history = np.arange(24.0)  # value at t0-23 ... t0
    fc = forecast_with(ColumnModel("lag_24"), [history])
    # hour t0+h needs the value 24 hours earlier, i.e. t0+h-24 = history[h-1]
    assert fc.sort_values("horizon_h")["predicted_valeur"].tolist() == history.tolist()


def test_recursive_forecast_rolling_3_is_past_only():
    history = np.arange(24.0)
    fc = forecast_with(ColumnModel("rolling_3"), [history], horizon=1)
    assert fc["predicted_valeur"].iloc[0] == np.mean([21.0, 22.0, 23.0])


def test_recursive_forecast_rejects_bad_history(lag_model):
    import pytest
    with pytest.raises(ValueError):
        forecast_with(lag_model, [np.arange(10.0)])
    with pytest.raises(ValueError):
        forecast_with(lag_model, [[np.nan] + list(range(23))])


def test_add_lag_features_matches_definitions(observations):
    feats = add_lag_features(observations)
    g = feats[feats[config.COL_SITE] == "FR02008"].set_index(config.COL_DATE)[config.COL_VALUE]
    row = feats[(feats[config.COL_SITE] == "FR02008")].iloc[30]
    t = row[config.COL_DATE]
    assert row["lag_1"] == g[t - pd.Timedelta(hours=1)]
    assert row["lag_24"] == g[t - pd.Timedelta(hours=24)]
    expected = np.mean([g[t - pd.Timedelta(hours=k)] for k in (1, 2, 3)])
    assert np.isclose(row["rolling_3"], expected)


def test_add_lag_features_do_not_leak_the_target(observations):
    """Changing the value at hour t must not change any feature of hour t."""
    t = pd.Timestamp("2026-10-11 12:00")
    changed = observations.copy()
    target = (changed[config.COL_DATE] == t) & (changed[config.COL_SITE] == "FR02008")
    changed.loc[target, config.COL_VALUE] += 1000

    def features_at_t(obs):
        f = add_lag_features(obs)
        return f[(f[config.COL_DATE] == t) & (f[config.COL_SITE] == "FR02008")][DYNAMIC_COLS].iloc[0]

    pd.testing.assert_series_equal(features_at_t(observations), features_at_t(changed))


def test_add_lag_features_respects_time_gaps(observations):
    # remove one hour: the next hour must not take an older value as lag_1
    gap = pd.Timestamp("2026-10-11 05:00")
    obs = observations[observations[config.COL_DATE] != gap]
    feats = add_lag_features(obs)
    after = feats[(feats[config.COL_SITE] == "FR02008") & (feats[config.COL_DATE] == gap + pd.Timedelta(hours=1))]
    assert after["lag_1"].isna().all()
    assert not (feats[config.COL_DATE] == gap).any()


def test_history_window_interpolates_short_gaps():
    idx = pd.date_range("2026-10-15 00:00", periods=30, freq="h")
    values = pd.Series(np.arange(30.0), index=idx).drop(idx[[20, 21]])  # 2-hour gap
    window = history_window(values)
    assert len(window) == HISTORY_HOURS
    assert window[-1] == 29.0 and window[0] == 6.0
    assert np.allclose(window, np.arange(6.0, 30.0))  # linear data, so interpolation is exact


def test_history_window_rejects_long_gaps():
    idx = pd.date_range("2026-10-15 00:00", periods=30, freq="h")
    values = pd.Series(np.arange(30.0), index=idx).drop(idx[15:20])  # 5-hour gap
    assert history_window(values) is None
    assert history_window(pd.Series(dtype=float)) is None
