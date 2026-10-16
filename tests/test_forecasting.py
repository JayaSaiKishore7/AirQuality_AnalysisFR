import pandas as pd

from airquality import config
from airquality.forecasting import build_live_features, forecast_all, latest_states, score_past_forecasts
from tests.conftest import make_observations


def test_unknown_sites_are_dropped(encoders, stations):
    obs = make_observations(series=(("FR02008", "PM10"), ("FR99999", "PM10"), ("FR02008", "SO2")))
    feats = build_live_features(obs, encoders, stations)
    assert set(zip(feats[config.COL_SITE], feats[config.COL_POLLUTANT])) == {("FR02008", "PM10")}
    assert {"pollutant_encoded", "site_encoded", "Latitude", "hour"} <= set(feats.columns)


def test_forecast_all_end_to_end(observations, encoders, stations, lag_model):
    feats = build_live_features(observations, encoders, stations)
    states = latest_states(feats)
    assert len(states) == 2
    assert (states[config.COL_DATE] == observations[config.COL_DATE].max()).all()

    fc = forecast_all(lag_model, states, pd.Timestamp("2026-10-13 06:00"))
    assert len(fc) == 48
    assert fc.groupby([config.COL_SITE, config.COL_POLLUTANT])["horizon_h"].max().eq(24).all()
    first = fc[(fc[config.COL_SITE] == "FR02008") & (fc["horizon_h"] == 1)].iloc[0]
    assert first["forecast_time"] == first["base_time"] + pd.Timedelta(hours=1)


def test_latest_states_skips_stale_series(encoders, stations):
    fresh = make_observations(start="2026-10-10", hours=96, series=(("FR02008", "PM10"),))
    stale = make_observations(start="2026-10-05", hours=48, series=(("FR03043", "NO2"),))
    feats = build_live_features(pd.concat([fresh, stale]), encoders, stations)
    assert latest_states(feats)[config.COL_SITE].tolist() == ["FR02008"]


def test_forecast_all_empty_states(lag_model):
    fc = forecast_all(lag_model, pd.DataFrame(), pd.Timestamp("2026-10-16"))
    assert fc.empty and "predicted_valeur" in fc.columns


def test_score_past_forecasts(observations):
    t = observations[config.COL_DATE].iloc[10]
    fc = pd.DataFrame({
        config.COL_SITE: ["FR02008", "FR02008"],
        config.COL_POLLUTANT: ["PM10", "PM10"],
        "forecast_time": [t.isoformat(), "2030-01-01T00:00:00"],  # second has no measurement yet
        "predicted_valeur": [1.0, 2.0],
        "last_valeur": [0.0, 0.0],
    })
    scored = score_past_forecasts(fc, observations)
    assert len(scored) == 1
    assert scored["actual_valeur"].iloc[0] == observations[config.COL_VALUE].iloc[10]
