"""Batch 24h forecasts for every station/pollutant with recent live data."""
import pandas as pd

from airquality import config
from airquality.artifacts import encode_frame
from airquality.features import DYNAMIC_COLS, STATIC_COLS, add_lag_features, recursive_forecast, time_features

SERIES_KEYS = [config.COL_SITE, config.COL_POLLUTANT]


def build_live_features(obs: pd.DataFrame, encoders: dict, stations: pd.DataFrame) -> pd.DataFrame:
    """Lag features + encodings + coordinates for live observations the model can score."""
    feats = add_lag_features(obs)
    feats = pd.concat([feats, time_features(feats[config.COL_DATE])], axis=1)
    feats = feats.merge(stations[[config.COL_SITE, "Latitude", "Longitude"]], on=config.COL_SITE, how="inner")
    return encode_frame(feats, encoders)


def latest_states(feats: pd.DataFrame, max_age_hours: int = 48) -> pd.DataFrame:
    """Most recent complete feature row per series, ignoring series that went quiet."""
    complete = feats.dropna(subset=DYNAMIC_COLS)
    if complete.empty:
        return complete
    latest = complete.sort_values(config.COL_DATE).groupby(SERIES_KEYS, sort=False).tail(1)
    cutoff = complete[config.COL_DATE].max() - pd.Timedelta(hours=max_age_hours)
    return latest[latest[config.COL_DATE] >= cutoff].reset_index(drop=True)


def forecast_all(model, states: pd.DataFrame, issued_at: pd.Timestamp) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame(columns=["issued_at", config.COL_SITE, config.COL_POLLUTANT, "base_time",
                                     "horizon_h", "forecast_time", "predicted_valeur", "last_valeur"])
    fc = recursive_forecast(model, states[config.COL_DATE], states[STATIC_COLS], states[DYNAMIC_COLS])
    meta = states[[config.COL_SITE, config.COL_POLLUTANT, config.COL_DATE, config.COL_VALUE]].rename(
        columns={config.COL_DATE: "base_time", config.COL_VALUE: "last_valeur"})
    out = fc.merge(meta, left_on="series", right_index=True).drop(columns="series")
    out.insert(0, "issued_at", issued_at)
    cols = ["issued_at", config.COL_SITE, config.COL_POLLUTANT, "base_time",
            "horizon_h", "forecast_time", "predicted_valeur", "last_valeur"]
    return out[cols].sort_values([config.COL_SITE, config.COL_POLLUTANT, "horizon_h"]).reset_index(drop=True)


def score_past_forecasts(forecasts: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Join archived forecasts with what was actually measured."""
    actual = obs[[config.COL_SITE, config.COL_POLLUTANT, config.COL_DATE, config.COL_VALUE]].rename(
        columns={config.COL_DATE: "forecast_time", config.COL_VALUE: "actual_valeur"})
    actual = actual.drop_duplicates([config.COL_SITE, config.COL_POLLUTANT, "forecast_time"], keep="last")
    fc = forecasts.copy()
    fc["forecast_time"] = pd.to_datetime(fc["forecast_time"])
    return fc.merge(actual, on=[config.COL_SITE, config.COL_POLLUTANT, "forecast_time"], how="inner")
