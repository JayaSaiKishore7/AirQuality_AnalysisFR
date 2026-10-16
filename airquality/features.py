"""Feature engineering shared by training-time data, the API and the daily jobs."""
import numpy as np
import pandas as pd

from airquality import config

# Column order the model was trained on (scripts/train_model.py)
FEATURE_ORDER = [
    "Latitude", "Longitude",
    "hour", "day", "month", "year",
    "weekday", "weekend",
    "pollutant_encoded",
    "influence_encoded",
    "evaluation_encoded",
    "implantation_encoded",
    "site_encoded",
    "lag_1", "lag_24", "rolling_3",
]

STATIC_COLS = [
    "Latitude", "Longitude",
    "pollutant_encoded", "influence_encoded", "evaluation_encoded",
    "implantation_encoded", "site_encoded",
]
DYNAMIC_COLS = ["lag_1", "lag_24", "rolling_3"]

HORIZON_HOURS = 24


def time_features(ts: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(ts)
    weekday = ts.dt.weekday
    return pd.DataFrame({
        "hour": ts.dt.hour,
        "day": ts.dt.day,
        "month": ts.dt.month,
        "year": ts.dt.year,
        "weekday": weekday,
        "weekend": (weekday >= 5).astype(int),
    }, index=ts.index)


def recursive_forecast(model, start: pd.Series, static: pd.DataFrame, dynamic: pd.DataFrame,
                       horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Forecast `horizon` hours ahead for N series at once.

    `start`, `static` and `dynamic` share the same index (one row per series).
    After each step lag_1 becomes the prediction and rolling_3 is updated with it;
    lag_24 stays fixed, matching the original API behaviour.

    Returns a long frame: series index, horizon_h, forecast_time, predicted_valeur.
    """
    start = pd.to_datetime(start)
    dyn = dynamic[DYNAMIC_COLS].astype(float).copy()
    out = []
    for h in range(1, horizon + 1):
        t = start + pd.Timedelta(hours=h)
        X = pd.concat([static[STATIC_COLS], time_features(t), dyn], axis=1)[FEATURE_ORDER]
        y = model.predict(X.to_numpy(dtype=float)).astype(float)
        out.append(pd.DataFrame({
            "series": static.index,
            "horizon_h": h,
            "forecast_time": t.values,
            "predicted_valeur": y,
        }))
        dyn["lag_1"] = y
        dyn["rolling_3"] = (dyn["rolling_3"].to_numpy() * 2 + y) / 3
    return pd.concat(out, ignore_index=True)


def add_lag_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Compute lag_1, lag_24 and rolling_3 per (site, pollutant) on an hourly time grid.

    Unlike the training script (row-based shifts), this uses timestamps, so a
    missing hour yields NaN instead of silently pulling an older value.
    `rolling_3` is the mean of the current hour and the two before it, as in training.
    """
    keys = [config.COL_SITE, config.COL_POLLUTANT]
    frames = []
    for _, g in obs.sort_values(config.COL_DATE).groupby(keys, sort=False):
        g = g.drop_duplicates(config.COL_DATE, keep="last").set_index(config.COL_DATE)
        v = g[config.COL_VALUE].asfreq("h")
        g = g.reindex(v.index)
        g["lag_1"] = v.shift(1)
        g["lag_24"] = v.shift(24)
        g["rolling_3"] = v.rolling(3, min_periods=3).mean()
        frames.append(g.dropna(subset=[config.COL_VALUE]).reset_index(names=config.COL_DATE))
    if not frames:
        return obs.assign(lag_1=np.nan, lag_24=np.nan, rolling_3=np.nan).iloc[0:0]
    return pd.concat(frames, ignore_index=True)
