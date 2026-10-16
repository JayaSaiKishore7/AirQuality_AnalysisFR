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
# hourly values needed before the first forecast hour (lag_24 reaches back 24 hours)
HISTORY_HOURS = 24


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


def recursive_forecast(model, start: pd.Series, static: pd.DataFrame, history: np.ndarray,
                       horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Forecast `horizon` hours ahead for N series at once.

    `start` is the time of the last measurement per series and `history` an (N, 24)
    array of the last 24 hourly values ending at `start`, oldest first. For every
    step the lag features are read from that history, extended with the model's own
    predictions, using exactly the definitions of `add_lag_features`.

    Returns a long frame: series index, horizon_h, forecast_time, predicted_valeur.
    """
    start = pd.to_datetime(pd.Series(start)).reset_index(drop=True)
    static = static[STATIC_COLS].reset_index(drop=True)
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] < HISTORY_HOURS:
        raise ValueError(f"history must have shape (n_series, {HISTORY_HOURS})")
    if np.isnan(history).any():
        raise ValueError("history contains missing values")

    n_hist = history.shape[1]
    buf = np.full((len(static), n_hist + horizon), np.nan)
    buf[:, :n_hist] = history
    out = []
    for h in range(1, horizon + 1):
        pos = n_hist - 1 + h  # column of the hour being predicted
        t = start + pd.Timedelta(hours=h)
        dyn = pd.DataFrame({
            "lag_1": buf[:, pos - 1],
            "lag_24": buf[:, pos - 24],
            "rolling_3": buf[:, pos - 3:pos].mean(axis=1),
        })
        X = pd.concat([static, time_features(t), dyn], axis=1)[FEATURE_ORDER]
        # concentrations cannot be negative; clip before feeding predictions back
        y = np.clip(model.predict(X.to_numpy(dtype=float)).astype(float), 0, None)
        buf[:, pos] = y
        out.append(pd.DataFrame({
            "series": np.arange(len(static)),
            "horizon_h": h,
            "forecast_time": t.values,
            "predicted_valeur": y,
        }))
    return pd.concat(out, ignore_index=True)


def history_window(values: pd.Series, hours: int = HISTORY_HOURS, max_gap_hours: int = 3) -> np.ndarray | None:
    """The last `hours` hourly values ending at the latest measurement, oldest first.

    `values` is indexed by timestamp. Gaps of up to `max_gap_hours` are filled by
    linear interpolation; returns None when the window has a longer gap.
    """
    v = values.dropna().sort_index()
    v = v[~v.index.duplicated(keep="last")]
    if v.empty:
        return None
    end = v.index.max()
    # look a little further back so a gap right at the start can be interpolated too
    grid = pd.date_range(end - pd.Timedelta(hours=hours - 1 + max_gap_hours), end, freq="h")
    window = v.reindex(grid).interpolate(limit=max_gap_hours, limit_area="inside").iloc[-hours:]
    return None if window.isna().any() else window.to_numpy(dtype=float)


def add_lag_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Compute lag_1, lag_24 and rolling_3 per (site, pollutant) on an hourly time grid.

    All three use only hours *before* the target hour t, so they are available when
    forecasting: lag_1 = value at t-1, lag_24 = value at t-24,
    rolling_3 = mean of t-3, t-2 and t-1. Timestamps (not row positions) are used,
    so a missing hour yields NaN instead of silently pulling an older value.
    """
    keys = [config.COL_SITE, config.COL_POLLUTANT]
    frames = []
    for _, g in obs.sort_values(config.COL_DATE).groupby(keys, sort=False):
        g = g.drop_duplicates(config.COL_DATE, keep="last").set_index(config.COL_DATE)
        v = g[config.COL_VALUE].asfreq("h")
        g = g.reindex(v.index)
        past = v.shift(1)
        g["lag_1"] = past
        g["lag_24"] = v.shift(24)
        g["rolling_3"] = past.rolling(3, min_periods=3).mean()
        frames.append(g.dropna(subset=[config.COL_VALUE]).reset_index(names=config.COL_DATE))
    if not frames:
        return obs.assign(lag_1=np.nan, lag_24=np.nan, rolling_3=np.nan).iloc[0:0]
    return pd.concat(frames, ignore_index=True)
