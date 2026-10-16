"""Static web page (Plotly charts + map) built from the daily job's outputs.

The page is one self-contained HTML file: the data is embedded as JSON and the
charts are drawn in the browser by Plotly.js, so it can be hosted anywhere
static (a Hugging Face static Space) without any Python server.
"""
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from airquality import config

TEMPLATE_PATH = Path(__file__).with_name("site_template.html")
SITE_DIR = config.ROOT / "site"

UNITS = {"µg-m3": "µg/m³", "mg-m3": "mg/m³"}


def _iso(ts: pd.Series) -> list[str]:
    return pd.to_datetime(ts).dt.strftime("%Y-%m-%dT%H:%M").tolist()


def build_site_data(forecasts: pd.DataFrame, observations: pd.DataFrame, stations: pd.DataFrame,
                    summary: dict | None, history: pd.DataFrame | None, obs_hours: int = 72) -> dict:
    """Everything the page draws, as plain JSON-serialisable data."""
    obs = observations.copy()
    if len(obs) and len(forecasts):
        cutoff = pd.to_datetime(forecasts["base_time"]).max() - timedelta(hours=obs_hours)
        obs = obs[pd.to_datetime(obs[config.COL_DATE]) > cutoff]

    coords = stations.set_index(config.COL_SITE)[["Latitude", "Longitude"]]
    has_obs = len(obs) > 0
    names = (obs.dropna(subset=["nom site"]).groupby(config.COL_SITE)["nom site"].last()
             if has_obs and "nom site" in obs else pd.Series(dtype=str))
    units = (obs.groupby(config.COL_POLLUTANT)["unité de mesure"].last()
             if has_obs and "unité de mesure" in obs else pd.Series(dtype=str))
    obs_groups = ({k: g.sort_values(config.COL_DATE) for k, g in obs.groupby([config.COL_SITE, config.COL_POLLUTANT])}
                  if has_obs else {})

    series = []
    for (site, pollutant), fc in forecasts.groupby([config.COL_SITE, config.COL_POLLUTANT]):
        if site not in coords.index:
            continue
        fc = fc.sort_values("horizon_h")
        o = obs_groups.get((site, pollutant))
        obs_t = _iso(o[config.COL_DATE]) if o is not None else []
        obs_v = [round(float(v), 3) for v in o[config.COL_VALUE]] if o is not None else []
        series.append({
            "site": site,
            "name": names.get(site, site),
            "pollutant": pollutant,
            "lat": round(float(coords.at[site, "Latitude"]), 5),
            "lon": round(float(coords.at[site, "Longitude"]), 5),
            "base_time": _iso(fc["base_time"].iloc[:1])[0],
            "last": round(float(fc["last_valeur"].iloc[0]), 3),
            "fc_t": _iso(fc["forecast_time"]),
            "fc_v": [round(float(v), 3) for v in fc["predicted_valeur"]],
            "obs_t": obs_t,
            "obs_v": obs_v,
        })

    pollutants = sorted({s["pollutant"] for s in series})
    hist = []
    if history is not None and len(history):
        h = history.dropna(subset=["drift_share"])
        hist = [{"date": r.run_date, "drift_share": round(float(r.drift_share), 3)} for r in h.itertuples()]

    return {
        "issued_at": str(forecasts["issued_at"].iloc[0]) if len(forecasts) else None,
        "pollutants": pollutants,
        "units": {p: UNITS.get(units.get(p, ""), units.get(p, "")) for p in pollutants},
        "series": series,
        "monitoring": summary,
        "history": hist,
    }


def render_page(data: dict, repo_url: str = "") -> str:
    # "</" is escaped so no value can close the <script> tag the JSON lives in
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (TEMPLATE_PATH.read_text(encoding="utf-8")
            .replace("__SITE_DATA__", payload)
            .replace("__REPO_URL__", repo_url))


def build_site(out_dir: Path = SITE_DIR, repo_url: str = "") -> Path:
    """Read the daily outputs from disk and write out_dir/index.html."""
    from airquality import geodair
    from airquality.artifacts import load_stations

    fc_path = config.FORECASTS_DIR / "latest.csv"
    forecasts = pd.read_csv(fc_path, dtype={config.COL_SITE: str}) if fc_path.exists() else pd.DataFrame(
        columns=["issued_at", config.COL_SITE, config.COL_POLLUTANT, "base_time", "horizon_h",
                 "forecast_time", "predicted_valeur", "last_valeur"])
    since = None
    if len(forecasts):
        since = (pd.to_datetime(forecasts["base_time"]).max() - timedelta(days=4)).date()
    observations = geodair.load_observations(since=since)

    summary_path = config.MONITORING_DIR / "drift_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    history_path = config.MONITORING_DIR / "history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else None

    data = build_site_data(forecasts, observations, load_stations(), summary, history)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(render_page(data, repo_url=repo_url), encoding="utf-8")
    return out
