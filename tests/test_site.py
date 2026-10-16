import json
import re

import pandas as pd

from airquality import config
from airquality.forecasting import build_live_features, forecast_all, latest_states
from airquality.site import build_site_data, render_page


def page_data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def make_inputs(observations, encoders, stations, lag_model):
    obs = observations.assign(**{"nom site": "Port de Bouc", "unité de mesure": "µg-m3"})
    states = latest_states(build_live_features(obs, encoders, stations))
    fc = forecast_all(lag_model, states, pd.Timestamp("2026-10-13 06:00"))
    return fc, obs


def test_site_data_has_one_entry_per_series(observations, encoders, stations, lag_model):
    fc, obs = make_inputs(observations, encoders, stations, lag_model)
    data = build_site_data(fc, obs, stations, summary=None, history=None)

    assert data["pollutants"] == ["NO2", "PM10"]
    assert data["units"]["PM10"] == "µg/m³"
    assert len(data["series"]) == 2
    s = next(x for x in data["series"] if x["pollutant"] == "PM10")
    assert len(s["fc_t"]) == len(s["fc_v"]) == 24
    assert s["name"] == "Port de Bouc"
    # only the last 72h of measurements are embedded
    assert len(s["obs_t"]) == 72 and s["obs_t"][-1] <= s["fc_t"][0]


def test_render_page_embeds_valid_json(observations, encoders, stations, lag_model):
    fc, obs = make_inputs(observations, encoders, stations, lag_model)
    summary = {"window": {"start": "2026-10-06", "end": "2026-10-13"}, "skill": None,
               "drift": {"dataset_drift": True, "n_drifted": 6, "n_columns": 9, "columns": {}}}
    history = pd.DataFrame({"run_date": ["2026-10-12", "2026-10-13"], "drift_share": [0.4, 0.67]})
    html = render_page(build_site_data(fc, obs, stations, summary, history), repo_url="https://github.com/x/y")

    data = page_data(html)
    assert data["monitoring"]["drift"]["n_drifted"] == 6
    assert data["history"][-1] == {"date": "2026-10-13", "drift_share": 0.67}
    assert "__SITE_DATA__" not in html and "__REPO_URL__" not in html
    assert "https://github.com/x/y" in html


def test_render_page_cannot_be_broken_by_data():
    data = {"issued_at": None, "pollutants": [], "units": {}, "series": [], "monitoring": None, "history": [],
            "note": "</script><script>alert(1)</script>"}
    html = render_page(data)
    assert "</script><script>alert(1)" not in html
    assert page_data(html)["note"] == "</script><script>alert(1)</script>"


def test_empty_forecasts_render(stations):
    empty = pd.DataFrame(columns=["issued_at", config.COL_SITE, config.COL_POLLUTANT, "base_time",
                                  "horizon_h", "forecast_time", "predicted_valeur", "last_valeur"])
    data = build_site_data(empty, pd.DataFrame(columns=[config.COL_DATE]), stations, None, None)
    assert data["series"] == [] and data["issued_at"] is None
    assert page_data(render_page(data))["series"] == []
