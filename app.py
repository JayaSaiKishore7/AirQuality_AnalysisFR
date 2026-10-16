# app.py
import streamlit as st
import pandas as pd
import requests
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys

# Find the project package and data next to this file, whatever folder the app is started from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airquality import config  # noqa: E402
from airquality.artifacts import load_artifacts  # noqa: E402
from airquality.features import history_window  # noqa: E402
from airquality.forecasting import forecast_24h  # noqa: E402


# Config
API_BASE = os.getenv("API_BASE", "http://api:8000")

PROCESSED_PATH = str(config.PROCESSED_PATH)
LIVE_FORECASTS_PATH = str(config.FORECASTS_DIR / "latest.csv")
OBSERVATIONS_DIR = str(config.OBSERVATIONS_DIR)
DRIFT_SUMMARY_PATH = str(config.MONITORING_DIR / "drift_summary.json")
MONITORING_HISTORY_PATH = str(config.MONITORING_DIR / "history.csv")


PALETTE = {
    "bg": "#0f1720",
    "text": "#E6EEF3",
    "muted": "#94A3B8",
    "primary": "#4ECDC4",   
    "accent": "#FF6B6B",    
    "line": "#60A5FA"       
}

st.set_page_config(layout="wide", page_title="Air Quality Forecast Dashboard", page_icon="🌍")


# Helpers / caching
@st.cache_data(ttl=300)
## @st.cache_data(ttl=600)
def check_api_root():
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=600)
# st.cache_data(ttl=1000)
def load_meta_from_api():
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# Only the columns the dashboard uses; repeated text stored as categories.
# Cuts memory from ~310 MB to ~45 MB, which matters on free hosting (~1 GB limit).
PROCESSED_COLUMNS = ["date", "valeur", "Latitude", "Longitude", "lag_1", "lag_24", "rolling_3"]
CATEGORY_COLUMNS = ["Polluant", "code site", "type d'influence", "type d'évaluation", "type d'implantation"]

@st.cache_data(ttl=600)
def load_processed(path=PROCESSED_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=PROCESSED_COLUMNS + CATEGORY_COLUMNS, parse_dates=["date"],
                         dtype={c: "category" for c in CATEGORY_COLUMNS})
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None

@st.cache_data(ttl=600)
def forecastable_sites(_df):
    """(pollutant, site) pairs whose last 24 hours are complete enough to forecast from."""
    return {
        (str(p), str(s))
        for (p, s), g in _df.groupby(["Polluant", "code site"], observed=True)
        if history_window(g.set_index("date")["valeur"]) is not None
    }

@st.cache_data(ttl=600)
def load_live():
    """Daily-job outputs: latest forecasts, recent observations, monitoring summary."""
    fc = pd.read_csv(LIVE_FORECASTS_PATH, parse_dates=["base_time", "forecast_time"], dtype={"code site": str}) \
        if os.path.exists(LIVE_FORECASTS_PATH) else None
    obs = None
    if os.path.isdir(OBSERVATIONS_DIR):
        parts = sorted(f for f in os.listdir(OBSERVATIONS_DIR) if f.endswith(".csv"))[-4:]
        frames = [pd.read_csv(os.path.join(OBSERVATIONS_DIR, f), parse_dates=["date"], dtype={"code site": str})
                  for f in parts]
        frames = [f for f in frames if len(f)]
        obs = pd.concat(frames, ignore_index=True) if frames else None
    summary = None
    if os.path.exists(DRIFT_SUMMARY_PATH):
        with open(DRIFT_SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)
    history = pd.read_csv(MONITORING_HISTORY_PATH) if os.path.exists(MONITORING_HISTORY_PATH) else None
    return fc, obs, summary, history

@st.cache_resource
def load_model_and_encoders():
    return load_artifacts()

def call_forecast(payload, use_api):
    """Use the FastAPI backend when it is running (Docker); otherwise run the model in this app."""
    try:
        if use_api:
            r = requests.post(f"{API_BASE}/forecast/24h", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        model, encoders = load_model_and_encoders()
        return forecast_24h(model, encoders, payload)
    except Exception as e:
        st.error(f"Forecast request failed: {e}")
        return None

def compute_empirical_std(filtered_df, window=168):
    """
    Compute empirical std-dev of recent residuals.
    Residuals are valeur - rolling_3 (if available) or valeur - lag_1.
    Use last `window` rows.
    """
    if filtered_df is None or filtered_df.empty:
        return 0.0
    recent = filtered_df.tail(window)
    if "rolling_3" in recent.columns and recent["rolling_3"].notna().sum() > 10:
        resid = recent["valeur"] - recent["rolling_3"]
    elif "lag_1" in recent.columns and recent["lag_1"].notna().sum() > 10:
        resid = recent["valeur"] - recent["lag_1"]
    else:
        resid = recent["valeur"] - recent["valeur"].mean()
    return float(np.nanstd(resid))


# Session states

if "forecast_data" not in st.session_state:
    st.session_state["forecast_data"] = None
if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None
if "meta_cached" not in st.session_state:
    st.session_state["meta_cached"] = None


st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
        .stMarkdown {{ color: {PALETTE['text']}; }}
        .css-18e3th9 {{ background-color: {PALETTE['bg']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 Air Quality Forecast Dashboard")
st.markdown("Interactive forecasting dashboard — PACA region")

## st.title(" Air Quality Forecast Dashboard")
## st.markdown("Interactive forecasting dashboard — PACA region")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls & Status")
    api_ok = check_api_root()
    if api_ok:
        st.success("FastAPI reachable ✅")
    else:
        st.info("Standalone mode: forecasts run inside this app (no FastAPI backend).")

    if api_ok and st.button("🔁 Test API /meta"):
        meta_test = load_meta_from_api()
        if meta_test:
            st.success("✅ /meta loaded")
            st.json({k: (v[:50] if isinstance(v, list) else v) for k, v in meta_test.items()})
            st.session_state["meta_cached"] = meta_test
        else:
            st.error("Failed to load /meta (server down or response invalid)")

    st.markdown("---")
    st.caption("Select pollutant & site, then generate a 24-hour forecast.")


# Loaddata & metadata

df = load_processed()
if df is None:
    st.error(f"Processed data not found or failed to parse: {PROCESSED_PATH}. "
             "Rebuild it from the monthly Geod'Air files with `python scripts/preprocess_data.py`.")
    st.stop()

meta = st.session_state.get("meta_cached") or load_meta_from_api()
pollutants = meta.get("pollutants", []) if meta else sorted(df["Polluant"].dropna().astype(str).unique())


# Sidebar filters

st.sidebar.subheader("Filters")
selected_pollutant = st.sidebar.selectbox(
    "Pollutant", options=pollutants, index=pollutants.index("NO2") if "NO2" in pollutants else 0)

# Only stations that measure this pollutant; default to one with enough recent data to forecast
forecastable = forecastable_sites(df)
pollutant_sites = sorted(df.loc[df["Polluant"] == selected_pollutant, "code site"].astype(str).unique())
ready = [s for s in pollutant_sites if (selected_pollutant, s) in forecastable]
selected_site = st.sidebar.selectbox(
    "Site (code)", options=pollutant_sites,
    index=pollutant_sites.index(ready[0]) if ready else 0,
    format_func=lambda s: s if (selected_pollutant, s) in forecastable else f"{s} (not enough recent data)")

# Influence, evaluation and implantation describe the station itself, so read them from its data
station_rows = df[(df["Polluant"] == selected_pollutant) & (df["code site"] == selected_site)]
station = station_rows.iloc[-1] if len(station_rows) else None
selected_influence = str(station["type d'influence"]) if station is not None else ""
selected_evaluation = str(station["type d'évaluation"]) if station is not None else ""
selected_implantation = str(station["type d'implantation"]) if station is not None else ""
st.sidebar.caption(f"Station type: {selected_influence} · {selected_evaluation} · {selected_implantation}")

with st.sidebar.expander("📈 Data Info", expanded=True):
    st.write(f"Total records: **{len(df):,}**")
    st.write(f"Date range: **{df['date'].min().date()}** → **{df['date'].max().date()}**")
    st.write(f"Unique pollutants: **{df['Polluant'].nunique() if 'Polluant' in df.columns else 0}**")
    st.write(f"Unique sites: **{df['code site'].nunique() if 'code site' in df.columns else 0}**")


# Tabs
tab_live, tab_data, tab_forecast, tab_about = st.tabs(
    ["🟢 Live", "📊 Data Explorer", "🔮 Forecast + Map", "ℹ️ About"])


# Live tab: outputs of the daily GitHub Actions job

with tab_live:
    st.header("Live forecasts from Geod'Air")
    live_fc, live_obs, drift_summary, monitoring_history = load_live()

    if live_fc is None or live_fc.empty:
        st.info("No live forecasts yet. They are produced daily by the `daily-forecast` workflow.")
    else:
        st.caption(f"Issued {live_fc['issued_at'].iloc[0]} UTC from ATMO SUD real-time measurements")

        l1, l2 = st.columns(2)
        live_pollutants = sorted(live_fc["Polluant"].unique())
        live_pollutant = l1.selectbox("Pollutant", live_pollutants, key="live_pollutant",
                                      index=live_pollutants.index(selected_pollutant)
                                      if selected_pollutant in live_pollutants else 0)
        live_sites = sorted(live_fc.loc[live_fc["Polluant"] == live_pollutant, "code site"].unique())
        live_site = l2.selectbox("Site (code)", live_sites, key="live_site")

        series_fc = live_fc[(live_fc["Polluant"] == live_pollutant) & (live_fc["code site"] == live_site)]
        lfig = go.Figure()
        if live_obs is not None:
            series_obs = live_obs[(live_obs["Polluant"] == live_pollutant) & (live_obs["code site"] == live_site)]
            lfig.add_trace(go.Scatter(x=series_obs["date"], y=series_obs["valeur"], mode="lines",
                                      name="measured", line=dict(color=PALETTE["line"])))
        lfig.add_trace(go.Scatter(x=series_fc["forecast_time"], y=series_fc["predicted_valeur"],
                                  mode="lines+markers", name="forecast", line=dict(color=PALETTE["accent"])))
        lfig.update_layout(title=f"{live_pollutant} @ {live_site} — measured vs next 24h",
                           template="plotly_dark", height=420)
        st.plotly_chart(lfig, use_container_width=True)

    st.subheader("Monitoring")
    if drift_summary is None:
        st.info("No monitoring run yet.")
    else:
        drift = drift_summary.get("drift") or {}
        skill = drift_summary.get("skill") or {}
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Data drift", "⚠️ detected" if drift.get("dataset_drift") else "✅ none")
        d2.metric("Drifted columns", f"{drift.get('n_drifted', '–')}/{drift.get('n_columns', '–')}")
        d3.metric("Forecast MAE", skill.get("mae", "–"))
        d4.metric("Persistence MAE", skill.get("persistence_mae", "–"),
                  help="Error of simply repeating the last measured value. The model should beat this.")
        st.caption(f"Window {drift_summary['window']['start']} → {drift_summary['window']['end']} "
                   "compared with the training data (Evidently).")
        if drift.get("columns"):
            st.dataframe(pd.DataFrame(drift["columns"]).T, use_container_width=True)
        if monitoring_history is not None and len(monitoring_history) > 1:
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(x=monitoring_history["run_date"], y=monitoring_history["drift_share"],
                                      mode="lines+markers", name="drift share", line=dict(color=PALETTE["primary"])))
            hfig.update_layout(title="Share of drifted columns per daily run", template="plotly_dark", height=300)
            st.plotly_chart(hfig, use_container_width=True)


# Data Explorer tab

with tab_data:
    st.header("Historical Data Explorer")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Pollutants", df["Polluant"].nunique() if "Polluant" in df.columns else "N/A")
    c3.metric("Sites", df["code site"].nunique() if "code site" in df.columns else "N/A")
    c4.metric("Avg value", f"{df['valeur'].mean():.2f}")

    if not selected_pollutant or not selected_site:
        st.info("Select pollutant and site in the sidebar.")
    else:
        mask = (df["Polluant"] == selected_pollutant) & (df["code site"] == selected_site)
        filtered = df.loc[mask]
        if filtered.empty:
            st.warning("No historical records for that selection.")
        else:
            recent = filtered.tail(168)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent["date"], y=recent["valeur"], mode="lines+markers",
                                     name="valeur", line=dict(color=PALETTE["line"])))
            if "rolling_3" in recent.columns:
                fig.add_trace(go.Scatter(x=recent["date"], y=recent["rolling_3"], mode="lines",
                                         name="rolling_3", line=dict(color=PALETTE["primary"])))
            fig.update_layout(title=f"{selected_pollutant} at {selected_site} — recent trend",
                              template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show recent data table"):
                st.dataframe(recent[["date", "valeur", "Latitude", "Longitude"]].tail(200), use_container_width=True)


# Forecast + Map tab

with tab_forecast:
    st.header("24-Hour Forecast")

    if not selected_pollutant or not selected_site:
        st.warning("Select pollutant and site in the sidebar.")
    else:
        series = df[(df["Polluant"] == selected_pollutant) & (df["code site"] == selected_site)]
        if series.empty:
            st.warning("No historical records for that selection; showing the latest record overall.")
            latest_row = df.iloc[-1]
            history = None
        else:
            latest_row = series.iloc[-1]
            # last 24 hourly values ending at the latest measurement (short gaps interpolated)
            history = history_window(series.set_index("date")["valeur"])
            if history is None:
                st.warning("The last 24 hours of this series have a gap longer than 3 hours, so no forecast can be made.")

        payload = {
            "datetime": latest_row["date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": float(latest_row["Latitude"]),
            "Longitude": float(latest_row["Longitude"]),
            "pollutant": selected_pollutant,
            "influence": selected_influence,
            "evaluation": selected_evaluation,
            "implantation": selected_implantation,
            "site_code": selected_site,
            "recent_values": [round(float(v), 4) for v in history] if history is not None else None,
        }

        with st.expander("Forecast payload (debug)", expanded=False):
            st.json(payload)

        col_left, col_right = st.columns([3, 1])
        with col_right:
            st.metric("Lat", f"{payload['Latitude']:.4f}")
            st.metric("Lon", f"{payload['Longitude']:.4f}")
            st.metric("Last value", f"{latest_row['valeur']:.2f}")

            # Small map: show station location
            try:
                map_df = pd.DataFrame([{
                    "lat": payload["Latitude"],
                    "lon": payload["Longitude"],
                    "site": selected_site,
                    "pollutant": selected_pollutant
                }])
                map_fig = px.scatter_mapbox(
                    map_df, lat="lat", lon="lon", hover_name="site", hover_data=["pollutant"],
                    zoom=9, height=300
                )
                map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0}, template="plotly_dark")
                st.plotly_chart(map_fig, use_container_width=True)
            except Exception as e:
                st.write("Map could not be rendered:", e)

        # Forecast action
        if st.button("🚀 Generate 24-hour forecast", disabled=history is None):
            with st.spinner("Generating forecast..."):
                results = call_forecast(payload, use_api=api_ok)
                if results:
                    st.session_state["forecast_data"] = results
                    df_fore = pd.DataFrame(results)
                    df_fore["forecast_time"] = pd.to_datetime(df_fore["forecast_time"])

                    # Empirical std from recent residuals
                    mask = (df["Polluant"] == selected_pollutant) & (df["code site"] == selected_site)
                    filtered_hist = df.loc[mask]
                    emp_std = compute_empirical_std(filtered_hist, window=168)
                    # Build confidence band (1 sigma)
                    upper = df_fore["predicted_valeur"] + emp_std
                    lower = df_fore["predicted_valeur"] - emp_std

                    # Plot forecast with confidence band
                    ffig = go.Figure()
                    ffig.add_trace(go.Scatter(
                        x=df_fore["forecast_time"], y=df_fore["predicted_valeur"],
                        mode="lines+markers", name="forecast", line=dict(color=PALETTE["accent"])
                    ))
                    ffig.add_trace(go.Scatter(
                        x=pd.concat([df_fore["forecast_time"], df_fore["forecast_time"][::-1]]),
                        y=pd.concat([upper, lower[::-1]]),
                        fill="toself", fillcolor="rgba(255,107,107,0.15)", line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip", showlegend=True, name="±1σ (empirical)"
                    ))
                    ffig.add_hline(y=latest_row["valeur"], line=dict(color="gray", dash="dash"), annotation_text="Last known")
                    ffig.update_layout(title=f"24h Forecast — {selected_pollutant} @ {selected_site}", template="plotly_dark", height=420)
                    st.plotly_chart(ffig, use_container_width=True)

                    # metrics & table
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Forecast", f"{df_fore['predicted_valeur'].mean():.2f}")
                    m2.metric("Max Forecast", f"{df_fore['predicted_valeur'].max():.2f}")
                    m3.metric("Min Forecast", f"{df_fore['predicted_valeur'].min():.2f}")

                    st.subheader("Forecast details (next 24 hours)")
                    display = df_fore.copy()
                    display["forecast_time"] = display["forecast_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(display, use_container_width=True)

                    csv = df_fore.to_csv(index=False)
                    st.download_button("📥 Download forecast CSV", data=csv,
                                       file_name=f"forecast_{selected_pollutant}_{selected_site}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                       mime="text/csv")
                else:
                    st.error("No forecast returned from API. Check API logs.")

        # previous forecast cache
        if st.session_state.get("forecast_data"):
            st.markdown("### Previous forecast (cached)")
            prev = pd.DataFrame(st.session_state["forecast_data"])
            prev["forecast_time"] = pd.to_datetime(prev["forecast_time"])
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(x=prev["forecast_time"], y=prev["predicted_valeur"], mode="lines+markers", line=dict(color=PALETTE["primary"])))
            pfig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(pfig, use_container_width=True)

# About tab
with tab_about:
    st.header("About this dashboard")
    st.markdown("""
    This interactive dashboard shows historical air quality and a 24-hour forecast produced by a FastAPI backend (XGBoost / RandomForest).
    - Backend: FastAPI
    - Frontend: Streamlit + Plotly
    - Forecast bands: approximate ±1σ computed from recent residuals (empirical)
    """)
    st.markdown("---")
    st.caption("© Air Quality Forecasting | PACA region")

# Footer
st.markdown("---")
st.caption("Made with — Streamlit + FastAPI")
