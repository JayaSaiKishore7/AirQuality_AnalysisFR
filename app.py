# app.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os
from datetime import datetime

API_URL = "http://127.0.0.1:8000"
PROCESSED_PATH = os.path.join("data", "processed", "df_raw_cleaned.csv")

st.set_page_config(layout="wide", page_title="Air Quality Forecast Dashboard")

st.title("Air Quality Forecast Dashboard")

# --- helpers
def api_ok():
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=300)
def load_processed(path=PROCESSED_PATH):
    if not os.path.exists(path):
        st.error(f"Processed file not found: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df

df = load_processed()
if df is None:
    st.stop()

if not api_ok():
    st.sidebar.error("API Not Connected. Start FastAPI at http://127.0.0.1:8000")
    st.info("Start API (python Api/main.py) then refresh this page.")
    st.stop()

# load meta
try:
    meta = requests.get(f"{API_URL}/meta", timeout=5).json()
except Exception as e:
    st.error(f"Failed to load metadata: {e}")
    st.stop()

pollutants = meta.get("pollutants") or sorted(df["Polluant"].unique().tolist())
sites = meta.get("sites_sample") or sorted(df["code site"].unique().tolist())

pollutant = st.sidebar.selectbox("Pollutant", pollutants, index=0)
site = st.sidebar.selectbox("Site", sites, index=0)
influence = st.sidebar.selectbox("Influence", meta.get("influences", []), index=0)
evaluation = st.sidebar.selectbox("Evaluation", meta.get("evaluations", []), index=0)
implantation = st.sidebar.selectbox("Implantation", meta.get("implantations", []), index=0)

history = df[(df["Polluant"] == pollutant) & (df["code site"] == site)].sort_values("date")
if history.empty:
    st.warning("No history for this pollutant/site. Try another selection.")
    st.stop()

last = history.iloc[-1]

st.subheader(f"History: {pollutant} @ {site}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"].tail(168), y=history["valeur"].tail(168),
                         mode="lines+markers", name="valeur"))
if "rolling_3" in history.columns:
    fig.add_trace(go.Scatter(x=history["date"].tail(168), y=history["rolling_3"].tail(168),
                             mode="lines", name="rolling_3"))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

# Prepare payload
payload = {
    "datetime": last["date"].strftime("%Y-%m-%d %H:%M:%S"),
    "Latitude": float(last["Latitude"]),
    "Longitude": float(last["Longitude"]),
    "pollutant": pollutant,
    "influence": influence,
    "evaluation": evaluation,
    "implantation": implantation,
    "site_code": site,
    "lag_1": float(last.get("lag_1", 0.0)),
    "lag_24": float(last.get("lag_24", 0.0)),
    "rolling_3": float(last.get("rolling_3", 0.0)),
}

if st.button("Generate 24-hour forecast"):
    try:
        r = requests.post(f"{API_URL}/forecast/24h", json=payload, timeout=30)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            forecast = r.json()
            df_fore = pd.DataFrame(forecast)
            df_fore["forecast_time"] = pd.to_datetime(df_fore["forecast_time"])
            st.success("Forecast received")
            fig2 = go.Figure()
            hist = history.tail(48)
            fig2.add_trace(go.Scatter(x=hist["date"], y=hist["valeur"], mode="lines+markers", name="history"))
            fig2.add_trace(go.Scatter(x=df_fore["forecast_time"], y=df_fore["predicted_valeur"], mode="lines+markers", name="forecast"))
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df_fore)
    except Exception as e:
        st.error(f"Failed to call API: {e}")

with st.expander("Debug / Info"):
    st.write("API_URL:", API_URL)
    st.write("Last row sample:", last.to_dict())
