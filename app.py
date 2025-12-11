# app.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import os

API_URL = "http://127.0.0.1:8000"  # FastAPI base URL
PROCESSED_PATH = os.path.join("data", "processed", "df_raw_cleaned.csv")

st.set_page_config(layout="wide", page_title="Air Quality Forecast Dashboard")

st.title("🌤️ Air Quality Forecast Dashboard (PACA)")

# -------------------------
# Helper: check API
# -------------------------
def api_ok():
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def load_meta():
    try:
        r = requests.get(f"{API_URL}/meta", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to load metadata from API: {e}")
        return None

# -------------------------
# Data loading (local processed file)
# -------------------------
@st.cache_data(ttl=600)
def load_processed_df(path=PROCESSED_PATH):
    if not os.path.exists(path):
        st.error(f"Processed data file not found: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df

df = load_processed_df()
if df is None:
    st.stop()

# -------------------------
# Check API connection and meta
# -------------------------
if api_ok():
    st.sidebar.success("API Connected")
    meta = load_meta()
    if meta is None:
        st.sidebar.error("Failed to load meta from API")
        st.stop()
else:
    st.sidebar.error("API Not Connected (expected at http://127.0.0.1:8000)")
    st.info("Start the FastAPI server (python Api/main.py) then refresh this page.")
    st.stop()

# -------------------------
# Sidebar: filters
# -------------------------
st.sidebar.header("Filters & Settings")
pollutants = meta.get("pollutants", []) or sorted(df["Polluant"].unique().tolist())
sites_sample = meta.get("sites_sample", []) or sorted(df["code site"].unique().tolist())

pollutant_sel = st.sidebar.selectbox("Pollutant", pollutants, index=0)
site_sel = st.sidebar.selectbox("Site (code)", sites_sample, index=0)

influence_choices = meta.get("influences", [])
evaluation_choices = meta.get("evaluations", [])
implantation_choices = meta.get("implantations", [])

influence_sel = st.sidebar.selectbox("Influence", influence_choices, index=0)
evaluation_sel = st.sidebar.selectbox("Evaluation", evaluation_choices, index=0)
implantation_sel = st.sidebar.selectbox("Implantation", implantation_choices, index=0)

# Filter history by pollutant/site
history = df[(df["Polluant"] == pollutant_sel) & (df["code site"] == site_sel)].sort_values("date")
if history.empty:
    st.warning("No historical records for this pollutant/site selection.")
    st.stop()

last_row = history.iloc[-1]

st.subheader(f"Historical pollution for {pollutant_sel} @ {site_sel}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"].tail(168), y=history["valeur"].tail(168),
                         mode="lines+markers", name="Actual"))
if "rolling_3" in history.columns:
    fig.add_trace(go.Scatter(x=history["date"].tail(168), y=history["rolling_3"].tail(168),
                             mode="lines", name="Rolling-3"))
fig.update_layout(height=350, xaxis_title="Date", yaxis_title="valeur")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Build forecast payload and call API
# -------------------------
st.subheader("🔮 Forecast (next 24 hours)")
payload = {
    "datetime": last_row["date"].strftime("%Y-%m-%d %H:%M:%S"),
    "Latitude": float(last_row["Latitude"]),
    "Longitude": float(last_row["Longitude"]),
    "pollutant": pollutant_sel,
    "influence": influence_sel,
    "evaluation": evaluation_sel,
    "implantation": implantation_sel,
    "site_code": site_sel,
    "lag_1": float(last_row.get("lag_1", 0.0)),
    "lag_24": float(last_row.get("lag_24", 0.0)),
    "rolling_3": float(last_row.get("rolling_3", 0.0)),
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
            st.success("Forecast received!")

            # Plot history + forecast
            fig2 = go.Figure()
            hist_last = history.tail(48)
            fig2.add_trace(go.Scatter(x=hist_last["date"], y=hist_last["valeur"], mode="lines+markers", name="History"))
            fig2.add_trace(go.Scatter(x=df_fore["forecast_time"], y=df_fore["predicted_valeur"], mode="lines+markers", name="Forecast"))
            fig2.update_layout(height=400, xaxis_title="Time", yaxis_title="valeur")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Forecast Table")
            st.dataframe(df_fore)
    except Exception as e:
        st.error(f"Failed to call API: {e}")

# -------------------------
# Footer / debug
# -------------------------
with st.expander("Debug"):
    st.write("API URL:", API_URL)
    st.write("Last row sample:", last_row.to_dict())
