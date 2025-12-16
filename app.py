# app.py
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# Config
API_BASE = "http://127.0.0.1:8000"
PROCESSED_PATH = "data/processed/df_raw_cleaned.csv"


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

@st.cache_data(ttl=600)
def load_processed(path=PROCESSED_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None

def call_forecast(payload):
    try:
        r = requests.post(f"{API_BASE}/forecast/24h", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
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


# Session state

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls & Status")
    api_ok = check_api_root()
    if api_ok:
        st.success("FastAPI reachable ✅")
    else:
        st.error("FastAPI not reachable ❌")
        st.caption("Start FastAPI: `uvicorn Api.main:app --host 127.0.0.1 --port 8000`")

    if st.button("🔁 Test API /meta"):
        meta_test = load_meta_from_api()
        if meta_test:
            st.success("✅ /meta loaded")
            st.json({k: (v[:50] if isinstance(v, list) else v) for k, v in meta_test.items()})
            st.session_state["meta_cached"] = meta_test
        else:
            st.error("Failed to load /meta (server down or response invalid)")

    st.markdown("---")
    st.caption("Select pollutant & site, then generate a 24-hour forecast.")


# Load data & metadata

df = load_processed()
if df is None:
    st.error(f"Processed data not found or failed to parse: {PROCESSED_PATH}")
    st.stop()

meta = st.session_state.get("meta_cached") or load_meta_from_api()
if meta is None:
    pollutants = sorted(df["Polluant"].dropna().unique().tolist()) if "Polluant" in df.columns else []
    sites = sorted(df["code site"].dropna().unique().tolist())[:500] if "code site" in df.columns else []
    influences = sorted(df["type d'influence"].dropna().unique().tolist()) if "type d'influence" in df.columns else []
    evaluations = sorted(df["type d'évaluation"].dropna().unique().tolist()) if "type d'évaluation" in df.columns else []
    implantations = sorted(df["type d'implantation"].dropna().unique().tolist()) if "type d'implantation" in df.columns else []
else:
    pollutants = meta.get("pollutants", [])
    sites = meta.get("sites_sample", [])
    influences = meta.get("influences", [])
    evaluations = meta.get("evaluations", [])
    implantations = meta.get("implantations", [])


# Sidebar filters

st.sidebar.subheader("Filters")
selected_pollutant = st.sidebar.selectbox("Pollutant", options=pollutants, index=0 if pollutants else None)
selected_site = st.sidebar.selectbox("Site (code)", options=sites[:200], index=0 if sites else None)
selected_influence = st.sidebar.selectbox("Influence", options=influences or ["Trafic routier"], index=0)
selected_evaluation = st.sidebar.selectbox("Evaluation", options=evaluations or ["Réglementaire"], index=0)
selected_implantation = st.sidebar.selectbox("Implantation", options=implantations or ["URBAIN"], index=0)
## selected_implantation = st.sidebar.selectbox("Implantation", options=implantations or ["URBAIN"], index=1)

with st.sidebar.expander("📈 Data Info", expanded=True):
    st.write(f"Total records: **{len(df):,}**")
    st.write(f"Date range: **{df['date'].min().date()}** → **{df['date'].max().date()}**")
    st.write(f"Unique pollutants: **{df['Polluant'].nunique() if 'Polluant' in df.columns else 0}**")
    st.write(f"Unique sites: **{df['code site'].nunique() if 'code site' in df.columns else 0}**")


# Tabs
tab_data, tab_forecast, tab_about = st.tabs(["📊 Data Explorer", "🔮 Forecast + Map", "ℹ️ About"])


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

    if not api_ok:
        st.error("FastAPI is not reachable. Start the API and try again.")
    elif not selected_pollutant or not selected_site:
        st.warning("Select pollutant and site in the sidebar.")
    else:
        # safe latest row
        try:
            latest_row = df[(df["Polluant"] == selected_pollutant) & (df["code site"] == selected_site)].iloc[-1]
        except Exception:
            latest_row = df.iloc[-1]

        payload = {
            "datetime": latest_row["date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": float(latest_row["Latitude"]),
            "Longitude": float(latest_row["Longitude"]),
            "pollutant": selected_pollutant,
            "influence": selected_influence,
            "evaluation": selected_evaluation,
            "implantation": selected_implantation,
            "site_code": selected_site,
            "lag_1": float(latest_row.get("lag_1", 0.0)),
            "lag_24": float(latest_row.get("lag_24", 0.0)),
            "rolling_3": float(latest_row.get("rolling_3", 0.0)),
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
        if st.button("🚀 Generate 24-hour forecast"):
            with st.spinner("Contacting API and generating forecast..."):
                results = call_forecast(payload)
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
