import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

st.set_page_config(page_title="Air Quality Forecast Dashboard", layout="wide")

st.title("🌍 Air Quality Analysis & 24-Hour Forecast")

# -----------------------------
# 1. Load data & model
# -----------------------------


@st.cache_data
def load_data():
    # Adjust path if needed
    df = pd.read_csv("data/processed/df_sample_processed.csv", parse_dates=["date"])
    df = df.sort_values("date")
    return df


@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    return model


@st.cache_resource
def load_mappings():
    """Load optional mappings for pollutant/site labels.
    If file not found, we fall back to generic names like 'Pollutant 2'."""
    try:
        mapping = joblib.load("models/mappings.pkl")
    except Exception:
        mapping = {}
    pollutant_map = mapping.get("pollutant_map", {})
    site_map = mapping.get("site_map", {})
    return pollutant_map, site_map


df = load_data()
model = load_model()
pollutant_map, site_map = load_mappings()

# -----------------------------
# Helpers
# -----------------------------


def classify_level(value: float):
    """Simple AQI-like category (generic, not official)."""
    if value <= 25:
        return "Low", "🟢"
    elif value <= 50:
        return "Moderate", "🟡"
    elif value <= 75:
        return "High", "🟠"
    else:
        return "Very High", "🔴"


feature_cols = [
    "Latitude",
    "Longitude",
    "hour",
    "day",
    "month",
    "year",
    "weekday",
    "is_weekend",
    "Polluant_encoded",
    "influence_encoded",
    "evaluation_encoded",
    "implantation_encoded",
    "site_encoded",
    "lag_1",
    "lag_24",
    "rolling_3",
]


def forecast_next_24(df_sub: pd.DataFrame, model):
    """Multi-step 24h forecast starting from the last row of df_sub."""
    df_sub = df_sub.sort_values("date")
    last_row = df_sub.iloc[-1:].copy()

    forecasts = []
    times = []

    for _ in range(24):
        X_last = last_row[feature_cols]
        next_val = model.predict(X_last)[0]

        next_time = last_row["date"].iloc[0] + pd.Timedelta(hours=1)
        times.append(next_time)
        forecasts.append(next_val)

        new_row = last_row.copy()
        new_row["date"] = next_time

        # update time features
        new_row["hour"] = next_time.hour
        new_row["day"] = next_time.day
        new_row["month"] = next_time.month
        new_row["year"] = next_time.year
        new_row["weekday"] = next_time.weekday()
        new_row["is_weekend"] = 1 if new_row["weekday"].iloc[0] >= 5 else 0

        # update lag/rolling features
        prev_val = last_row["valeur"].iloc[0]
        new_row["lag_1"] = prev_val
        new_row["rolling_3"] = (last_row["rolling_3"] * 2 + next_val) / 3

        # we approximate lag_24 by keeping previous lag_24 (simple approach)
        new_row["valeur"] = next_val

        last_row = new_row.copy()

    return pd.DataFrame({"forecast_time": times, "forecast_value": forecasts})


# -----------------------------
# 2. Sidebar filters (with nicer labels)
# -----------------------------

st.sidebar.header("Filters")

# Pollutant selection
pollutant_codes = sorted(df["Polluant_encoded"].unique())
pollutant_labels = [pollutant_map.get(c, f"Pollutant {c}") for c in pollutant_codes]
poll_label_to_code = dict(zip(pollutant_labels, pollutant_codes))

selected_pollutant_label = st.sidebar.selectbox("Pollutant", pollutant_labels)
selected_pollutant = poll_label_to_code[selected_pollutant_label]

# Site selection
site_codes = sorted(df["site_encoded"].unique())
site_labels = [site_map.get(s, f"Site {s}") for s in site_codes]
site_label_to_code = dict(zip(site_labels, site_codes))

selected_site_label = st.sidebar.selectbox("Site", site_labels)
selected_site = site_label_to_code[selected_site_label]

df_sel = df[
    (df["Polluant_encoded"] == selected_pollutant)
    & (df["site_encoded"] == selected_site)
].copy()

if df_sel.empty:
    st.warning("No data for this pollutant / site combination.")
    st.stop()

df_sel = df_sel.sort_values("date")
recent = df_sel.tail(24 * 7)  # last 7 days


# -----------------------------
# 3. Tabs layout
# -----------------------------

tab_overview, tab_forecast, tab_map, tab_trends = st.tabs(
    ["📊 Overview", "🔮 Forecast", "🗺️ Map", "📈 Trends"]
)

# -----------------------------
# OVERVIEW TAB
# -----------------------------
with tab_overview:
    st.subheader("Overview – Current Status")

    last_actual = df_sel["valeur"].iloc[-1]
    last_time = df_sel["date"].iloc[-1]
    avg_7d = recent["valeur"].mean()

    level_now, icon_now = classify_level(last_actual)
    level_avg, icon_avg = classify_level(avg_7d)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Latest Measurement",
        f"{last_actual:.2f}",
        help=f"At {last_time}",
    )
    col2.metric(
        "Current Level",
        f"{icon_now} {level_now}",
        help="Based on latest measurement",
    )
    col3.metric(
        "7-Day Avg Level",
        f"{icon_avg} {level_avg}",
        help=f"Mean of last 7 days: {avg_7d:.2f}",
    )

    st.markdown(
        f"**Pollutant:** {selected_pollutant_label}  |  **Site:** {selected_site_label}"
    )

    st.markdown("### Historical Data (last 7 days)")
    fig_hist, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent["date"], recent["valeur"], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Pollution level")
    ax.set_title("Historical Pollution (last 7 days)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig_hist.autofmt_xdate(rotation=45)
    fig_hist.tight_layout()
    st.pyplot(fig_hist)


# -----------------------------
# FORECAST TAB
# -----------------------------
with tab_forecast:
    st.subheader("Next 24-Hour Forecast")

    if len(df_sel) < 30:
        st.warning("Not enough data for this pollutant/site to generate a forecast.")
    else:
        forecast_df = forecast_next_24(df_sel, model)

        avg_forecast = forecast_df["forecast_value"].mean()
        level_forecast, icon_forecast = classify_level(avg_forecast)

        c1, c2 = st.columns(2)
        c1.metric("Mean Forecast (next 24h)", f"{avg_forecast:.2f}")
        c2.metric("Forecast Level", f"{icon_forecast} {level_forecast}")

        fig_fore, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(recent["date"], recent["valeur"], label="Actual (last 7 days)", linewidth=2)
        ax2.plot(
            forecast_df["forecast_time"],
            forecast_df["forecast_value"],
            label="Forecast (next 24h)",
            linestyle="--",
            linewidth=2,
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Pollution level")
        ax2.set_title("Historical vs Forecast")
        ax2.legend()
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig_fore.autofmt_xdate(rotation=45)
        fig_fore.tight_layout()
        st.pyplot(fig_fore)

        st.markdown("### Forecast Table (Next 24 Hours)")
        st.dataframe(forecast_df)

        csv_data = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download forecast as CSV",
            csv_data,
            file_name="forecast_next_24h.csv",
            mime="text/csv",
        )


# -----------------------------
# MAP TAB
# -----------------------------
with tab_map:
    st.subheader("Station Location")

    map_df = df_sel[["Latitude", "Longitude"]].dropna().copy()
    map_df = map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})

    if map_df.empty:
        st.warning("No latitude/longitude data available for this station.")
    else:
        # show last known position
        st.map(map_df.tail(1))

    st.caption("Map shows the location of the selected measuring station.")


# -----------------------------
# TRENDS TAB
# -----------------------------
with tab_trends:
    st.subheader("Trends & Aggregations")

    # Daily and weekly mean series
    daily = df_sel.set_index("date")["valeur"].resample("D").mean().dropna()
    weekly = df_sel.set_index("date")["valeur"].resample("W").mean().dropna()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Daily Mean Pollution**")
        fig_daily, axd = plt.subplots(figsize=(6, 3))
        axd.plot(daily.index, daily.values, marker="o")
        axd.set_xlabel("Date")
        axd.set_ylabel("Mean pollution")
        axd.set_title("Daily Mean")

        axd.xaxis.set_major_locator(mdates.AutoDateLocator())
        axd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig_daily.autofmt_xdate(rotation=45)

        fig_daily.tight_layout()
        st.pyplot(fig_daily)

    with col_b:
        st.markdown("**Weekly Mean Pollution**")
        fig_weekly, axw = plt.subplots(figsize=(6, 3))
        axw.plot(weekly.index, weekly.values, marker="o")
        axw.set_xlabel("Week")
        axw.set_ylabel("Mean pollution")
        axw.set_title("Weekly Mean")

        axw.xaxis.set_major_locator(mdates.AutoDateLocator())
        axw.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig_weekly.autofmt_xdate(rotation=45)

        fig_weekly.tight_layout()
        st.pyplot(fig_weekly)

    st.markdown(
        "You can use these trends to see if the air quality is improving or worsening over time."
    )
