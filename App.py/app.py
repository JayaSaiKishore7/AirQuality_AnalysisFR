import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Air Quality Forecast Dashboard", layout="wide")

st.title("🌍 Air Quality Analysis & 24-Hour Forecast")

@st.cache_data
def load_data():
    df = pd.read_csv("df_sample_processed.csv", parse_dates=["date"])
    df = df.sort_values("date")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    return model

df = load_data()
model = load_model()

# Sidebar filters
st.sidebar.header("Filters")
pollutants = sorted(df["Polluant_encoded"].unique())
sites = sorted(df["site_encoded"].unique())

selected_pollutant = st.sidebar.selectbox("Pollutant (encoded)", pollutants)
selected_site = st.sidebar.selectbox("Site (encoded)", sites)

# Filter data
df_sel = df[(df["Polluant_encoded"] == selected_pollutant) &
            (df["site_encoded"] == selected_site)].copy()

st.subheader("📈 Historical Data (last 7 days for selection)")
recent = df_sel.tail(24*7)  # last 7 days

fig_hist, ax = plt.subplots(figsize=(10, 4))
ax.plot(recent["date"], recent["valeur"], linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Pollution level")
ax.set_title("Historical Pollution")
st.pyplot(fig_hist)

# --- Forecast next 24h using same function logic ---
feature_cols = [
    "Latitude", "Longitude",
    "hour", "day", "month", "year", "weekday", "is_weekend",
    "Polluant_encoded", "influence_encoded", "evaluation_encoded",
    "implantation_encoded", "site_encoded",
    "lag_1", "lag_24", "rolling_3"
]

def forecast_next_24_for_subset(df_sub, model):
    df_sub = df_sub.sort_values("date")
    last_row = df_sub.iloc[-1:].copy()
    forecasts = []
    times = []

    for _ in range(24):
        X_last = last_row[feature_cols]
        next_val = model.predict(X_last)[0]
        next_time = last_row["date"].iloc[0] + pd.Timedelta(hours=1)

        forecasts.append(next_val)
        times.append(next_time)

        new_row = last_row.copy()
        new_row["date"] = next_time

        new_row["hour"] = next_time.hour
        new_row["day"] = next_time.day
        new_row["month"] = next_time.month
        new_row["year"] = next_time.year
        new_row["weekday"] = next_time.weekday()
        new_row["is_weekend"] = 1 if new_row["weekday"].iloc[0] >= 5 else 0

        prev_val = last_row["valeur"].iloc[0]
        new_row["lag_1"] = prev_val
        new_row["rolling_3"] = (last_row["rolling_3"] * 2 + next_val) / 3
        new_row["valeur"] = next_val

        last_row = new_row.copy()

    return pd.DataFrame({"forecast_time": times, "forecast_value": forecasts})

st.subheader("🔮 Next 24-Hour Forecast")
if len(df_sel) < 24:
    st.warning("Not enough data for this pollutant/site to forecast.")
else:
    forecast_df = forecast_next_24_for_subset(df_sel, model)

    fig_fore, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(recent["date"], recent["valeur"], label="Actual (last 7 days)", linewidth=2)
    ax2.plot(forecast_df["forecast_time"], forecast_df["forecast_value"],
             label="Forecast (next 24h)", linestyle="--", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Pollution level")
    ax2.legend()
    ax2.set_title("Historical vs Forecast")
    st.pyplot(fig_fore)

    st.dataframe(forecast_df)
