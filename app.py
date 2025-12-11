import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Configuration
API_BASE = "http://127.0.0.1:8000"
DATA_PATH = "data/processed/df_raw_cleaned.csv"

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'last_pollutant' not in st.session_state:
    st.session_state.last_pollutant = None
if 'last_site' not in st.session_state:
    st.session_state.last_site = None

# Test API connection
@st.cache_data(ttl=300)
def test_api():
    try:
        response = requests.get(f"{API_BASE}/", timeout=3)
        return response.status_code == 200
    except:
        return False

# Load data
@st.cache_data(ttl=600)
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'], nrows=10000)  # Load only first 10k rows for speed
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Get metadata from API
def get_metadata():
    try:
        response = requests.get(f"{API_BASE}/meta", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

# Main app
st.title("🌤️ Air Quality Forecast Dashboard")

# Sidebar
st.sidebar.header("Settings")

# API Status
api_ok = test_api()
if api_ok:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Not Connected")
    st.sidebar.info("Make sure FastAPI is running on port 8000")

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df is None:
    st.error("Could not load data file")
    st.stop()

# Get available pollutants and sites from data
if 'Polluant' in df.columns:
    pollutants = df['Polluant'].unique().tolist()
else:
    pollutants = []

if 'code site' in df.columns:
    sites = df['code site'].unique().tolist()[:50]  # First 50
else:
    sites = []

# Filters
if pollutants:
    selected_pollutant = st.sidebar.selectbox("Pollutant", pollutants, index=0)
else:
    selected_pollutant = None

if sites:
    selected_site = st.sidebar.selectbox("Site", sites, index=0)
else:
    selected_site = None

# Data info
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Data Info:**
- Total rows: {len(df):,}
- Pollutants: {len(pollutants)}
- Sites: {len(sites)}
""")

# Main content
if selected_pollutant and selected_site:
    # Filter data
    filtered_data = df[
        (df['Polluant'] == selected_pollutant) & 
        (df['code site'] == selected_site)
    ]
    
    if not filtered_data.empty:
        # Show latest data
        latest = filtered_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Value", f"{latest['valeur']:.2f}")
        with col2:
            if 'lag_1' in latest:
                st.metric("1h Ago", f"{latest['lag_1']:.2f}")
        with col3:
            if 'lag_24' in latest:
                st.metric("24h Ago", f"{latest['lag_24']:.2f}")
        with col4:
            st.metric("Location", f"{latest['Latitude']:.2f}, {latest['Longitude']:.2f}")
        
        # Plot recent data
        st.subheader(f"Recent Data: {selected_pollutant} at {selected_site}")
        
        recent = filtered_data.tail(72)  # Last 72 hours
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(recent['date'], recent['valeur'], 'b-', linewidth=2, label='Actual')
        
        if 'rolling_3' in recent.columns:
            ax.plot(recent['date'], recent['rolling_3'], 'r--', linewidth=1, label='3h Avg')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Forecast section
        st.subheader("24-Hour Forecast")
        
        if not api_ok:
            st.warning("API is not connected. Cannot generate forecast.")
        else:
            # Prepare forecast request
            forecast_payload = {
                "datetime": latest['date'].strftime("%Y-%m-%d %H:%M:%S"),
                "Latitude": float(latest['Latitude']),
                "Longitude": float(latest['Longitude']),
                "pollutant": selected_pollutant,
                "influence": "Trafic routier",  # Default
                "evaluation": "Réglementaire",  # Default
                "implantation": "URBAIN",  # Default
                "site_code": selected_site,
                "lag_1": float(latest.get('lag_1', 0)),
                "lag_24": float(latest.get('lag_24', 0)),
                "rolling_3": float(latest.get('rolling_3', 0)),
            }
            
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Calling forecast API..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/forecast/24h",
                            json=forecast_payload,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            forecast_data = response.json()
                            st.session_state.forecast_data = forecast_data
                            st.session_state.last_pollutant = selected_pollutant
                            st.session_state.last_site = selected_site
                        else:
                            st.error(f"API Error: {response.status_code}")
                            st.code(response.text[:200])
                    except Exception as e:
                        st.error(f"Request failed: {e}")
            
            # Show previous forecast if available
            if st.session_state.forecast_data and \
               st.session_state.last_pollutant == selected_pollutant and \
               st.session_state.last_site == selected_site:
                
                forecast_df = pd.DataFrame(st.session_state.forecast_data)
                forecast_df['forecast_time'] = pd.to_datetime(forecast_df['forecast_time'])
                
                # Plot forecast
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                
                # Plot actual data
                ax2.plot(recent['date'], recent['valeur'], 'b-', linewidth=1, label='Actual', alpha=0.7)
                
                # Plot forecast
                ax2.plot(forecast_df['forecast_time'], forecast_df['predicted_valeur'], 
                        'r-', linewidth=2, marker='o', markersize=4, label='Forecast')
                
                # Connect last actual to first forecast
                last_actual_time = recent['date'].iloc[-1]
                first_forecast_time = forecast_df['forecast_time'].iloc[0]
                ax2.plot([last_actual_time, first_forecast_time],
                        [latest['valeur'], forecast_df['predicted_valeur'].iloc[0]],
                        'r--', alpha=0.5)
                
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Value')
                ax2.set_title(f'24-Hour Forecast for {selected_pollutant}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Format x-axis
                ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig2)
                
                # Show forecast table
                with st.expander("Show Forecast Data"):
                    display_df = forecast_df.copy()
                    display_df['forecast_time'] = display_df['forecast_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(display_df)
    else:
        st.warning(f"No data found for {selected_pollutant} at {selected_site}")
else:
    st.info("👈 Please select a pollutant and site from the sidebar")

# API Test section
with st.expander("Test API Connection"):
    if st.button("Test API Now"):
        with st.spinner("Testing..."):
            api_ok = test_api()
            if api_ok:
                st.success("✅ API is responding")
                
                # Try to get metadata
                meta = get_metadata()
                if meta:
                    st.success("✅ Metadata loaded successfully")
                    st.json(meta)
                else:
                    st.error("❌ Could not load metadata")
            else:
                st.error("❌ API is not responding")

# Footer
st.markdown("---")
st.caption("Dashboard | FastAPI Backend | Streamlit Frontend")