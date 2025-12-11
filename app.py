import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ==================== CONFIGURATION ====================
API_URL = "http://127.0.0.1:8000"
DATA_FILE = "data/processed/df_raw_cleaned.csv"

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Air Quality Forecast",
    page_icon="🌍",
    layout="wide"
)

# ==================== HELPER FUNCTIONS ====================
def test_api():
    """Test API connection"""
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200
    except:
        return False

def load_data_sample():
    """Load sample data"""
    try:
        # Load only essential columns
        df = pd.read_csv(
            DATA_FILE,
            parse_dates=['date'],
            usecols=['date', 'valeur', 'Latitude', 'Longitude', 
                    'Polluant', 'code site', 'lag_1', 'lag_24', 'rolling_3'],
            nrows=20000  # First 20k rows
        )
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# ==================== INITIALIZE ====================
st.title("🌤️ Air Quality Forecast Dashboard")
st.markdown("Real-time pollution forecasting for PACA region")

# Sidebar header
st.sidebar.header("🔧 Configuration")

# API Status
api_connected = test_api()
if api_connected:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Not Connected")
    st.sidebar.info("""
    **To connect:**
    1. Open a terminal
    2. Run: `python Api/main.py`
    3. Wait for server to start
    4. Refresh this page
    """)

# Load data
with st.spinner("📊 Loading data..."):
    df = load_data_sample()

if df is None:
    st.error("Failed to load data file")
    st.stop()

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("🔍 Filters")

# Get unique values
pollutants = sorted(df['Polluant'].unique().tolist()) if 'Polluant' in df.columns else []
sites = sorted(df['code site'].unique().tolist())[:100] if 'code site' in df.columns else []

if not pollutants or not sites:
    st.error("No pollutants or sites found in data")
    st.stop()

# Select boxes
selected_pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    pollutants,
    index=0
)

selected_site = st.sidebar.selectbox(
    "Select Site",
    sites,
    index=0
)

# ==================== MAIN DASHBOARD ====================
if selected_pollutant and selected_site:
    # Filter data
    filtered_data = df[
        (df['Polluant'] == selected_pollutant) & 
        (df['code site'] == selected_site)
    ].copy()
    
    if not filtered_data.empty:
        # Get latest data
        latest = filtered_data.iloc[-1]
        
        # Display metrics
        st.header(f"📈 {selected_pollutant} at {selected_site}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Value", f"{latest['valeur']:.2f}")
        with col2:
            st.metric("Latitude", f"{latest['Latitude']:.4f}")
        with col3:
            st.metric("Longitude", f"{latest['Longitude']:.4f}")
        with col4:
            st.metric("Last Update", latest['date'].strftime("%H:%M"))
        
        # ==================== TIME SERIES PLOT ====================
        st.subheader("📊 Historical Data")
        
        # Get last 3 days
        recent_data = filtered_data.tail(72)
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['valeur'],
            mode='lines+markers',
            name='Pollution Level',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add rolling average
        if 'rolling_3' in recent_data.columns:
            fig.add_trace(go.Scatter(
                x=recent_data['date'],
                y=recent_data['rolling_3'],
                mode='lines',
                name='3h Average',
                line=dict(color='orange', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{selected_pollutant} Concentration - Last 3 Days",
            xaxis_title="Time",
            yaxis_title="Concentration",
            hovermode="x unified",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==================== FORECAST SECTION ====================
        st.subheader("🔮 24-Hour Forecast")
        
        if api_connected:
            # Prepare forecast request
            forecast_payload = {
                "datetime": latest['date'].strftime("%Y-%m-%d %H:%M:%S"),
                "Latitude": float(latest['Latitude']),
                "Longitude": float(latest['Longitude']),
                "pollutant": selected_pollutant,
                "influence": "Trafic routier",
                "evaluation": "Réglementaire",
                "implantation": "URBAIN",
                "site_code": selected_site,
                "lag_1": float(latest.get('lag_1', 0)),
                "lag_24": float(latest.get('lag_24', 0)),
                "rolling_3": float(latest.get('rolling_3', 0)),
            }
            
            # Generate forecast button
            if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("Generating 24-hour forecast..."):
                    try:
                        # Call API
                        response = requests.post(
                            f"{API_URL}/forecast/24h",
                            json=forecast_payload,
                            timeout=20
                        )
                        
                        if response.status_code == 200:
                            forecast_data = response.json()
                            
                            # Convert to DataFrame
                            forecast_df = pd.DataFrame(forecast_data)
                            forecast_df['forecast_time'] = pd.to_datetime(forecast_df['forecast_time'])
                            
                            # Display success
                            st.success(f"✅ Forecast generated! ({len(forecast_df)} hours)")
                            
                            # ==================== FORECAST PLOT ====================
                            # Combine historical and forecast data
                            historical_last_24h = filtered_data.tail(24)
                            
                            fig2 = go.Figure()
                            
                            # Historical data
                            fig2.add_trace(go.Scatter(
                                x=historical_last_24h['date'],
                                y=historical_last_24h['valeur'],
                                mode='lines+markers',
                                name='Historical (24h)',
                                line=dict(color='blue', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Forecast data
                            fig2.add_trace(go.Scatter(
                                x=forecast_df['forecast_time'],
                                y=forecast_df['predicted_valeur'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', width=2),
                                marker=dict(size=6, symbol='circle')
                            ))
                            
                            fig2.update_layout(
                                title="24-Hour Forecast",
                                xaxis_title="Time",
                                yaxis_title="Predicted Value",
                                hovermode="x unified",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # ==================== FORECAST STATISTICS ====================
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average", f"{forecast_df['predicted_valeur'].mean():.2f}")
                            with col2:
                                st.metric("Maximum", f"{forecast_df['predicted_valeur'].max():.2f}")
                            with col3:
                                st.metric("Minimum", f"{forecast_df['predicted_valeur'].min():.2f}")
                            with col4:
                                st.metric("Change", 
                                         f"{(forecast_df['predicted_valeur'].iloc[-1] - latest['valeur']):.2f}",
                                         delta=f"{(forecast_df['predicted_valeur'].iloc[-1] - latest['valeur']):.2f}")
                            
                            # ==================== FORECAST TABLE ====================
                            with st.expander("📋 View Forecast Details"):
                                display_df = forecast_df.copy()
                                display_df['forecast_time'] = display_df['forecast_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Download button
                                csv = forecast_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Forecast as CSV",
                                    data=csv,
                                    file_name=f"forecast_{selected_pollutant}_{selected_site}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                        
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.code(response.text[:500])
                            
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. Try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ API is not connected. Connect to FastAPI to generate forecasts.")
            
        # ==================== DATA TABLE ====================
        with st.expander("📄 View Raw Data"):
            st.dataframe(
                filtered_data[['date', 'valeur', 'Latitude', 'Longitude']].tail(20),
                use_container_width=True
            )
    else:
        st.warning(f"No data found for {selected_pollutant} at {selected_site}")
else:
    st.info("👈 Please select a pollutant and site from the sidebar")

# ==================== DEBUG SECTION ====================
with st.sidebar.expander("🛠️ Debug Tools"):
    if st.button("Test API Connection"):
        if test_api():
            st.success("✅ API is responding")
        else:
            st.error("❌ API is not responding")
    
    if st.button("Load Metadata"):
        try:
            response = requests.get(f"{API_URL}/meta", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Metadata loaded")
                st.write(f"Pollutants: {len(data.get('pollutants', []))}")
                st.write(f"Sites: {len(data.get('sites_sample', []))}")
            else:
                st.error(f"Failed: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.write(f"Data: {len(df)} rows")
    st.write(f"Pollutants: {len(pollutants)}")
    st.write(f"Sites: {len(sites)}")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Air Quality Forecast Dashboard | PACA Region | Made with ❤️ using Streamlit & FastAPI")