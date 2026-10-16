"""Paths and constants shared by the API, dashboard and daily jobs.

Every path can be overridden with an environment variable so tests and
deployments can point at other locations.
"""
import os
from pathlib import Path

ROOT = Path(os.getenv("AQ_ROOT", Path(__file__).resolve().parents[1]))

MODEL_PATH = Path(os.getenv("AQ_MODEL_PATH", ROOT / "models" / "best_model.pkl"))
ENCODER_DIR = Path(os.getenv("AQ_ENCODER_DIR", ROOT / "models" / "encoders"))
PROCESSED_PATH = Path(os.getenv("AQ_PROCESSED_PATH", ROOT / "data" / "processed" / "df_raw_cleaned.csv"))

REFERENCE_DIR = ROOT / "data" / "reference"
STATIONS_PATH = Path(os.getenv("AQ_STATIONS_PATH", REFERENCE_DIR / "stations.csv"))
DRIFT_REFERENCE_PATH = Path(os.getenv("AQ_DRIFT_REFERENCE_PATH", REFERENCE_DIR / "drift_reference.csv"))

LIVE_DIR = Path(os.getenv("AQ_LIVE_DIR", ROOT / "data" / "live"))
OBSERVATIONS_DIR = LIVE_DIR / "observations"
FORECASTS_DIR = LIVE_DIR / "forecasts"
MONITORING_DIR = Path(os.getenv("AQ_MONITORING_DIR", ROOT / "monitoring"))

# Geod'Air real-time hourly files, published by LCSQA/INERIS on data.gouv.fr
# https://www.data.gouv.fr/datasets/5b98b648634f415309d52a50
GEODAIR_BASE_URL = os.getenv(
    "GEODAIR_BASE_URL",
    "https://files.data.gouv.fr/ineris/lcsqa/concentrations-de-polluants-atmospheriques-reglementes/temps-reel",
)
# The model was trained on ATMO SUD (PACA) stations only
REGION_ORGANISME = os.getenv("AQ_ORGANISME", "ATMO SUD")

# Encoder file stems written by scripts/preprocess_data.py
ENCODER_FILES = {
    "pollutant": "pollutant_encoder.pkl",
    "influence": "influence_encoder.pkl",
    "evaluation": "evaluation_encoder.pkl",
    "implantation": "implantation_encoder.pkl",
    "site": "site_encoder.pkl",
}

# Raw column names, as found in Geod'Air exports and the processed dataset
COL_DATE = "date"
COL_VALUE = "valeur"
COL_POLLUTANT = "Polluant"
COL_SITE = "code site"
COL_INFLUENCE = "type d'influence"
COL_EVALUATION = "type d'évaluation"
COL_IMPLANTATION = "type d'implantation"
