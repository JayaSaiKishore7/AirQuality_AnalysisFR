"""Loading of the trained model, label encoders and station coordinates."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from airquality import config

# Encoder name -> raw column it encodes
ENCODED_COLUMNS = {
    "pollutant": config.COL_POLLUTANT,
    "influence": config.COL_INFLUENCE,
    "evaluation": config.COL_EVALUATION,
    "implantation": config.COL_IMPLANTATION,
    "site": config.COL_SITE,
}


def load_model(path: Path = config.MODEL_PATH):
    return joblib.load(path)


def load_encoders(encoder_dir: Path = config.ENCODER_DIR) -> dict:
    missing = [f for f in config.ENCODER_FILES.values() if not (Path(encoder_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing encoders in {encoder_dir}: {missing}")
    return {name: joblib.load(Path(encoder_dir) / f) for name, f in config.ENCODER_FILES.items()}


@lru_cache(maxsize=1)
def load_artifacts():
    """Model and encoders, loaded once per process."""
    return load_model(), load_encoders()


def load_stations(path: Path = config.STATIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={config.COL_SITE: str})


def encode_frame(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Add `<name>_encoded` columns, dropping rows whose category the encoders never saw."""
    df = df.copy()
    known = pd.Series(True, index=df.index)
    for name, col in ENCODED_COLUMNS.items():
        known &= df[col].astype(str).isin(set(encoders[name].classes_))
    df = df[known]
    for name, col in ENCODED_COLUMNS.items():
        df[f"{name}_encoded"] = encoders[name].transform(df[col].astype(str)) if len(df) else []
    return df
