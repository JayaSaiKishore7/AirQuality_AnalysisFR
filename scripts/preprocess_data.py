import argparse
import glob
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airquality import config  # noqa: E402
from airquality.artifacts import ENCODED_COLUMNS  # noqa: E402
from airquality.features import add_lag_features  # noqa: E402

RAW_DIR = config.ROOT / "data" / "raw"


def build_combined_raw(raw_dir, combined_path):
    """Concatenate the monthly Geod'Air exports (data/raw/*_Analysis.csv, semicolon-separated).

    The combined file is too large for GitHub, so a fresh clone rebuilds it from the monthly files.
    """
    monthly = sorted(glob.glob(os.path.join(raw_dir, "*_Analysis.csv")))
    if not monthly:
        raise FileNotFoundError(f"No monthly *_Analysis.csv files in {raw_dir}")
    print(f"\n Building {combined_path} from {len(monthly)} monthly files")
    df = pd.concat([pd.read_csv(f, sep=";") for f in monthly], ignore_index=True)
    df.to_csv(combined_path, index=False)
    return df


def preprocess_raw_data(processed_path=config.PROCESSED_PATH, encoder_dir=config.ENCODER_DIR):

    # Define paths (relative to the project, whatever folder the script is run from)
    raw_dir = str(RAW_DIR)
    raw_path = os.path.join(raw_dir, "Airquality_PACA_2025_Combined.csv")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(encoder_dir, exist_ok=True)

    if os.path.exists(raw_path):
        print(f"\n Loading raw dataset from: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        df = build_combined_raw(raw_dir, raw_path)

    #  Keep relevant columns (must exist in raw file) 
    selected_cols = [
        "Date de début",
        "valeur",
        "Latitude",
        "Longitude",
        "Polluant",
        "type d'influence",
        "type d'évaluation",
        "type d'implantation",
        "code site",
    ]

    df = df[selected_cols].copy()

    #  Drop unusable rows 
    df = df.dropna(subset=["Date de début", "valeur", "Latitude", "Longitude"])

    #  Convert date & extract features 
    df["date"] = pd.to_datetime(df["Date de début"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df = df.sort_values("date")

    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.weekday
    df["weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Label encoding for categorical features: <name>_encoded columns and one encoder
    # file per category, with the same names the API and daily jobs load
    for name, col in ENCODED_COLUMNS.items():
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[f"{name}_encoded"] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(encoder_dir, config.ENCODER_FILES[name]))

    print("Encoders saved.")

    # Lag & rolling features per site+pollutant series. Shared with the live pipeline,
    # and past-only: rolling_3 for hour t is the mean of t-3..t-1, never including t.
    df = add_lag_features(df)
    df = df.sort_values(["pollutant_encoded", "site_encoded", "date"])

    # Remove missing lag rows
    df = df.dropna(subset=["lag_1", "lag_24", "rolling_3"]).reset_index(drop=True)

    # Filling the hourly grid made integer columns float; restore them (keeps the CSV small)
    int_cols = ["hour", "day", "month", "year", "weekday", "weekend", "pollutant_encoded",
                "influence_encoded", "evaluation_encoded", "implantation_encoded", "site_encoded"]
    df[int_cols] = df[int_cols].astype(int)

    #  Final modeling features 
    final_cols = [
        "valeur",
        "Latitude",
        "Longitude",
        "date",
        "hour",
        "day",
        "month",
        "year",
        "weekday",
        "weekend",
        "pollutant_encoded",
        "influence_encoded",
        "evaluation_encoded",
        "implantation_encoded",
        "site_encoded",
        "Polluant",
        "type d'influence",
        "type d'évaluation",
        "type d'implantation",
        "code site",
        "lag_1",
        "lag_24",
        "rolling_3",
    ]

    df_final = df[final_cols].copy()

    # Save dataset 
    df_final.to_csv(processed_path, index=False)
    print(f"\n Saved cleaned dataset → {processed_path}")
    print(" Final shape:", df_final.shape)

    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the raw Geod'Air data and build model features.")
    parser.add_argument("--output", default=str(config.PROCESSED_PATH))
    parser.add_argument("--encoder-dir", default=str(config.ENCODER_DIR))
    args = parser.parse_args()
    preprocess_raw_data(args.output, args.encoder_dir)
