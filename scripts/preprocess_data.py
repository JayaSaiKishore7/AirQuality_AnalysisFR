import argparse
import glob
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airquality.features import add_lag_features  # noqa: E402


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


def preprocess_raw_data(processed_path=os.path.join("data", "processed", "df_raw_cleaned.csv"),
                        encoder_dir=os.path.join("models", "encoders")):

    # Define paths
    raw_dir = os.path.join("data", "raw")
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
    
    ## df = df[selected_cols].copy()

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

    # Label encoding for categorical features 
    categorical_cols = [
        "Polluant",
        "type d'influence",
        "type d'évaluation",
        "type d'implantation",
        "code site",
    ]

    # map raw column names -> clean encoder file stem
    encoder_name_map = {
        "Polluant": "pollutant",
        "type d'influence": "influence",
        "type d'évaluation": "evaluation",
        "type d'implantation": "implantation",
        "code site": "site",
    }

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

        clean_name = encoder_name_map[col]  # e.g. "pollutant"
        encoder_path = os.path.join(encoder_dir, f"{clean_name}_encoder.pkl")
        joblib.dump(le, encoder_path)

    print("Encoders saved.")

    # Rename encoded columns for consistency with modeling
    df = df.rename(
        columns={
            "Polluant_encoded": "pollutant_encoded",
            "type d'influence_encoded": "influence_encoded",
            "type d'évaluation_encoded": "evaluation_encoded",
            "type d'implantation_encoded": "implantation_encoded",
            "code site_encoded": "site_encoded",
        }
    )

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
    parser.add_argument("--output", default=os.path.join("data", "processed", "df_raw_cleaned.csv"))
    parser.add_argument("--encoder-dir", default=os.path.join("models", "encoders"))
    args = parser.parse_args()
    preprocess_raw_data(args.output, args.encoder_dir)
