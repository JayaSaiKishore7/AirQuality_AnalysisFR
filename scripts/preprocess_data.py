

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    # Paths
    raw_path = os.path.join("data", "raw", "Airquality_PACA_2025_Combined.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "df_raw_cleaned.csv")

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    #Basic cleaning: keep only useful columns
    # Make sure these columns exist in your raw file:
    # "Date de début", "valeur", "Latitude", "Longitude",
    # "Polluant", "type d'influence", "type d'évaluation", "type d'implantation", "code site"

    cols_to_keep = [
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
    df = df[cols_to_keep].copy()

    # Drop rows with missing essential values
    df = df.dropna(subset=["Date de début", "valeur", "Latitude", "Longitude"])

    # Create datetime & time features
    df["date"] = pd.to_datetime(df["Date de début"])
    df = df.sort_values("date").reset_index(drop=True)

    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.weekday
    df["weekend"] = df["weekday"].isin([5, 6]).astype(int)

    #Label encoding for categorical columns
    label_enc_cols = ["Polluant", "type d'influence", "type d'évaluation", "type d'implantation", "code site"]

    encoders = {}
    for col in label_enc_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    # Rename site encoder for consistency with notebooks
    df = df.rename(columns={"code site_encoded": "site_encoded",
                            "Polluant_encoded": "pollutant_encoded",
                            "type d'influence_encoded": "influence_encoded",
                            "type d'évaluation_encoded": "evaluation_encoded",
                            "type d'implantation_encoded": "implantation_encoded",
                            })

    #Lag & rolling features (per site & pollutant)
    df = df.sort_values(["pollutant_encoded", "site_encoded", "date"])

    df["lag_1"] = df.groupby(["pollutant_encoded", "site_encoded"])["valeur"].shift(1)
    df["lag_24"] = df.groupby(["pollutant_encoded", "site_encoded"])["valeur"].shift(24)
    df["rolling_3"] = df.groupby(["pollutant_encoded", "site_encoded"])["valeur"].rolling(3).mean().reset_index(level=[0, 1], drop=True)

    # Drop rows where lag/rolling features are NaN (initial rows)
    df = df.dropna(subset=["lag_1", "lag_24", "rolling_3"]).reset_index(drop=True)

    #Keep final columns for modeling
    final_cols = [
        "valeur",
        "Latitude", "Longitude",
        "date",
        "hour", "day", "month", "year",
        "weekday", "weekend",
        "pollutant_encoded",
        "influence_encoded",
        "evaluation_encoded",
        "implantation_encoded",
        "site_encoded",
        "lag_1", "lag_24", "rolling_3",
    ]

    df_final = df[final_cols].copy()

    #Save cleaned data
    df_final.to_csv(processed_path, index=False)
    print(f"Saved cleaned data to: {processed_path}")
    print(f"Shape: {df_final.shape}")


if __name__ == "__main__":
    main()
