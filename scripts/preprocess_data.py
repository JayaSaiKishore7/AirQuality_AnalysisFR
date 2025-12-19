import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def preprocess_raw_data():

    # Define paths 
    raw_path = os.path.join("data", "raw", "Airquality_PACA_2025_Combined.csv")
    processed_dir = os.path.join("data", "processed")
    encoder_dir = os.path.join("models", "encoders")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(encoder_dir, exist_ok=True)

    processed_path = os.path.join(processed_dir, "df_raw_cleaned.csv")

    print(f"\n Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)

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

    print("✔ Encoders saved.")

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

    #  Create lag & rolling features (per site+pollutant grouped TS) 
    df = df.sort_values(["pollutant_encoded", "site_encoded", "date"])

    df["lag_1"] = df.groupby(["pollutant_encoded", "site_encoded"])["valeur"].shift(1)
    df["lag_24"] = df.groupby(["pollutant_encoded", "site_encoded"])["valeur"].shift(24)

    df["rolling_3"] = (
        df.groupby(["pollutant_encoded", "site_encoded"])["valeur"]
        .rolling(3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Remove missing lag rows 
    df = df.dropna(subset=["lag_1", "lag_24", "rolling_3"]).reset_index(drop=True)

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
    preprocess_raw_data()
