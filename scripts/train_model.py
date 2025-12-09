import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Fit model and compute metrics"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Performance:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2}


def main():

    # Enable MLflow experiment grouping
    mlflow.set_experiment("AirQuality_Forecasting")

    # 1. Load cleaned data
    data_path = os.path.join("data", "processed", "df_raw_cleaned.csv")
    print(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Define target & features
    target_col = "valeur"
    feature_cols = [
        "Latitude", "Longitude",
        "hour", "day", "month", "year",
        "weekday", "weekend",
        "pollutant_encoded",
        "influence_encoded",
        "evaluation_encoded",
        "implantation_encoded",
        "site_encoded",
        "lag_1", "lag_24", "rolling_3",
    ]

    X = df[feature_cols].values
    y = df[target_col].values

    # 3. Train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # 4. Define models
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    results = []

    # 5. RandomForest with MLflow logging
    with mlflow.start_run(run_name="RandomForest"):
        res_rf = evaluate_model("RandomForest", rf, X_train, y_train, X_test, y_test)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("mae", res_rf["mae"])
        mlflow.log_metric("rmse", res_rf["rmse"])
        mlflow.log_metric("r2", res_rf["r2"])
        mlflow.sklearn.log_model(res_rf["model"], artifact_path="model")

        results.append(res_rf)

    # 6. XGBoost with MLflow logging
    with mlflow.start_run(run_name="XGBoost"):
        res_xgb = evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test)

        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("mae", res_xgb["mae"])
        mlflow.log_metric("rmse", res_xgb["rmse"])
        mlflow.log_metric("r2", res_xgb["r2"])
        mlflow.sklearn.log_model(res_xgb["model"], artifact_path="model")

        results.append(res_xgb)

    # 7. Pick & save best model
    best = min(results, key=lambda r: r["rmse"])
    best_model = best["model"]
    print(f"\n Best model: {best['name']} (RMSE = {best['rmse']:.4f})")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
