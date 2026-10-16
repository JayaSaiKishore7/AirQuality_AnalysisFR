import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost


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

    # MLflow 3 no longer writes to the ./mlruns file store; use a local SQLite database
    # unless MLFLOW_TRACKING_URI points elsewhere. View with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

    # Baseline to beat: persistence ("next hour = last measured hour", i.e. lag_1)
    lag_1_test = df["lag_1"].values[split_idx:]
    baseline = {
        "name": "Persistence",
        "mae": mean_absolute_error(y_test, lag_1_test),
        "rmse": np.sqrt(mean_squared_error(y_test, lag_1_test)),
        "r2": r2_score(y_test, lag_1_test),
    }
    print(f"\nPersistence baseline: MAE = {baseline['mae']:.4f}, RMSE = {baseline['rmse']:.4f}, "
          f"R² = {baseline['r2']:.4f}")

    # 4. Define models
    # rf = RandomForestRegressor(
    #     n_estimators=300,
    #     random_state=42,
    #     n_jobs=-1
    # )

    rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=2
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
        # skops (MLflow's default) needs tree types trusted explicitly; safe, we just trained this model
        mlflow.sklearn.log_model(res_rf["model"], name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])


        results.append(res_rf)

    # 6. XGBoost with MLflow logging
    with mlflow.start_run(run_name="XGBoost"):
        res_xgb = evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test)

        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("mae", res_xgb["mae"])
        mlflow.log_metric("rmse", res_xgb["rmse"])
        mlflow.log_metric("r2", res_xgb["r2"])
        mlflow.xgboost.log_model(res_xgb["model"], name="model")


        results.append(res_xgb)

    # 7. Pick & save best model
    best = min(results, key=lambda r: r["rmse"])
    best_model = best["model"]
    print(f"\n Best model: {best['name']} (RMSE = {best['rmse']:.4f})")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"Saved best model to: {model_path} ({size_mb:.1f} MB)")
    if size_mb > 50:
        print("WARNING: the model is larger than 50 MB; GitHub rejects files over 100 MB.")

    # 8. Save the comparison (read by notebooks/DeepLearning.ipynb)
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "rf_xgb_results.csv")
    pd.DataFrame(
        [{"Model": r["name"], "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"]} for r in results + [baseline]]
    ).to_csv(report_path, index=False)
    print(f"Saved test-set metrics to: {report_path}")


if __name__ == "__main__":
    main()
