import os
import pickle
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airquality.model import POLLUTANT_COL, PollutantScaledModel, fit_scales  # noqa: E402

MAX_MODEL_MB = 50


def mae_by_pollutant(y_true, y_pred, pollutants):
    return (pd.DataFrame({"err": np.abs(y_true - y_pred), "p": pollutants})
            .groupby("p")["err"].mean().to_dict())


def evaluate_model(name, model, X_train, y_train, X_test, y_test, pollutants_test):
    """Fit model and compute metrics (overall and per pollutant)"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    per_pollutant = mae_by_pollutant(y_test, y_pred, pollutants_test)

    print(f"\n{name} Performance:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    print("  MAE per pollutant: " + ", ".join(f"{p} {v:.3f}" for p, v in per_pollutant.items()))

    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2, "mae_by_pollutant": per_pollutant}


def log_metrics(res):
    mlflow.log_metric("mae", res["mae"])
    mlflow.log_metric("rmse", res["rmse"])
    mlflow.log_metric("r2", res["r2"])
    for p, v in res["mae_by_pollutant"].items():
        mlflow.log_metric(f"mae_{p.replace('.', '_')}", v)


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
    pollutants_test = df["Polluant"].values[split_idx:]
    baseline["mae_by_pollutant"] = mae_by_pollutant(y_test, lag_1_test, pollutants_test)
    print("  MAE per pollutant: " + ", ".join(f"{p} {v:.3f}" for p, v in baseline["mae_by_pollutant"].items()))

    # 4. Per-pollutant scales (median level, training split only) so CO counts as much as NO2
    scales = fit_scales(X_train[:, POLLUTANT_COL], y_train)
    print(f"\nPollutant scales (median of training target): {scales}")

    # 5. Define models
    rf = PollutantScaledModel(RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=2,
    ), scales)

    xgb = PollutantScaledModel(XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    ), scales)

    results = []

    # 6. RandomForest with MLflow logging
    with mlflow.start_run(run_name="RandomForest"):
        res_rf = evaluate_model("RandomForest", rf, X_train, y_train, X_test, y_test, pollutants_test)

        mlflow.log_params({"model": "RandomForest", "target_scaling": "per-pollutant median"})
        log_metrics(res_rf)
        mlflow.log_dict(scales, "pollutant_scales.json")
        # skops (MLflow's default) needs tree types trusted explicitly; safe, we just trained this model
        mlflow.sklearn.log_model(res_rf["model"].estimator, name="model",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

        results.append(res_rf)

    # 7. XGBoost with MLflow logging
    with mlflow.start_run(run_name="XGBoost"):
        res_xgb = evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test, pollutants_test)

        mlflow.log_params({"model": "XGBoost", "target_scaling": "per-pollutant median"})
        log_metrics(res_xgb)
        mlflow.log_dict(scales, "pollutant_scales.json")
        mlflow.xgboost.log_model(res_xgb["model"].estimator, name="model")

        results.append(res_xgb)

    # 8. Pick & save the best model that can be deployed. The model file is committed
    # to Git (GitHub rejects files over 100 MB); a deep Random Forest easily exceeds that.
    for r in results:
        r["size_mb"] = len(pickle.dumps(r["model"])) / 1e6
        print(f"{r['name']}: {r['size_mb']:.1f} MB")
    deployable = [r for r in results if r["size_mb"] <= MAX_MODEL_MB]
    if not deployable:
        sys.exit(f"No model is under {MAX_MODEL_MB} MB; nothing saved.")
    best = min(deployable, key=lambda r: r["rmse"])
    print(f"\n Best deployable model: {best['name']} (RMSE = {best['rmse']:.4f}, {best['size_mb']:.1f} MB)")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best["model"], model_path)
    print(f"Saved best model to: {model_path}")

    # 9. Save the comparison (read by notebooks/DeepLearning.ipynb)
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "rf_xgb_results.csv")
    pd.DataFrame(
        [{"Model": r["name"], "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"]} for r in results + [baseline]]
    ).to_csv(report_path, index=False)
    print(f"Saved test-set metrics to: {report_path}")


if __name__ == "__main__":
    main()
