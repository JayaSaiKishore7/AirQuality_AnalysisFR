"""Data and prediction drift between the training data and recent live data (Evidently)."""
from dataclasses import asdict, dataclass, field

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from airquality import config
from airquality.features import FEATURE_ORDER

NUMERIC_COLS = [config.COL_VALUE, "lag_1", "lag_24", "rolling_3", "hour", "weekday", "prediction"]
CATEGORICAL_COLS = [config.COL_POLLUTANT, config.COL_SITE]
DRIFT_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Dataset-level drift is flagged when at least this share of columns drifted
DRIFT_SHARE = 0.5


@dataclass
class DriftResult:
    dataset_drift: bool
    drift_share: float
    n_drifted: int
    n_columns: int
    reference_rows: int
    current_rows: int
    columns: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def with_predictions(model, feats: pd.DataFrame) -> pd.DataFrame:
    """Score feature rows (one-step-ahead, as in training) and keep the drift columns."""
    feats = feats.dropna(subset=FEATURE_ORDER)
    out = feats.copy()
    out["prediction"] = model.predict(feats[FEATURE_ORDER].to_numpy(dtype=float)) if len(feats) else []
    return out[DRIFT_COLS].reset_index(drop=True)


def _column_drifted(method: str, value: float, threshold: float) -> bool:
    # p-value tests drift when below the threshold; distances when above it
    return value < threshold if "p_value" in method else value >= threshold


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, html_path=None) -> DriftResult:
    definition = DataDefinition(numerical_columns=NUMERIC_COLS, categorical_columns=CATEGORICAL_COLS)
    ref = reference[DRIFT_COLS].astype({c: str for c in CATEGORICAL_COLS})
    cur = current[DRIFT_COLS].astype({c: str for c in CATEGORICAL_COLS})

    snapshot = Report([DataDriftPreset(drift_share=DRIFT_SHARE)]).run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )
    if html_path:
        snapshot.save_html(str(html_path))

    columns = {}
    share = 0.0
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("ValueDrift"):
            columns[cfg["column"]] = {
                "method": cfg["method"],
                "score": float(m["value"]),
                "threshold": float(cfg["threshold"]),
                "drifted": _column_drifted(cfg["method"], float(m["value"]), float(cfg["threshold"])),
            }
        elif cfg["type"].endswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])

    n_drifted = sum(c["drifted"] for c in columns.values())
    return DriftResult(
        dataset_drift=share >= DRIFT_SHARE,
        drift_share=share,
        n_drifted=n_drifted,
        n_columns=len(columns),
        reference_rows=len(ref),
        current_rows=len(cur),
        columns=columns,
    )
