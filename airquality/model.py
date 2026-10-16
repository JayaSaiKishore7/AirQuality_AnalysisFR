"""Model wrapper that puts every pollutant on the same scale.

Concentrations differ by two orders of magnitude between pollutants (CO is
about 0.2 mg/m³, NO2 about 30 µg/m³). Trained on raw values, one model
optimises the large pollutants and practically ignores CO. This wrapper divides
the target and the value-based features (lag_1, lag_24, rolling_3) by the
pollutant's median training level, and multiplies predictions back, so callers
keep passing and receiving raw concentrations.
"""
import numpy as np

from airquality.features import FEATURE_ORDER

POLLUTANT_COL = FEATURE_ORDER.index("pollutant_encoded")
VALUE_COLS = [FEATURE_ORDER.index(c) for c in ("lag_1", "lag_24", "rolling_3")]


def fit_scales(pollutant_codes, y) -> dict[int, float]:
    """Median concentration per pollutant code (from the training split only)."""
    pollutant_codes = np.asarray(pollutant_codes).astype(int)
    y = np.asarray(y, dtype=float)
    scales = {}
    for code in np.unique(pollutant_codes):
        median = float(np.median(y[pollutant_codes == code]))
        scales[int(code)] = median if median > 0 else float(np.mean(np.abs(y[pollutant_codes == code])) or 1.0)
    return scales


class PollutantScaledModel:
    """Wraps a regressor so it learns concentrations relative to each pollutant's usual level."""

    def __init__(self, estimator, scales: dict[int, float]):
        self.estimator = estimator
        self.scales = {int(k): float(v) for k, v in scales.items()}
        self.n_features_in_ = len(FEATURE_ORDER)

    def _row_scales(self, X: np.ndarray) -> np.ndarray:
        codes = X[:, POLLUTANT_COL].astype(int)
        unknown = set(np.unique(codes)) - set(self.scales)
        if unknown:
            raise ValueError(f"No scale for pollutant codes {sorted(unknown)}")
        return np.array([self.scales[c] for c in codes], dtype=float)

    def _scaled(self, X):
        X = np.array(X, dtype=float)  # copy
        s = self._row_scales(X)
        X[:, VALUE_COLS] /= s[:, None]
        return X, s

    def fit(self, X, y):
        Xs, s = self._scaled(X)
        self.estimator.fit(Xs, np.asarray(y, dtype=float) / s)
        return self

    def predict(self, X):
        Xs, s = self._scaled(X)
        return self.estimator.predict(Xs) * s
