import numpy as np
import pytest

from airquality.features import FEATURE_ORDER
from airquality.model import POLLUTANT_COL, VALUE_COLS, PollutantScaledModel, fit_scales

LAG_1 = FEATURE_ORDER.index("lag_1")


class Recorder:
    """Estimator that remembers what it was given and predicts lag_1."""

    def fit(self, X, y):
        self.X_fit, self.y_fit = X, y
        return self

    def predict(self, X):
        self.X_pred = X
        return X[:, LAG_1]


def rows(pollutant_codes, lag_1):
    X = np.ones((len(pollutant_codes), len(FEATURE_ORDER)))
    X[:, POLLUTANT_COL] = pollutant_codes
    X[:, VALUE_COLS] = np.asarray(lag_1, dtype=float)[:, None]
    return X


def test_fit_scales_uses_median_per_pollutant():
    scales = fit_scales([0, 0, 0, 1, 1], [0.1, 0.2, 0.9, 30.0, 50.0])
    assert scales == {0: 0.2, 1: 40.0}


def test_inner_model_sees_relative_values_and_predictions_return_to_units():
    scales = {0: 0.2, 1: 40.0}  # CO-like and NO2-like
    model = PollutantScaledModel(Recorder(), scales).fit(rows([0, 1], [0.4, 80.0]), [0.2, 40.0])

    # both rows are "2x normal" for their pollutant, so the inner model sees the same number
    assert model.estimator.X_fit[:, LAG_1].tolist() == [2.0, 2.0]
    assert model.estimator.y_fit.tolist() == [1.0, 1.0]
    # other features are untouched
    assert (np.delete(model.estimator.X_fit, VALUE_COLS, axis=1) == np.delete(rows([0, 1], [0, 0]), VALUE_COLS, axis=1)).all()

    # Recorder predicts the scaled lag_1 (2.0); the wrapper converts back to units
    assert model.predict(rows([0, 1], [0.4, 80.0])).tolist() == pytest.approx([0.4, 80.0])


def test_predict_does_not_modify_the_callers_array():
    X = rows([0], [0.4])
    PollutantScaledModel(Recorder(), {0: 0.2}).predict(X)
    assert X[0, LAG_1] == 0.4


def test_unknown_pollutant_fails_loudly():
    with pytest.raises(ValueError, match="No scale"):
        PollutantScaledModel(Recorder(), {0: 0.2}).predict(rows([7], [1.0]))
