import numpy as np
import pandas as pd
import pytest

from airquality import config
from airquality.artifacts import load_encoders, load_stations


class LagModel:
    """Stand-in model: predicts lag_1 + 1, so recursive updates are easy to check."""

    def predict(self, X):
        return np.asarray(X)[:, 13] + 1.0


@pytest.fixture
def lag_model():
    return LagModel()


@pytest.fixture(scope="session")
def encoders():
    return load_encoders()


@pytest.fixture(scope="session")
def stations():
    return load_stations()


def make_observations(start="2026-10-10 00:00", hours=72, series=(("FR02008", "PM10"), ("FR03043", "NO2"))):
    """Hourly observations with a daily cycle for a few known site/pollutant pairs."""
    times = pd.date_range(start, periods=hours, freq="h")
    rows = []
    for i, (site, pollutant) in enumerate(series):
        values = 20 + 10 * np.sin(np.arange(hours) * 2 * np.pi / 24) + i
        rows.append(pd.DataFrame({
            config.COL_DATE: times,
            config.COL_VALUE: values,
            config.COL_POLLUTANT: pollutant,
            config.COL_SITE: site,
            config.COL_INFLUENCE: "Industrielle",
            config.COL_EVALUATION: "mesures fixes",
            config.COL_IMPLANTATION: "Urbaine",
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def observations():
    return make_observations()
