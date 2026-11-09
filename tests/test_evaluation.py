"""Tests for the evaluation protocol and baselines, on synthetic data."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nanoweather.evaluation.evaluate import default_model_name, evaluate_forecaster, get_init_indices, get_targets
from nanoweather.evaluation.forecasters import ClimatologyForecaster, persistence_forecaster


NUM_STEPS = 20


@pytest.fixture
def data():
    times = pd.date_range("2020-01-01T00", "2021-01-10T18", freq="6h").values
    weights = np.cos(np.deg2rad(np.linspace(-80, 80, 4)))
    return SimpleNamespace(
        times=times,
        states=np.random.default_rng(0).normal(size=(len(times), 2, 4, 8)).astype("f4"),
        latitude_weights=weights / weights.mean(),
        time_step_hours=6,
    )


@pytest.fixture
def init_indices(data):
    return get_init_indices(data.times, "2020-01-01T00", "2020-12-31T12", 12, NUM_STEPS)


def test_init_indices(data, init_indices):
    assert len(init_indices) == 732
    assert pd.Timestamp(data.times[init_indices[1]]).hour == 12


def test_perfect_forecast_scores_zero(data, init_indices):
    results = evaluate_forecaster(get_targets, data, init_indices, NUM_STEPS, batch_size=64)
    np.testing.assert_allclose(results["rmse"], 0)
    assert results["lead_hours"][-1] == 120


def test_persistence_first_step(data, init_indices):
    results = evaluate_forecaster(persistence_forecaster, data, init_indices, NUM_STEPS, batch_size=64)
    weights = data.latitude_weights[:, None]
    expected = np.mean(
        [np.sqrt(((data.states[i] - data.states[i + 1]) ** 2 * weights).mean(axis=(-2, -1))) for i in init_indices],
        axis=0,
    )
    np.testing.assert_allclose(results["rmse"][0], expected, atol=1e-5)


def test_climatology_lookup(data):
    rng = np.random.default_rng(1)
    climatology = xr.Dataset(
        {name: (("dayofyear", "hour", "latitude", "longitude"), rng.normal(size=(366, 4, 4, 8))) for name in "ab"},
        coords={"dayofyear": np.arange(1, 367), "hour": [0, 6, 12, 18]},
    )
    init = np.flatnonzero(data.times == np.datetime64("2020-03-01T12"))
    forecast = ClimatologyForecaster(climatology)(data, init, 2)
    np.testing.assert_allclose(forecast[0, 0, 0], climatology["a"].sel(dayofyear=61, hour=18).values)
    np.testing.assert_allclose(forecast[0, 1, 1], climatology["b"].sel(dayofyear=62, hour=0).values)


def test_default_model_name():
    assert default_model_name("checkpoints/epochs50/best") == "epochs50"
    assert default_model_name("user/repo") == "repo"
