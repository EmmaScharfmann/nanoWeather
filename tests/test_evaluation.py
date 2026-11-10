"""Tests for the evaluation protocol and baselines, on synthetic data."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nanoweather.evaluation.diagnostics import compute_diagnostics, zonal_power_spectrum
from nanoweather.evaluation.evaluate import (
    default_model_name,
    evaluate_forecaster,
    get_common_variables,
    get_init_indices,
    get_targets,
)
from nanoweather.evaluation.forecasters import ClimatologyForecaster, persistence_forecaster
from nanoweather.evaluation.metrics import anomaly_correlation, skill_horizon_hours


NUM_STEPS = 20
VARIABLES = ["a", "b"]


@pytest.fixture
def data():
    times = pd.date_range("2020-01-01T00", "2021-01-10T18", freq="6h").values
    weights = np.cos(np.deg2rad(np.linspace(-80, 80, 4)))
    return SimpleNamespace(
        times=times,
        states=np.random.default_rng(0).normal(size=(len(times), 2, 4, 8)).astype("f4"),
        latitude_weights=weights / weights.mean(),
        time_step_hours=6,
        variables=VARIABLES,
    )


@pytest.fixture
def init_indices(data):
    return get_init_indices(data.times, "2020-01-01T00", "2020-12-31T12", 12, NUM_STEPS)


def test_init_indices(data, init_indices):
    assert len(init_indices) == 732
    assert pd.Timestamp(data.times[init_indices[1]]).hour == 12


def test_perfect_forecast_scores_zero(data, init_indices):
    results = evaluate_forecaster(get_targets, data, init_indices, NUM_STEPS, batch_size=64, variables=VARIABLES)
    np.testing.assert_allclose(results["rmse"], 0)
    assert results["lead_hours"][-1] == 120


def test_persistence_first_step(data, init_indices):
    results = evaluate_forecaster(
        persistence_forecaster, data, init_indices, NUM_STEPS, batch_size=64, variables=VARIABLES
    )
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


def test_scoring_a_subset_of_variables(data, init_indices):
    def forecaster(data, init_indices, num_steps):
        return get_targets(data, init_indices, num_steps)[:, :, ::-1]

    forecaster.variables = ["b", "a"]
    assert get_common_variables([forecaster, persistence_forecaster], data) == ["a", "b"]
    results = evaluate_forecaster(forecaster, data, init_indices, NUM_STEPS, batch_size=64, variables=["b"])
    np.testing.assert_allclose(results["rmse"], 0)
    assert results["rmse"].shape == (NUM_STEPS, 1)


def test_zonal_spectrum_of_a_single_wave():
    longitude = np.arange(64)
    field = np.tile(np.cos(2 * np.pi * 3 * longitude / 64), (4, 1))
    spectrum = zonal_power_spectrum(field, np.ones(4))
    assert np.argmax(spectrum) == 3
    np.testing.assert_allclose(spectrum[3], 0.25)


def test_diagnostics_of_a_biased_forecast(data, init_indices):
    def forecaster(data, init_indices, num_steps):
        return get_targets(data, init_indices, num_steps) + 1.0

    diagnostics = compute_diagnostics(forecaster, data, init_indices, [1, 4], VARIABLES, batch_size=64)
    np.testing.assert_allclose(diagnostics["bias"], 1.0, atol=1e-6)
    np.testing.assert_allclose(diagnostics["rmse_by_latitude"], 1.0, atol=1e-6)
    assert diagnostics["bias"].shape == (2, 2, 4, 8)
    assert list(diagnostics["lead_hours"]) == [6, 24]


def test_anomaly_correlation():
    rng = np.random.default_rng(2)
    climatology = rng.normal(size=(3, 4, 8))
    truth = climatology + rng.normal(size=(3, 4, 8))
    weights = np.ones(4)
    np.testing.assert_allclose(anomaly_correlation(truth, truth, climatology, weights), 1.0)
    np.testing.assert_allclose(anomaly_correlation(2 * climatology - truth, truth, climatology, weights), -1.0)
    assert np.isnan(anomaly_correlation(climatology, truth, climatology, weights)).all()


def test_skill_horizon():
    lead_hours = np.array([6, 12, 18, 24])
    assert skill_horizon_hours(lead_hours, np.array([0.9, 0.8, 0.5, 0.4]), 0.6) == pytest.approx(16.0)
    assert skill_horizon_hours(lead_hours, np.array([0.9, 0.8, 0.7, 0.65]), 0.6) is None
    assert skill_horizon_hours(lead_hours, np.full(4, np.nan), 0.6) is None


def test_evaluation_with_acc(data, init_indices):
    def climatology_forecaster(data, init_indices, num_steps):
        return np.zeros_like(get_targets(data, init_indices, num_steps))

    climatology_forecaster.variables = VARIABLES
    results = evaluate_forecaster(
        get_targets,
        data,
        init_indices,
        NUM_STEPS,
        batch_size=64,
        variables=VARIABLES,
        climatology=climatology_forecaster,
    )
    np.testing.assert_allclose(results["acc"], 1.0)
    results = evaluate_forecaster(
        persistence_forecaster, data, init_indices, NUM_STEPS, batch_size=64, variables=VARIABLES
    )
    assert results["acc"] is None


def test_init_indices_with_history(data):
    indices = get_init_indices(data.times, "2020-01-01T00", "2020-12-31T12", 12, NUM_STEPS, num_history_steps=1)
    assert len(indices) == 731
    assert indices.min() >= 1
