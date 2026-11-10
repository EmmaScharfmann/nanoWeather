"""Tests for models with several input time steps."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from nanoweather.data.datasets import ForecastPairs
from nanoweather.evaluation.forecasters import ModelForecaster
from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting
from nanoweather.processing.feature_extraction_nanoweather import NanoWeatherFeatureExtractor


VARIABLES = ["z500", "t850", "t2m", "u10", "v10", "msl"]


def two_step_model():
    config = NanoWeatherConfig(
        num_input_steps=2, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64
    )
    model = NanoWeatherForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.05)
    return model


def test_forward_requires_previous_states():
    model = two_step_model()
    states, static = torch.randn(2, 6, 32, 64), torch.randn(2, 32, 64)
    with pytest.raises(ValueError):
        model(states, static=static)
    output = model(states, static=static, previous_states=torch.randn(2, 1, 6, 32, 64))
    assert output.prediction.shape == states.shape


def test_rollout_shifts_history():
    model = two_step_model()
    torch.manual_seed(0)
    previous, states, static = torch.randn(1, 1, 6, 32, 64), torch.randn(1, 6, 32, 64), torch.randn(2, 32, 64)
    rollout = model.rollout(states, static=static, num_steps=3, previous_states=previous)
    step1 = model(states, static=static, previous_states=previous).prediction
    step2 = model(step1, static=static, previous_states=states[:, None]).prediction
    step3 = model(step2, static=static, previous_states=step1[:, None]).prediction
    torch.testing.assert_close(rollout, torch.stack([step1, step2, step3], dim=1))


def fake_data(num_times=12):
    rng = np.random.default_rng(0)
    states = rng.normal(size=(num_times, 6, 32, 64)).astype(np.float32)
    return SimpleNamespace(
        times=pd.date_range("2020-01-01", periods=num_times, freq="6h").values,
        states=states,
        normalized_states=states,
        latitude=np.linspace(-87, 87, 32),
        longitude=np.arange(64) * 5.625,
        time_step_hours=6,
        variables=VARIABLES,
    )


def test_forecast_pairs_with_history():
    data = fake_data()
    pairs = ForecastPairs(data, num_input_steps=3)
    assert len(pairs) == 12 - 2 - 1
    sample = pairs[0]
    assert sample["previous_states"].shape == (2, 6, 32, 64)
    np.testing.assert_array_equal(sample["previous_states"].numpy(), data.states[0:2])
    np.testing.assert_array_equal(sample["states"].numpy(), data.states[2])
    np.testing.assert_array_equal(sample["labels"].numpy(), data.states[3])


def test_model_forecaster_with_history(tmp_path):
    model = two_step_model()
    model.save_pretrained(tmp_path)
    NanoWeatherFeatureExtractor(variables=VARIABLES, mean=[0.0] * 6, std=[1.0] * 6).save_pretrained(tmp_path)
    data = fake_data()
    static = np.random.default_rng(1).normal(size=(2, 32, 64)).astype(np.float32)
    forecaster = ModelForecaster(str(tmp_path), static, device="cpu")
    assert forecaster.num_input_steps == 2
    with pytest.raises(ValueError):
        forecaster(data, np.array([0]), 2)
    forecast = forecaster(data, np.array([3]), 2)
    expected = model.rollout(
        torch.from_numpy(data.states[3:4]),
        static=torch.from_numpy(static),
        num_steps=2,
        previous_states=torch.from_numpy(data.states[2:3])[:, None],
    )
    np.testing.assert_allclose(forecast, expected.numpy(), atol=1e-5)
