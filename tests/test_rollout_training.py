"""Tests for multi-step samples and the rollout loss."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from nanoweather.data.datasets import ForecastPairs
from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting
from nanoweather.training.trainer import rollout_loss


def fake_data(num_times=12):
    states = np.random.default_rng(0).normal(size=(num_times, 6, 32, 64)).astype(np.float32)
    return SimpleNamespace(
        times=pd.date_range("2020-01-01", periods=num_times, freq="6h").values,
        normalized_states=states,
        latitude=np.linspace(-87, 87, 32),
        longitude=np.arange(64) * 5.625,
        time_step_hours=6,
    )


def small_model(**kwargs):
    config = NanoWeatherConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, **kwargs
    )
    model = NanoWeatherForWeatherForecasting(config)
    torch.nn.init.normal_(model.head.weight, std=0.05)
    return model


def batch_of(pairs, indices):
    return torch.utils.data.default_collate([pairs[index] for index in indices])


def test_multi_step_samples():
    data = fake_data()
    pairs = ForecastPairs(data, num_rollout_steps=3, time_features=True, num_input_steps=2)
    assert len(pairs) == 12 - 1 - 3
    sample = pairs[0]
    assert sample["labels"].shape == (3, 6, 32, 64)
    assert sample["time_features"].shape == (3, 4, 32, 64)
    np.testing.assert_array_equal(sample["previous_states"].numpy(), data.normalized_states[0:1])
    np.testing.assert_array_equal(sample["states"].numpy(), data.normalized_states[1])
    np.testing.assert_array_equal(sample["labels"].numpy(), data.normalized_states[2:5])
    assert ForecastPairs(data)[0]["labels"].shape == (6, 32, 64)


def test_one_step_rollout_loss_is_the_model_loss():
    model = small_model()
    batch = batch_of(ForecastPairs(fake_data()), [0, 1])
    static = torch.randn(2, 32, 64)
    torch.testing.assert_close(rollout_loss(model, batch, static), model(**batch, static=static).loss)


def test_rollout_loss_feeds_predictions_back():
    torch.manual_seed(0)
    model = small_model(num_input_steps=2)
    batch = batch_of(ForecastPairs(fake_data(), num_rollout_steps=3, num_input_steps=2), [0, 1])
    static = torch.randn(2, 32, 64)

    states, previous, labels = batch["states"], batch["previous_states"], batch["labels"]
    expected = []
    for step in range(3):
        output = model(states, static=static, previous_states=previous, labels=labels[:, step])
        expected.append(output.loss)
        previous, states = states[:, None], output.prediction
    torch.testing.assert_close(rollout_loss(model, batch, static), torch.stack(expected).mean())


def test_gradient_checkpointing_gives_the_same_gradients():
    torch.manual_seed(0)
    model = small_model()
    batch = batch_of(ForecastPairs(fake_data(), num_rollout_steps=3), [0, 1])
    static = torch.randn(2, 32, 64)
    gradients = []
    for gradient_checkpointing in (False, True):
        model.zero_grad()
        loss = rollout_loss(model, batch, static, gradient_checkpointing)
        loss.backward()
        gradients.append([parameter.grad.clone() for parameter in model.parameters()])
    for without, with_checkpointing in zip(*gradients):
        torch.testing.assert_close(without, with_checkpointing)
