"""Tests for tendency normalization and per-variable loss weights."""

import json

import numpy as np
import pytest
import torch

from nanoweather.config import load_yaml
from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting, latitude_weighted_mse
from nanoweather.processing.prepare import expand_variables
from nanoweather.training.loss_weights import get_pressure_loss_weights


SURFACE_WEIGHTS = {"t2m": 1.0, "u10": 0.1, "v10": 0.1, "msl": 0.1}


def small_config(**kwargs):
    return NanoWeatherConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, **kwargs
    )


def test_pressure_weights_six_variables():
    specs = expand_variables(load_yaml("experiments/configs/data_6vars.yaml"))
    weights = get_pressure_loss_weights(list(specs), specs, SURFACE_WEIGHTS)
    expected = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(weights, expected / expected.mean())


def test_pressure_weights_29_variables():
    specs = expand_variables(load_yaml("configs/data.yaml"))
    variables = list(specs)
    weights = dict(zip(variables, get_pressure_loss_weights(variables, specs, SURFACE_WEIGHTS)))
    assert np.mean(list(weights.values())) == pytest.approx(1.0)
    levels = [250, 500, 700, 850, 925]
    for short_name in "ztuvq":
        group = [weights[f"{short_name}{level}"] for level in levels]
        assert sum(group) == pytest.approx(weights["t2m"])
        np.testing.assert_allclose(np.array(group) / group[0], np.array(levels) / levels[0])
    assert weights["u10"] == pytest.approx(0.1 * weights["t2m"])


def test_weighted_mse():
    torch.manual_seed(0)
    prediction, target = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)
    latitude_weights = torch.ones(4)
    variable_weights = torch.tensor([2.0, 0.5, 0.5])
    per_variable = ((prediction - target) ** 2).mean(dim=(0, 2, 3))
    expected = (per_variable * variable_weights).mean()
    torch.testing.assert_close(latitude_weighted_mse(prediction, target, latitude_weights, variable_weights), expected)
    torch.testing.assert_close(latitude_weighted_mse(prediction, target, latitude_weights), per_variable.mean())


def test_tendency_normalization():
    residual_scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    model = NanoWeatherForWeatherForecasting(small_config(residual_scale=residual_scale)).eval()
    states = torch.randn(2, 6, 32, 64)
    static = torch.randn(2, 32, 64)
    assert torch.equal(model(states, static=static).prediction, states)

    torch.nn.init.normal_(model.head.weight, std=0.02)
    output = model(states, static=static, labels=states)
    scale = torch.tensor(residual_scale)[:, None, None]
    change = output.prediction - states
    expected_loss = latitude_weighted_mse(change / scale, torch.zeros_like(change), model.latitude_weights)
    torch.testing.assert_close(output.loss, expected_loss)


def test_new_fields_round_trip_and_old_configs_load(tmp_path):
    config = small_config(residual_scale=[0.5] * 6, loss_weights=[1.0] * 6)
    model = NanoWeatherForWeatherForecasting(config)
    model.save_pretrained(tmp_path)
    loaded = NanoWeatherForWeatherForecasting.from_pretrained(tmp_path)
    assert loaded.config.residual_scale == [0.5] * 6
    torch.testing.assert_close(loaded.residual_scale, model.residual_scale)

    old_config = json.loads((tmp_path / "config.json").read_text())
    del old_config["residual_scale"], old_config["loss_weights"]
    old = NanoWeatherConfig.from_dict(old_config)
    assert old.residual_scale is None and old.loss_weights is None
    assert NanoWeatherForWeatherForecasting(old).residual_scale is None


def test_invalid_settings():
    with pytest.raises(ValueError):
        small_config(loss_weights=[1.0] * 5)
    with pytest.raises(ValueError):
        small_config(residual_scale=[1.0] * 6, predict_residual=False)


def test_latitude_weight_exponent():
    from nanoweather.models.modeling_utils import get_latitude_weights

    torch.testing.assert_close(get_latitude_weights(32, 0.0), torch.ones(32))
    area, softened = get_latitude_weights(32, 1.0), get_latitude_weights(32, 0.5)
    assert softened[0] > 2 * area[0]
    assert torch.isclose(softened.mean(), torch.tensor(1.0))


def test_change_std_by_latitude_matches_direct_computation():
    from nanoweather.training.loss_weights import change_std_by_latitude

    states = np.random.default_rng(0).normal(size=(50, 2, 4, 8)) * np.arange(1, 5)[None, None, :, None]
    expected = np.diff(states, axis=0).std(axis=(0, 3))
    np.testing.assert_allclose(change_std_by_latitude(states, chunk_size=7), expected, rtol=1e-10)


@pytest.mark.parametrize("architecture", ["vit", "mesh"])
def test_residual_scale_per_latitude(architecture, tmp_path):
    from nanoweather.models.auto import load_model, model_class_for
    from nanoweather.models.configuration_nanoweather_mesh import NanoWeatherMeshConfig

    scale = (np.linspace(0.2, 1.0, 32)[None, :] * np.arange(1, 7)[:, None] / 6).tolist()
    sizes = {"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 64}
    if architecture == "vit":
        config = NanoWeatherConfig(residual_scale=scale, latitude_weight_exponent=0.5, **sizes)
    else:
        config = NanoWeatherMeshConfig(residual_scale=scale, latitude_weight_exponent=0.5, mesh_refinement=1, **sizes)
    model = model_class_for(config.model_type)(config).eval()
    states, static = torch.randn(2, 6, 32, 64), torch.randn(2, 32, 64)
    assert torch.equal(model(states, static=static).prediction, states)

    torch.nn.init.normal_(model.head.weight, std=0.05)
    output = model(states, static=static, labels=states)
    change = output.prediction - states
    scale_grid = torch.tensor(scale, dtype=torch.float32)[:, :, None]
    expected = latitude_weighted_mse(change / scale_grid, torch.zeros_like(change), model.latitude_weights)
    torch.testing.assert_close(output.loss, expected)

    model.save_pretrained(tmp_path)
    loaded = load_model(tmp_path)
    assert loaded.config.residual_scale == scale and loaded.config.latitude_weight_exponent == 0.5
    torch.testing.assert_close(loaded(states, static=static).prediction, model(states, static=static).prediction)


def test_invalid_per_latitude_residual_scale():
    with pytest.raises(ValueError):
        small_config(residual_scale=[[1.0] * 31] * 6)
