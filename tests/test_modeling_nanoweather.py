"""Tests for the model."""

import pytest
import torch

from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting, get_latitude_weights


BATCH_SIZE = 3


@pytest.fixture
def config():
    return NanoWeatherConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)


@pytest.fixture
def inputs(config):
    torch.manual_seed(0)
    shape = (BATCH_SIZE, config.num_variables, *config.grid_size)
    return {
        "states": torch.randn(shape),
        "static": torch.randn(config.num_static_channels, *config.grid_size),
        "labels": torch.randn(shape),
    }


def test_forward_shapes_and_loss(config, inputs):
    model = NanoWeatherForWeatherForecasting(config)
    output = model(**inputs)
    assert output.prediction.shape == inputs["states"].shape
    assert output.last_hidden_state.shape == (BATCH_SIZE, config.num_patches, config.hidden_size)
    assert output.loss.dim() == 0


def test_untrained_model_is_persistence(config, inputs):
    model = NanoWeatherForWeatherForecasting(config)
    output = model(inputs["states"], static=inputs["static"])
    assert torch.equal(output.prediction, inputs["states"])


def test_all_parameters_receive_gradients(config, inputs):
    model = NanoWeatherForWeatherForecasting(config)
    torch.nn.init.normal_(model.head.weight, std=0.02)
    model(**inputs).loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing, missing


def test_unpatchify_inverts_row_major_patching(config):
    model = NanoWeatherForWeatherForecasting(config)
    patch_size = config.patch_size
    num_rows, num_columns = (size // patch_size for size in config.grid_size)
    grid = torch.randn(2, config.num_variables, *config.grid_size)
    patches = grid.reshape(2, config.num_variables, num_rows, patch_size, num_columns, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(2, config.num_patches, -1)
    assert torch.equal(model.unpatchify(patches), grid)


def test_eager_and_sdpa_match(config, inputs):
    model = NanoWeatherForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.02)
    sdpa_prediction = model(inputs["states"], static=inputs["static"]).prediction
    model.config.attn_implementation = "eager"
    eager_prediction = model(inputs["states"], static=inputs["static"]).prediction
    torch.testing.assert_close(sdpa_prediction, eager_prediction, atol=1e-5, rtol=1e-5)


def test_save_and_load(config, inputs, tmp_path):
    model = NanoWeatherForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.02)
    model.save_pretrained(tmp_path)
    loaded = NanoWeatherForWeatherForecasting.from_pretrained(tmp_path)
    assert loaded.config == model.config
    expected = model(inputs["states"], static=inputs["static"]).prediction
    torch.testing.assert_close(loaded(inputs["states"], static=inputs["static"]).prediction, expected)


def test_rollout(config, inputs):
    model = NanoWeatherForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.02)
    predictions = model.rollout(inputs["states"], static=inputs["static"], num_steps=4)
    assert predictions.shape == (BATCH_SIZE, 4, config.num_variables, *config.grid_size)
    first_step = model(inputs["states"], static=inputs["static"]).prediction
    torch.testing.assert_close(predictions[:, 0], first_step)


def test_latitude_weights():
    weights = get_latitude_weights(32)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[16] > weights[0]


def test_invalid_config():
    with pytest.raises(ValueError):
        NanoWeatherConfig(patch_size=3)
    with pytest.raises(ValueError):
        NanoWeatherConfig(hidden_size=100, num_attention_heads=8)


def test_time_features(config, inputs):
    config.num_time_features = 4
    model = NanoWeatherForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.02)
    time_features = torch.rand(BATCH_SIZE, 4, *config.grid_size)
    with pytest.raises(ValueError):
        model(inputs["states"], static=inputs["static"])
    output = model(inputs["states"], static=inputs["static"], time_features=time_features)
    assert output.prediction.shape == inputs["states"].shape

    num_steps = 3
    step_features = torch.rand(BATCH_SIZE, num_steps, 4, *config.grid_size)
    predictions = model.rollout(
        inputs["states"], static=inputs["static"], num_steps=num_steps, time_features=step_features
    )
    assert predictions.shape == (BATCH_SIZE, num_steps, config.num_variables, *config.grid_size)
    first_step = model(inputs["states"], static=inputs["static"], time_features=step_features[:, 0]).prediction
    torch.testing.assert_close(predictions[:, 0], first_step)
