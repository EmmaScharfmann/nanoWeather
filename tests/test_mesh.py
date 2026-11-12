"""Tests for the mesh geometry and the mesh model."""

import json

import numpy as np
import pytest
import torch

from nanoweather.models import mesh_utils
from nanoweather.models.auto import load_model
from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.configuration_nanoweather_mesh import NanoWeatherMeshConfig
from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting
from nanoweather.models.modeling_nanoweather_mesh import (
    NanoWeatherMeshForWeatherForecasting,
    build_mesh_edges,
    dense_attention_forward,
    sparse_attention_forward,
)


def small_config(**kwargs):
    defaults = {"mesh_refinement": 2, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4}
    return NanoWeatherMeshConfig(**{**defaults, "intermediate_size": 64, **kwargs})


def inputs(config, batch_size=2):
    torch.manual_seed(0)
    states = torch.randn(batch_size, config.num_variables, *config.grid_size)
    return states, torch.randn(config.num_static_channels, *config.grid_size)


@pytest.mark.parametrize("refinement", [0, 1, 2, 3])
def test_icosahedral_mesh(refinement):
    nodes = mesh_utils.icosahedral_mesh(refinement)
    assert nodes.shape == (10 * 4**refinement + 2, 3)
    np.testing.assert_allclose(np.linalg.norm(nodes, axis=1), 1.0)
    assert len(np.unique(nodes.round(8), axis=0)) == len(nodes)


def test_grid_points_follow_the_data_layout():
    points, weights = mesh_utils.grid_points((32, 64))
    latitude, longitude = mesh_utils.grid_latitudes_longitudes((32, 64))
    np.testing.assert_allclose(points[64 * 5 + 7], mesh_utils.latitude_longitude_to_xyz(latitude[5], longitude[7]))
    assert np.isclose(weights.mean(), 1.0) and weights[0] < weights[16 * 64]


def test_relative_positions():
    receivers = mesh_utils.latitude_longitude_to_xyz(np.array([0.0]), np.array([0.0]))
    east = mesh_utils.latitude_longitude_to_xyz(np.array([0.0]), np.array([10.0]))
    north = mesh_utils.latitude_longitude_to_xyz(np.array([10.0]), np.array([0.0]))
    to_east = mesh_utils.relative_positions(receivers, east, 1.0)[0]
    to_north = mesh_utils.relative_positions(receivers, north, 1.0)[0]
    assert to_east[0] > 0 and abs(to_east[1]) < 1e-9
    assert to_north[1] > 0 and abs(to_north[0]) < 1e-9
    pole = np.array([[0.0, 0.0, 1.0]])
    assert np.isfinite(mesh_utils.relative_positions(pole, receivers, 1.0)).all()


def test_edges_reach_every_grid_point_and_node():
    edges = build_mesh_edges(small_config())
    encoder, processor, decoder = edges["encoder"], edges["processor"], edges["decoder"]
    assert set(encoder.senders.tolist()) == set(range(32 * 64))
    assert (processor.senders.view(-1, processor.num_neighbors)[:, 0] == torch.arange(processor.num_receivers)).all()
    assert decoder.num_receivers == 32 * 64
    assert torch.isfinite(encoder.log_weights).all()


@pytest.mark.parametrize("edge_set", ["encoder", "processor", "decoder"])
def test_sparse_and_dense_attention_match(edge_set):
    edges = build_mesh_edges(small_config())[edge_set]
    torch.manual_seed(0)
    query = torch.randn(2, edges.num_receivers, 4, 8, requires_grad=True)
    key = torch.randn(2, edges.num_senders, 4, 8, requires_grad=True)
    value = torch.randn(2, edges.num_senders, 4, 8, requires_grad=True)
    bias = torch.randn(edges.receivers.numel(), 4, requires_grad=True)
    results = []
    for function in (sparse_attention_forward, dense_attention_forward):
        output = function(query, key, value, edges, bias, 0.35)
        gradients = torch.autograd.grad(output.square().sum(), (query, key, value, bias))
        results.append((output, gradients))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=1e-5)
    for sparse_gradient, dense_gradient in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(sparse_gradient, dense_gradient, atol=1e-4, rtol=1e-4)


def test_area_weights_balance_crowded_polar_points():
    """With uninformative queries and keys, a node averages the grid points it receives, each weighted by its area."""
    edges = build_mesh_edges(small_config())["encoder"]
    receiver = int(edges.receivers[0])
    mine = edges.receivers == receiver
    senders = edges.senders[mine]
    query = torch.zeros(1, edges.num_receivers, 1, 4)
    key = torch.zeros(1, edges.num_senders, 1, 4)
    value = torch.zeros(1, edges.num_senders, 1, 4)
    value[0, senders, 0, 0] = torch.arange(len(senders), dtype=torch.float32)
    bias = edges.log_weights[:, None].clone()
    output = sparse_attention_forward(query, key, value, edges, bias, 1.0)[0, receiver, 0, 0]
    weights = torch.exp(edges.log_weights[mine])
    expected = (weights * torch.arange(len(senders))).sum() / weights.sum()
    torch.testing.assert_close(output, expected)


def test_untrained_model_is_persistence_and_all_parameters_train():
    config = small_config()
    model = NanoWeatherMeshForWeatherForecasting(config)
    states, static = inputs(config)
    assert torch.equal(model(states, static=static).prediction, states)
    torch.nn.init.normal_(model.head.weight, std=0.05)
    model(states, static=static, labels=states + 0.1).loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("implementation", ["sparse", "dense", "auto"])
def test_attention_implementations_give_the_same_forecast(implementation):
    config = small_config()
    model = NanoWeatherMeshForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.05)
    states, static = inputs(config)
    reference = model(states, static=static).prediction
    model.config.attn_implementation = implementation
    torch.testing.assert_close(model(states, static=static).prediction, reference, atol=1e-5, rtol=1e-5)


def test_two_input_steps_time_features_and_rollout():
    config = small_config(num_input_steps=2, num_time_features=4)
    model = NanoWeatherMeshForWeatherForecasting(config).eval()
    torch.nn.init.normal_(model.head.weight, std=0.05)
    states, static = inputs(config)
    previous = torch.randn(2, 1, config.num_variables, *config.grid_size)
    time_features = torch.rand(2, 3, 4, *config.grid_size)
    rollout = model.rollout(states, static=static, num_steps=3, time_features=time_features, previous_states=previous)
    assert rollout.shape == (2, 3, config.num_variables, *config.grid_size)
    first = model(states, static=static, time_features=time_features[:, 0], previous_states=previous).prediction
    torch.testing.assert_close(rollout[:, 0], first)


def test_save_and_load_by_architecture(tmp_path):
    mesh = NanoWeatherMeshForWeatherForecasting(small_config(residual_scale=[0.5] * 6)).eval()
    torch.nn.init.normal_(mesh.head.weight, std=0.05)
    mesh.save_pretrained(tmp_path / "mesh")
    loaded = load_model(tmp_path / "mesh")
    assert isinstance(loaded, NanoWeatherMeshForWeatherForecasting) and loaded.config == mesh.config
    states, static = inputs(mesh.config)
    torch.testing.assert_close(loaded(states, static=static).prediction, mesh(states, static=static).prediction)
    assert not any("edges" in name or "node_positions" in name for name in mesh.state_dict())

    vit = NanoWeatherForWeatherForecasting(
        NanoWeatherConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=1)
    )
    vit.save_pretrained(tmp_path / "vit")
    config = json.loads((tmp_path / "vit" / "config.json").read_text())
    config.pop("model_type")
    (tmp_path / "vit" / "config.json").write_text(json.dumps(config))
    assert isinstance(load_model(tmp_path / "vit"), NanoWeatherForWeatherForecasting)


def test_invalid_mesh_config():
    with pytest.raises(ValueError):
        small_config(processor_num_neighbors=0)
    with pytest.raises(ValueError):
        small_config(mesh_refinement=0, processor_num_neighbors=20)
    with pytest.raises(ValueError):
        small_config(attn_implementation="eager")
