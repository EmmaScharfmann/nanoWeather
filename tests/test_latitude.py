"""Tests for loading selected time steps and statistics by latitude."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from nanoweather.data.datasets import WeatherData
from nanoweather.evaluation.diagnostics import statistics_by_latitude
from nanoweather.processing.feature_extraction_nanoweather import NanoWeatherFeatureExtractor


def test_loading_selected_time_steps(tmp_path):
    times = pd.date_range("2020-01-01", periods=10, freq="6h")
    coords = {"latitude": np.linspace(-80, 80, 4), "longitude": np.arange(8) * 45.0}
    values = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((1, 4, 8), dtype=np.float32)
    xr.Dataset({"a": (("time", "latitude", "longitude"), values)}, coords={"time": times, **coords}).to_zarr(
        tmp_path / "store.zarr", group="test", zarr_format=2
    )
    static = {"land_sea_mask": (("latitude", "longitude"), np.zeros((4, 8), np.float32))}
    static["orography"] = static["land_sea_mask"]
    xr.Dataset(static, coords=coords).to_zarr(tmp_path / "store.zarr", group="static", mode="a", zarr_format=2)
    NanoWeatherFeatureExtractor(variables=["a"], mean=[0.0], std=[1.0]).save_pretrained(tmp_path)

    data = WeatherData(tmp_path, "test", "store.zarr", time_indices=np.array([2, 3, 7, 8]))
    assert data.states[:, 0, 0, 0].tolist() == [2.0, 3.0, 7.0, 8.0]
    assert data.time_step_hours == 6


def test_statistics_by_latitude():
    rng = np.random.default_rng(0)
    spread = np.array([1.0, 2.0, 4.0])
    states = rng.normal(size=(2000, 1, 3, 16)) * spread[None, None, :, None]
    data = SimpleNamespace(states=states, variables=["a"])
    statistics = statistics_by_latitude(data, np.arange(0, 1998, 2), ["a"])
    np.testing.assert_allclose(statistics["state_std"][0], spread, rtol=0.05)
    np.testing.assert_allclose(statistics["change_std"][0], spread * np.sqrt(2), rtol=0.05)
