"""Tests for dataset preparation, on a synthetic store in the source layout."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nanoweather.config import load_yaml
from nanoweather.processing.feature_extraction_nanoweather import NanoWeatherFeatureExtractor
from nanoweather.processing.prepare import compute_statistics, expand_variables, select_variables, write_dataset_card


VARIABLES = {
    "z500": {"source": "geopotential", "level": 500},
    "t2m": {"source": "2m_temperature"},
}


@pytest.fixture
def source():
    rng = np.random.default_rng(0)
    coords = {
        "time": pd.date_range("2000-01-01", periods=8, freq="6h"),
        "level": [500, 850],
        "longitude": np.arange(8) * 45.0,
        "latitude": np.linspace(-60, 60, 4),
    }
    dataset = xr.Dataset(coords=coords)
    dataset["geopotential"] = (("time", "level", "longitude", "latitude"), rng.normal(size=(8, 2, 8, 4)).astype("f4"))
    dataset["2m_temperature"] = (("time", "longitude", "latitude"), rng.normal(size=(8, 8, 4)).astype("f4"))
    dataset["2m_temperature"].attrs["units"] = "K"
    return dataset


def test_select_variables_layout(source):
    source["2m_temperature"][0, 3, 1] = 999.0
    selected = select_variables(source, VARIABLES)
    assert list(selected.data_vars) == ["z500", "t2m"]
    assert selected["t2m"].dims == ("time", "latitude", "longitude")
    assert float(selected["t2m"][0, 1, 3]) == 999.0
    expected = source["geopotential"].sel(level=500).values.transpose(0, 2, 1)
    np.testing.assert_array_equal(selected["z500"].values, expected)


def test_compute_statistics(source):
    statistics = compute_statistics(select_variables(source, VARIABLES))
    t2m = source["2m_temperature"].values.astype(np.float64)
    assert statistics["mean"][1] == pytest.approx(t2m.mean())
    assert statistics["diff_std"][1] == pytest.approx(np.diff(t2m, axis=0).std())
    assert statistics["units"] == ["", "K"]


def test_dataset_card(tmp_path):
    data_config = load_yaml("configs/data.yaml")
    feature_extractor = NanoWeatherFeatureExtractor(
        variables=list(data_config["variables"]),
        mean=[1.0] * 6,
        std=[2.0] * 6,
        diff_std=[0.5] * 6,
        units=["K"] * 6,
    )
    num_steps = dict.fromkeys(data_config["splits"], 10)
    write_dataset_card(tmp_path, load_yaml("configs/dataset_card.yaml"), data_config, feature_extractor, num_steps)
    card = (tmp_path / "README.md").read_text()
    assert "| `z500` | geopotential | 500 | K |" in card
    assert "| `t2m` | 2m_temperature | surface | K |" in card
    assert "| train | 2000-01-01T00 | 2015-12-31T18 | 10 |" in card
    assert "{" not in card


def test_expand_variables():
    config = {
        "pressure_levels": [500, 850],
        "pressure_level_variables": {"z": "geopotential", "u": "u_component_of_wind"},
        "variables": {"t2m": {"source": "2m_temperature"}},
    }
    assert expand_variables(config) == {
        "z500": {"source": "geopotential", "level": 500},
        "z850": {"source": "geopotential", "level": 850},
        "u500": {"source": "u_component_of_wind", "level": 500},
        "u850": {"source": "u_component_of_wind", "level": 850},
        "t2m": {"source": "2m_temperature"},
    }
    assert list(expand_variables(load_yaml("configs/data.yaml"))) == ["z500", "t850", "t2m", "u10", "v10", "msl"]
    assert len(expand_variables(load_yaml("configs/data_29vars.yaml"))) == 29
