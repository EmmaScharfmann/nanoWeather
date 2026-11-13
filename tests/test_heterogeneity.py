"""Tests for the analysis of where forecast errors are."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nanoweather.evaluation.diagnostics import compute_diagnostics
from nanoweather.evaluation.evaluate import get_targets
from nanoweather.evaluation.heterogeneity import (
    analyze_heterogeneity,
    concentration_curve,
    region_masks,
    scores_by_region,
    share_from_worst,
    systematic_share,
)


LATITUDE = np.linspace(-80, 80, 8)
AREA = np.cos(np.deg2rad(LATITUDE))[:, None] * np.ones((8, 16))


def test_concentration_of_evenly_spread_error_is_zero():
    area_share, error_share, gini = concentration_curve(np.ones((8, 16)), AREA)
    np.testing.assert_allclose(error_share, area_share, atol=1e-12)
    assert gini == pytest.approx(0.0, abs=1e-12)


def test_concentration_of_error_in_one_point_is_close_to_one():
    squared_error = np.zeros((8, 16))
    squared_error[4, 3] = 1.0
    _, _, gini = concentration_curve(squared_error, AREA)
    assert gini > 0.98


def test_share_from_worst():
    values = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert share_from_worst(values, 0.1) == pytest.approx(10 / 19)


def test_region_masks():
    land = np.zeros((8, 16))
    land[3:5, 4:8] = 1.0
    orography = np.zeros((8, 16))
    orography[3:5, 6:8] = 2000 * 9.80665
    masks = region_masks(land, orography, LATITUDE)
    assert masks["high terrain"].sum() == 4 and masks["land"].sum() == 4
    assert masks["coast"][3, 3] and masks["coast"][3, 4] and not masks["coast"][0, 0]
    assert (masks["tropics"] | masks["mid-latitudes"] | masks["polar"]).all()
    assert not (masks["ocean"] & (masks["land"] | masks["high terrain"])).any()


def test_scores_by_region_and_systematic_share():
    masks = {"all": np.ones((8, 16), bool), "north": np.broadcast_to(LATITUDE[:, None] > 0, (8, 16))}
    scores = scores_by_region(np.full((8, 16), 2.0), np.full((8, 16), 4.0), masks, AREA)
    assert scores["all"]["ratio"] == pytest.approx(0.5) and scores["north"]["area_share"] == pytest.approx(0.5)
    np.testing.assert_allclose(systematic_share(np.full(3, 1.0), np.full(3, 2.0)), 0.25)


def test_maps_and_forecasts_from_the_diagnostics_pass():
    times = pd.date_range("2020-01-01T00", periods=60, freq="6h").values
    rng = np.random.default_rng(0)
    weights = AREA[:, 0] / AREA[:, 0].mean()
    data = SimpleNamespace(
        times=times,
        states=rng.normal(size=(60, 1, 8, 16)).astype("f4"),
        latitude_weights=weights,
        time_step_hours=6,
        variables=["a"],
    )

    def forecaster(data, init_indices, num_steps):
        error = np.zeros((len(init_indices), num_steps, 1, 8, 16))
        error[..., 2, 5] = 3.0
        return get_targets(data, init_indices, num_steps) + error

    def climatology(data, init_indices, num_steps):
        return np.zeros_like(get_targets(data, init_indices, num_steps))

    inits = np.arange(0, 40, 2)
    diagnostics = compute_diagnostics(forecaster, data, inits, [1, 4], ["a"], batch_size=8)
    assert diagnostics["rmse_map"].shape == (2, 1, 8, 16)
    assert diagnostics["rmse_by_initialization"].shape == (20, 2, 1)
    np.testing.assert_allclose(diagnostics["rmse_map"][..., 2, 5], 3.0)
    np.testing.assert_allclose((diagnostics["rmse_map"] ** 2).mean(axis=-1), diagnostics["rmse_by_latitude"] ** 2)

    diagnostics["climatology_rmse_map"] = compute_diagnostics(climatology, data, inits, [1, 4], ["a"], 8)["rmse_map"]
    land, orography = np.zeros((8, 16)), np.zeros((8, 16))
    summary = analyze_heterogeneity(diagnostics, ["a"], LATITUDE, land, orography)
    assert summary["a"]["1"]["gini"] > 0.95
    assert summary["a"]["1"]["systematic_share"] == pytest.approx(1.0)
    assert summary["a"]["1"]["worst_10_percent_forecasts"] == pytest.approx(0.1, abs=0.01)
