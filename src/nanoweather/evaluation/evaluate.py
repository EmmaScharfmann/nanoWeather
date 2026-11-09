"""Scoring forecasters with latitude-weighted RMSE per variable and lead time."""

import json
import logging
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from .metrics import latitude_weighted_rmse


logger = logging.getLogger(__name__)


def get_init_indices(times: np.ndarray, start: str, end: str, frequency_hours: int, num_steps: int) -> np.ndarray:
    """
    Indices in `times` of initializations from `start` to `end` every `frequency_hours`, keeping only those followed
    by at least `num_steps` time steps.
    """
    init_times = pd.date_range(start, end, freq=f"{frequency_hours}h").values
    position = {time: index for index, time in enumerate(times)}
    indices = np.array([position[time] for time in init_times if time in position])
    indices = indices[indices + num_steps < len(times)]
    if len(indices) != len(init_times):
        logger.warning(f"Only {len(indices)} of {len(init_times)} initializations have all {num_steps} targets.")
    return indices


def get_targets(data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
    """Truth at each lead time, shape `(num_inits, num_steps, num_variables, latitude, longitude)`."""
    leads = np.arange(1, num_steps + 1)
    return data.states[init_indices[:, None] + leads[None, :]]


def get_forecast_variables(forecaster: Callable, data) -> list[str]:
    return list(getattr(forecaster, "variables", data.variables))


def get_common_variables(forecasters: list[Callable], data) -> list[str]:
    """Variables forecast by every forecaster, in the order of `data.variables`."""
    common = set(data.variables)
    for forecaster in forecasters:
        common &= set(get_forecast_variables(forecaster, data))
    return [name for name in data.variables if name in common]


def evaluate_forecaster(
    forecaster: Callable, data, init_indices: np.ndarray, num_steps: int, batch_size: int, variables: list[str]
) -> dict:
    """
    Latitude-weighted RMSE of a forecaster, computed for each initialization and then averaged.

    Args:
        forecaster (`Callable`):
            `forecaster(data, init_indices, num_steps)`, see `nanoweather.evaluation.forecasters`.
        data (`WeatherData`):
            Split to evaluate on.
        init_indices (`np.ndarray`):
            Indices of the initializations in `data.times`.
        num_steps (`int`):
            Number of time steps to forecast.
        batch_size (`int`):
            Number of initializations forecast at once.
        variables (`list[str]`):
            Variables to score. Must be forecast by `forecaster` and present in `data`.

    Returns:
        `dict` with `lead_hours` of shape `(num_steps,)` and `rmse` of shape `(num_steps, len(variables))`.
    """
    forecast_variables = get_forecast_variables(forecaster, data)
    forecast_channels = [forecast_variables.index(name) for name in variables]
    target_channels = [data.variables.index(name) for name in variables]
    rmse_per_init = []
    for start in range(0, len(init_indices), batch_size):
        batch = init_indices[start : start + batch_size]
        forecast = forecaster(data, batch, num_steps)[:, :, forecast_channels]
        targets = get_targets(data, batch, num_steps)[:, :, target_channels]
        if forecast.shape != targets.shape:
            raise ValueError(f"Forecast shape {forecast.shape} does not match targets {targets.shape}.")
        rmse_per_init.append(latitude_weighted_rmse(forecast, targets, data.latitude_weights))
    return {
        "lead_hours": np.arange(1, num_steps + 1) * data.time_step_hours,
        "rmse": np.concatenate(rmse_per_init).mean(axis=0),
    }


def save_results(results: dict, variables: list[str], path: str | os.PathLike):
    """Writes `{"lead_hours": [...], "rmse": {variable: [...]}}` as JSON."""
    output = {
        "lead_hours": results["lead_hours"].tolist(),
        "rmse": {name: results["rmse"][:, index].tolist() for index, name in enumerate(variables)},
    }
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(output, writer, indent=2)


def plot_rmse(all_results: dict[str, dict], variables: list[str], plot_variables: list[str], path=None):
    """Plots RMSE against lead time, one panel per variable and one line per forecaster."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(plot_variables), figsize=(5 * len(plot_variables), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, plot_variables):
        index = variables.index(name)
        for forecaster_name, results in all_results.items():
            ax.plot(
                results["lead_hours"] / 24, results["rmse"][:, index], marker="o", markersize=3, label=forecaster_name
            )
        ax.set_title(name)
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
    axes[0].legend()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def default_model_name(checkpoint: str) -> str:
    """Name of a checkpoint in results: `checkpoints/epochs50/best` gives `epochs50`, `user/repo` gives `repo`."""
    parts = [part for part in str(checkpoint).rstrip("/").split("/") if part not in ("best", "last")]
    return parts[-1]
