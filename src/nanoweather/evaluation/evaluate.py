"""Scoring forecasters with latitude-weighted RMSE per variable and lead time."""

import json
import logging
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from .metrics import anomaly_correlation, latitude_weighted_rmse, skill_horizon_hours


logger = logging.getLogger(__name__)


def get_init_indices(
    times: np.ndarray, start: str, end: str, frequency_hours: int, num_steps: int, num_history_steps: int = 0
) -> np.ndarray:
    """
    Indices in `times` of initializations from `start` to `end` every `frequency_hours`, keeping only those followed
    by at least `num_steps` time steps and preceded by at least `num_history_steps`.
    """
    init_times = pd.date_range(start, end, freq=f"{frequency_hours}h").values
    position = {time: index for index, time in enumerate(times)}
    indices = np.array([position[time] for time in init_times if time in position])
    indices = indices[(indices + num_steps < len(times)) & (indices >= num_history_steps)]
    if len(indices) != len(init_times):
        logger.warning(
            f"Using {len(indices)} of {len(init_times)} initializations: the others lack {num_history_steps} previous "
            f"or {num_steps} following time steps in the data."
        )
    return indices


def get_targets(data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
    """Truth at each lead time, shape `(num_inits, num_steps, num_variables, latitude, longitude)`."""
    leads = np.arange(1, num_steps + 1)
    return data.states[init_indices[:, None] + leads[None, :]]


def get_forecast_variables(forecaster: Callable, data) -> list[str]:
    return list(getattr(forecaster, "variables", data.variables))


def get_num_history_steps(forecasters: list[Callable]) -> int:
    """Previous time steps needed by the forecaster that needs the most, so all are scored on the same cases."""
    return max(getattr(forecaster, "num_input_steps", 1) for forecaster in forecasters) - 1


def get_common_variables(forecasters: list[Callable], data) -> list[str]:
    """Variables forecast by every forecaster, in the order of `data.variables`."""
    common = set(data.variables)
    for forecaster in forecasters:
        common &= set(get_forecast_variables(forecaster, data))
    return [name for name in data.variables if name in common]


def evaluate_forecaster(
    forecaster: Callable,
    data,
    init_indices: np.ndarray,
    num_steps: int,
    batch_size: int,
    variables: list[str],
    climatology: Callable | None = None,
) -> dict:
    """
    Latitude-weighted RMSE and anomaly correlation of a forecaster, computed for each initialization and averaged.

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
        climatology (`Callable`, *optional*):
            Forecaster returning the climatology at each valid time, for the anomaly correlation. Without it, `acc` is
            not computed.

    Returns:
        `dict` with `lead_hours` of shape `(num_steps,)`, and `rmse` and `acc` (or `None`) of shape
        `(num_steps, len(variables))`.
    """
    forecast_variables = get_forecast_variables(forecaster, data)
    forecast_channels = [forecast_variables.index(name) for name in variables]
    target_channels = [data.variables.index(name) for name in variables]
    if climatology is not None:
        climatology_variables = get_forecast_variables(climatology, data)
        climatology_channels = [climatology_variables.index(name) for name in variables]
    rmse_per_init, acc_per_init = [], []
    for start in range(0, len(init_indices), batch_size):
        batch = init_indices[start : start + batch_size]
        forecast = forecaster(data, batch, num_steps)[:, :, forecast_channels]
        targets = get_targets(data, batch, num_steps)[:, :, target_channels]
        if forecast.shape != targets.shape:
            raise ValueError(f"Forecast shape {forecast.shape} does not match targets {targets.shape}.")
        rmse_per_init.append(latitude_weighted_rmse(forecast, targets, data.latitude_weights))
        if climatology is not None:
            climatology_values = climatology(data, batch, num_steps)[:, :, climatology_channels]
            acc_per_init.append(anomaly_correlation(forecast, targets, climatology_values, data.latitude_weights))
    return {
        "lead_hours": np.arange(1, num_steps + 1) * data.time_step_hours,
        "rmse": np.concatenate(rmse_per_init).mean(axis=0),
        "acc": np.concatenate(acc_per_init).mean(axis=0) if acc_per_init else None,
    }


def save_results(results: dict, variables: list[str], path: str | os.PathLike, acc_threshold: float):
    """
    Writes RMSE, ACC and skill horizon per variable as JSON.

    The skill horizon is the lead time in hours at which ACC falls below `acc_threshold` (`null` if it does not
    within the evaluated lead times).
    """
    output = {
        "lead_hours": results["lead_hours"].tolist(),
        "rmse": {name: results["rmse"][:, index].tolist() for index, name in enumerate(variables)},
    }
    if results["acc"] is not None:
        output["acc"] = {name: _nan_to_none(results["acc"][:, index]) for index, name in enumerate(variables)}
        output["acc_threshold"] = acc_threshold
        output["skill_horizon_hours"] = {
            name: skill_horizon_hours(results["lead_hours"], results["acc"][:, index], acc_threshold)
            for index, name in enumerate(variables)
        }
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(output, writer, indent=2)


def _nan_to_none(values: np.ndarray) -> list:
    return [None if np.isnan(value) else float(value) for value in values]


def plot_scores(
    all_results: dict[str, dict], variables: list[str], plot_variables: list[str], acc_threshold: float, path=None
):
    """
    Plots RMSE (top row) and ACC (bottom row) against lead time, one column per variable and one line per forecaster.

    Forecasters without ACC (such as the climatology) are left out of the bottom row.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(plot_variables), figsize=(5 * len(plot_variables), 7.5), squeeze=False, constrained_layout=True
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for column, name in enumerate(plot_variables):
        index = variables.index(name)
        for number, (forecaster_name, results) in enumerate(all_results.items()):
            style = {"marker": "o", "markersize": 3, "label": forecaster_name, "color": colors[number % len(colors)]}
            days = results["lead_hours"] / 24
            axes[0, column].plot(days, results["rmse"][:, index], **style)
            acc = results["acc"][:, index] if results["acc"] is not None else None
            if acc is not None and not np.isnan(acc).all():
                axes[1, column].plot(days, acc, **style)
        axes[0, column].set_title(f"{name} RMSE")
        axes[1, column].set_title(f"{name} ACC")
        axes[1, column].axhline(acc_threshold, color="k", linestyle="--", linewidth=0.8)
        axes[0, column].set_ylabel("RMSE")
        axes[1, column].set_ylabel("ACC")
        for row in range(2):
            axes[row, column].set_xlabel("lead time (days)")
            axes[row, column].grid(alpha=0.3)
    axes[0, 0].legend()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def default_model_name(checkpoint: str) -> str:
    """Name of a checkpoint in results: `checkpoints/epochs50/best` gives `epochs50`, `user/repo` gives `repo`."""
    parts = [part for part in str(checkpoint).rstrip("/").split("/") if part not in ("best", "last")]
    return parts[-1]
