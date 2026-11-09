"""Diagnostics of forecast errors over many initializations: bias maps, RMSE by latitude and zonal spectra."""

from collections.abc import Callable

import numpy as np

from .evaluate import get_forecast_variables, get_targets


def zonal_power_spectrum(fields: np.ndarray, latitude_weights: np.ndarray) -> np.ndarray:
    """
    Power spectrum along longitude, averaged over latitudes with area weights.

    Args:
        fields (`np.ndarray` of shape `(..., latitude, longitude)`):
            Fields on a regular grid, periodic in longitude.
        latitude_weights (`np.ndarray` of shape `(latitude,)`):
            Area weights with mean 1.

    Returns:
        `np.ndarray` of shape `(..., longitude // 2 + 1)`: power at zonal wavenumbers 0 to longitude // 2.
    """
    coefficients = np.fft.rfft(fields, axis=-1) / fields.shape[-1]
    power = np.abs(coefficients) ** 2
    return (power * latitude_weights[:, None]).mean(axis=-2)


def compute_diagnostics(
    forecaster: Callable,
    data,
    init_indices: np.ndarray,
    lead_steps: list[int],
    variables: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """
    Error statistics at selected lead times, accumulated over all initializations.

    Args:
        forecaster (`Callable`):
            `forecaster(data, init_indices, num_steps)`, see `nanoweather.evaluation.forecasters`.
        data (`WeatherData`):
            Split to evaluate on.
        init_indices (`np.ndarray`):
            Indices of the initializations in `data.times`.
        lead_steps (`list[int]`):
            Lead times, as numbers of time steps (1 is the first forecast step).
        variables (`list[str]`):
            Variables to analyze.
        batch_size (`int`):
            Number of initializations forecast at once.

    Returns:
        `dict` with, for each lead time and variable:
            - `bias` `(num_leads, num_variables, latitude, longitude)`: mean of forecast minus truth.
            - `rmse_by_latitude` `(num_leads, num_variables, latitude)`: RMSE over longitudes and initializations.
            - `forecast_spectrum`, `truth_spectrum` `(num_leads, num_variables, num_wavenumbers)`: mean zonal power.
    """
    forecast_variables = get_forecast_variables(forecaster, data)
    forecast_channels = [forecast_variables.index(name) for name in variables]
    target_channels = [data.variables.index(name) for name in variables]
    lead_index = np.array(lead_steps) - 1
    num_steps = max(lead_steps)

    sums = {}
    for start in range(0, len(init_indices), batch_size):
        batch = init_indices[start : start + batch_size]
        forecast = forecaster(data, batch, num_steps)[:, lead_index][:, :, forecast_channels].astype(np.float64)
        truth = get_targets(data, batch, num_steps)[:, lead_index][:, :, target_channels].astype(np.float64)
        error = forecast - truth
        batch_sums = {
            "bias": error.sum(axis=0),
            "squared_error_by_latitude": (error**2).mean(axis=-1).sum(axis=0),
            "forecast_spectrum": zonal_power_spectrum(forecast, data.latitude_weights).sum(axis=0),
            "truth_spectrum": zonal_power_spectrum(truth, data.latitude_weights).sum(axis=0),
        }
        for key, value in batch_sums.items():
            sums[key] = sums.get(key, 0) + value

    num_inits = len(init_indices)
    return {
        "lead_hours": np.array(lead_steps) * data.time_step_hours,
        "bias": sums["bias"] / num_inits,
        "rmse_by_latitude": np.sqrt(sums["squared_error_by_latitude"] / num_inits),
        "forecast_spectrum": sums["forecast_spectrum"] / num_inits,
        "truth_spectrum": sums["truth_spectrum"] / num_inits,
    }


def plot_diagnostics(diagnostics: dict, variables: list[str], latitude, longitude, land_sea_mask, units: list[str]):
    """
    Three figures: bias maps (variables x lead times), RMSE by latitude, and zonal spectra.

    Returns:
        `dict` of figure name to `matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    lead_days = diagnostics["lead_hours"] / 24
    num_leads, num_variables = len(lead_days), len(variables)
    figures = {}

    fig, axes = plt.subplots(
        num_variables,
        num_leads,
        figsize=(4.5 * num_leads, 2.6 * num_variables),
        squeeze=False,
        constrained_layout=True,
    )
    for row, name in enumerate(variables):
        bias = diagnostics["bias"][:, row]
        limit = np.abs(bias).max()
        for column in range(num_leads):
            ax = axes[row, column]
            mesh = ax.pcolormesh(longitude, latitude, bias[column], cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.contour(longitude, latitude, land_sea_mask, levels=[0.5], colors="k", linewidths=0.5)
            ax.set_title(f"{name} bias, day {lead_days[column]:g}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(mesh, ax=axes[row, -1], label=units[row])
    figures["bias"] = fig

    fig, axes = plt.subplots(
        1, num_variables, figsize=(4.5 * num_variables, 3.8), squeeze=False, constrained_layout=True
    )
    for column, name in enumerate(variables):
        ax = axes[0, column]
        for lead in range(num_leads):
            ax.plot(latitude, diagnostics["rmse_by_latitude"][lead, column], label=f"day {lead_days[lead]:g}")
        ax.set_title(f"{name} RMSE by latitude")
        ax.set_xlabel("latitude")
        ax.set_ylabel(units[column])
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    figures["rmse_by_latitude"] = fig

    fig, axes = plt.subplots(
        1, num_variables, figsize=(4.5 * num_variables, 3.8), squeeze=False, constrained_layout=True
    )
    wavenumbers = np.arange(diagnostics["truth_spectrum"].shape[-1])
    for column, name in enumerate(variables):
        ax = axes[0, column]
        ax.loglog(wavenumbers[1:], diagnostics["truth_spectrum"][0, column, 1:], "k", label="truth")
        for lead in range(num_leads):
            ax.loglog(
                wavenumbers[1:],
                diagnostics["forecast_spectrum"][lead, column, 1:],
                label=f"forecast day {lead_days[lead]:g}",
            )
        ax.set_title(f"{name} zonal power spectrum")
        ax.set_xlabel("zonal wavenumber")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend()
    figures["spectrum"] = fig
    return figures
