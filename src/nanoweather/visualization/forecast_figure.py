"""Maps of a forecast against the truth: one row per variable, columns truth, forecast and difference."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from ..evaluation.forecasters import ModelForecaster


def get_forecast_and_truth(data, checkpoint: str, init_time: str, num_steps: int):
    """
    Forecast from `init_time` and the matching truth.

    Returns:
        `forecast` and `truth`, each `(num_steps, num_variables, latitude, longitude)`, and the names of their
        variables (those of the model).
    """
    init_index = np.flatnonzero(data.times == np.datetime64(pd.Timestamp(init_time)))
    forecaster = ModelForecaster(checkpoint, data.static)
    history = forecaster.num_input_steps - 1
    if len(init_index) == 0 or init_index[0] + num_steps >= len(data.times) or init_index[0] < history:
        raise ValueError(f"{init_time} with {history} previous and {num_steps} following steps is not in the split.")
    init_index = init_index[0]
    forecast = forecaster(data, np.array([init_index]), num_steps)[0]
    channels = [data.variables.index(name) for name in forecaster.variables]
    truth = data.states[init_index + 1 : init_index + 1 + num_steps][:, channels]
    return forecast, truth, forecaster.variables


def load_land_sea_mask(store_path: str) -> np.ndarray:
    """Fraction of land in each grid cell, from the `static` group of a prepared store."""
    return xr.open_zarr(store_path, group="static")["land_sea_mask"].values


class ForecastFigure:
    """
    Grid of maps with one row per variable and three columns: truth, forecast, forecast minus truth.

    Color limits are fixed per variable over all lead times, so frames are comparable. Coastlines are the 0.5 contour
    of the land-sea mask.

    Args:
        data (`WeatherData`):
            Split the forecast comes from, for coordinates and time step.
        forecast (`np.ndarray` of shape `(num_steps, num_variables, latitude, longitude)`):
            Forecast in physical units.
        truth (`np.ndarray` of shape `(num_steps, num_variables, latitude, longitude)`):
            Truth in physical units.
        array_variables (`list[str]`):
            Names of the variables of `forecast` and `truth`, in channel order.
        variables (`list[str]`):
            Variables to show, one per row.
        land_sea_mask (`np.ndarray` of shape `(latitude, longitude)`):
            Fraction of land in each grid cell.
        init_time (`str`):
            Initialization time, for titles.
        display (`dict[str, dict]`):
            Per variable: `scale`, `offset`, `units` and `cmap`. Shown values are `value * scale + offset`.
    """

    def __init__(
        self,
        data,
        forecast,
        truth,
        array_variables,
        variables,
        land_sea_mask,
        init_time: str,
        display: dict[str, dict],
    ):
        self.data, self.variables, self.land_sea_mask, self.display = data, variables, land_sea_mask, display
        self.init_time = pd.Timestamp(init_time)
        self.num_steps = forecast.shape[0]
        self.fields = {}
        for name in variables:
            index = array_variables.index(name)
            scale, offset = display[name]["scale"], display[name]["offset"]
            self.fields[name] = (truth[:, index] * scale + offset, forecast[:, index] * scale + offset)

    def draw(self, step: int, fig=None, axes=None):
        """Draws lead time `step` (0-based), creating the figure if `fig` is None. Returns `(fig, axes)`."""
        if fig is None:
            fig, axes = plt.subplots(
                len(self.variables), 3, figsize=(15, 3.2 * len(self.variables)), squeeze=False, constrained_layout=True
            )
        for row, name in enumerate(self.variables):
            truth, forecast = self.fields[name]
            units, cmap = self.display[name]["units"], self.display[name]["cmap"]
            error_limit = np.abs(forecast - truth).max()
            panels = [
                (truth[step], "truth", cmap, (truth.min(), truth.max()), units),
                (forecast[step], "forecast", cmap, (truth.min(), truth.max()), units),
                (
                    forecast[step] - truth[step],
                    "forecast - truth",
                    "RdBu_r",
                    (-error_limit, error_limit),
                    f"{units} (error)",
                ),
            ]
            for column, (values, title, colormap, (vmin, vmax), label) in enumerate(panels):
                ax = axes[row, column]
                ax.clear()
                mesh = ax.pcolormesh(
                    self.data.longitude, self.data.latitude, values, cmap=colormap, vmin=vmin, vmax=vmax
                )
                ax.contour(
                    self.data.longitude,
                    self.data.latitude,
                    self.land_sea_mask,
                    levels=[0.5],
                    colors="k",
                    linewidths=0.6,
                )
                ax.set_title(f"{name} {title}")
                ax.set_xticks([])
                ax.set_yticks([])
                if not getattr(ax, "has_colorbar", False):
                    fig.colorbar(mesh, ax=ax, label=label, shrink=0.9)
                    ax.has_colorbar = True
        lead_hours = (step + 1) * self.data.time_step_hours
        valid_time = self.init_time + pd.Timedelta(hours=lead_hours)
        fig.suptitle(
            f"init {self.init_time:%Y-%m-%d %H}h | lead {lead_hours}h ({lead_hours / 24:.2g} days) | "
            f"valid {valid_time:%Y-%m-%d %H}h"
        )
        return fig, axes

    def save_animation(self, path: str, fps: int):
        fig, axes = self.draw(0)
        animation = FuncAnimation(fig, lambda step: self.draw(step, fig, axes), frames=self.num_steps)
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)

    def save_snapshots(self, path: str, steps: list[int]):
        """Draws each lead time in `steps` and stacks the figures vertically in one image."""
        images = []
        for step in steps:
            fig, _ = self.draw(step)
            fig.canvas.draw()
            images.append(np.asarray(fig.canvas.buffer_rgba()))
            plt.close(fig)
        plt.imsave(path, np.concatenate(images, axis=0))
