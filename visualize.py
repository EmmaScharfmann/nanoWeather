# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualizes one forecast against ERA5: an animation over lead times and a snapshot figure.

Each row is a variable, with three columns: ERA5, forecast, and forecast minus ERA5. Coastlines are drawn from the
land-sea mask, so no cartography package is needed (they are coarse at 5.625 deg).

Usage:
    uv run visualize.py --checkpoint checkpoints/v0/best --init_time 2020-07-01T00
    uv run visualize.py --checkpoint checkpoints/v0/best --init_time 2020-01-15T00 --num_steps 40 --snapshot_days 1 5 10
"""

import argparse
import logging
import os

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from data import WeatherData, get_store_path
from evaluate import TIME_STEP_HOURS, ModelForecaster


logger = logging.getLogger(__name__)

# Variable -> (conversion from stored units to displayed units, displayed units, colormap)
DISPLAY = {
    "z500": (lambda values: values / 9.80665, "m", "viridis"),  # geopotential -> geopotential height
    "t850": (lambda values: values - 273.15, "°C", "RdYlBu_r"),
    "t2m": (lambda values: values - 273.15, "°C", "RdYlBu_r"),
    "u10": (lambda values: values, "m/s", "PuOr_r"),
    "v10": (lambda values: values, "m/s", "PuOr_r"),
    "msl": (lambda values: values / 100, "hPa", "viridis"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, default="data/nanoweather_v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--init_time", type=str, default="2020-07-01T00", help="Initialization time in the test split."
    )
    parser.add_argument("--num_steps", type=int, default=20, help="Number of 6h steps.")
    parser.add_argument("--variables", type=str, nargs="+", default=["z500", "t2m"])
    parser.add_argument("--snapshot_days", type=float, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def get_forecast_and_truth(data: WeatherData, checkpoint: str, init_time: str, num_steps: int):
    """Returns forecast and ERA5 truth, each `(num_steps, num_variables, latitude, longitude)` in stored units."""
    init_index = np.flatnonzero(data.times == np.datetime64(pd.Timestamp(init_time)))
    if len(init_index) == 0 or init_index[0] + num_steps >= len(data.times):
        raise ValueError(f"{init_time} with {num_steps} steps is not covered by the split.")
    init_index = init_index[0]
    forecaster = ModelForecaster(checkpoint, data.static)
    forecast = forecaster(data, np.array([init_index]), num_steps)[0]
    truth = data.states[init_index + 1 : init_index + 1 + num_steps]
    return forecast, truth


def get_land_sea_mask(data_path: str) -> np.ndarray:
    """Raw land-sea mask (fraction of land in each cell), for drawing coastlines."""
    return xr.open_zarr(get_store_path(data_path), group="static")["land_sea_mask"].values


class ForecastFigure:
    r"""
    Grid of maps: one row per variable, columns ERA5 / forecast / forecast minus ERA5.

    Color limits are fixed per variable over all lead times (from ERA5 for the fields, from the largest error for the
    difference), so frames of the animation are comparable.

    Args:
        data ([`WeatherData`]): Split the forecast comes from, for coordinates and variable order.
        forecast (`np.ndarray` of shape `(num_steps, num_variables, latitude, longitude)`): Forecast in stored units.
        truth (`np.ndarray` of shape `(num_steps, num_variables, latitude, longitude)`): ERA5 in stored units.
        variables (`list[str]`): Variables to show, one per row.
        land_sea_mask (`np.ndarray` of shape `(latitude, longitude)`): For coastlines.
        init_time (`str`): Initialization time, for titles.
    """

    def __init__(self, data, forecast, truth, variables, land_sea_mask, init_time: str):
        self.data, self.variables, self.land_sea_mask = data, variables, land_sea_mask
        self.init_time = pd.Timestamp(init_time)
        self.fields = {}
        for name in variables:
            convert, _, _ = DISPLAY[name]
            index = data.variables.index(name)
            self.fields[name] = (convert(truth[:, index]), convert(forecast[:, index]))

    def draw(self, step: int, fig=None, axes=None):
        """Draws lead time `step` (0-based) on `axes`, creating the figure if needed. Returns (fig, axes)."""
        if fig is None:
            fig, axes = plt.subplots(
                len(self.variables), 3, figsize=(15, 3.2 * len(self.variables)), squeeze=False, constrained_layout=True
            )
        for row, name in enumerate(self.variables):
            truth, forecast = self.fields[name]
            _, units, cmap = DISPLAY[name]
            limits = (truth.min(), truth.max())
            error_limit = np.abs(forecast - truth).max()
            panels = [
                (truth[step], "ERA5", cmap, limits),
                (forecast[step], "forecast", cmap, limits),
                (forecast[step] - truth[step], "forecast - ERA5", "RdBu_r", (-error_limit, error_limit)),
            ]
            for column, (values, title, colormap, (vmin, vmax)) in enumerate(panels):
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
                if not getattr(ax, "_has_colorbar", False):
                    fig.colorbar(mesh, ax=ax, label=units if column < 2 else f"{units} (error)", shrink=0.9)
                    ax._has_colorbar = True
        lead_hours = (step + 1) * TIME_STEP_HOURS
        valid_time = self.init_time + pd.Timedelta(hours=lead_hours)
        fig.suptitle(
            f"init {self.init_time:%Y-%m-%d %H}h | lead {lead_hours}h ({lead_hours / 24:.2g} days) | "
            f"valid {valid_time:%Y-%m-%d %H}h"
        )
        return fig, axes

    def save_animation(self, path: str, fps: int = 4):
        fig, axes = self.draw(0)
        num_steps = next(iter(self.fields.values()))[0].shape[0]
        animation = FuncAnimation(fig, lambda step: self.draw(step, fig, axes), frames=num_steps)
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)

    def save_snapshots(self, path: str, steps: list[int]):
        """One figure per lead time, stacked vertically in a single image."""
        figures = []
        for step in steps:
            fig, _ = self.draw(step)
            fig.canvas.draw()
            figures.append(np.asarray(fig.canvas.buffer_rgba()))
            plt.close(fig)
        plt.imsave(path, np.concatenate(figures, axis=0))


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    data = WeatherData(args.data_path, split="test")
    forecast, truth = get_forecast_and_truth(data, args.checkpoint, args.init_time, args.num_steps)
    figure = ForecastFigure(data, forecast, truth, args.variables, get_land_sea_mask(args.data_path), args.init_time)

    stem = f"forecast_{pd.Timestamp(args.init_time):%Y%m%d%H}"
    animation_path = os.path.join(args.output_dir, f"{stem}.gif")
    figure.save_animation(animation_path, fps=args.fps)
    logger.info(f"Animation written to {animation_path}")

    steps = [round(days * 24 / TIME_STEP_HOURS) - 1 for days in args.snapshot_days]
    steps = [step for step in steps if 0 <= step < args.num_steps]
    snapshot_path = os.path.join(args.output_dir, f"{stem}_snapshots.png")
    figure.save_snapshots(snapshot_path, steps)
    logger.info(f"Snapshots at {args.snapshot_days} days written to {snapshot_path}")


if __name__ == "__main__":
    main()