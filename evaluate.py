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
"""Evaluates forecasts on the test split with latitude-weighted RMSE per variable and lead time.

Protocol (close to WeatherBench 2 at this resolution):
    - initializations every 12h (00 and 12 UTC) in 2020
    - lead times from 6h to `num_steps * 6h` (default 20 steps, 5 days)
    - truth is ERA5, RMSE is computed per initialization over the globe, then averaged over initializations

A forecaster is any callable `forecaster(data, init_indices, num_steps)` returning an array of shape
`(num_inits, num_steps, num_variables, latitude, longitude)` in physical units, where `init_indices` index
`data.times`. Persistence, climatology and trained nanoWeather models are scored by the same code.

Usage:
    uv run evaluate.py --data_path data/nanoweather_v0 --output_dir results
    uv run evaluate.py --data_path data/nanoweather_v0 --output_dir results --checkpoint checkpoints/v0/best
    uv run evaluate.py --output_dir results/round1 --checkpoint checkpoints/v0/best checkpoints/epochs50/best \\
        --model_name v0 epochs50
"""

import argparse
import json
import logging
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from data import WeatherData
from feature_extraction_nanoweather import NanoWeatherFeatureExtractor, get_time_features
from modeling_nanoweather import NanoWeatherForWeatherForecasting


logger = logging.getLogger(__name__)

TIME_STEP_HOURS = 6
INIT_START = "2020-01-01T00"
INIT_END = "2020-12-31T12"
INIT_FREQUENCY_HOURS = 12

Forecaster = Callable[..., np.ndarray]


def latitude_weighted_rmse(forecast: np.ndarray, truth: np.ndarray, latitude_weights: np.ndarray) -> np.ndarray:
    """
    Latitude-weighted RMSE over the globe, computed per field.

    Args:
        forecast (`np.ndarray` of shape `(..., latitude, longitude)`):
            Forecast in physical units.
        truth (`np.ndarray` of shape `(..., latitude, longitude)`):
            Truth in physical units.
        latitude_weights (`np.ndarray` of shape `(latitude,)`):
            Area weights with mean 1.

    Returns:
        `np.ndarray` of shape `(...)`: one RMSE per leading index.
    """
    squared_error = (forecast.astype(np.float64) - truth) ** 2
    return np.sqrt((squared_error * latitude_weights[:, None]).mean(axis=(-2, -1)))


def get_init_indices(times: np.ndarray, num_steps: int) -> np.ndarray:
    """Indices of initializations in `times` every 12h through 2020, keeping only those with all targets available."""
    init_times = pd.date_range(INIT_START, INIT_END, freq=f"{INIT_FREQUENCY_HOURS}h").values
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


def persistence_forecaster(data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
    """Forecast = initial state, at every lead time."""
    initial_states = data.states[init_indices]
    return np.repeat(initial_states[:, None], num_steps, axis=1)


class ClimatologyForecaster:
    r"""
    Forecast = WeatherBench 2 ERA5 climatology (1990-2019) for the day of year and hour of each valid time.

    Args:
        climatology (`xr.Dataset`):
            Climatology with dims `(dayofyear, hour, latitude, longitude)`, as returned by `WeatherData.climatology()`.
        variables (`list[str]`):
            Variables in channel order.
    """

    def __init__(self, climatology, variables: list[str]):
        # (dayofyear, hour, num_variables, latitude, longitude), about 70 MB at 5.625 deg
        self.values = np.stack([climatology[name].values for name in variables], axis=2).astype(np.float32)
        self.hours = list(climatology.hour.values)

    def __call__(self, data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
        leads = np.arange(1, num_steps + 1)
        valid_times = pd.DatetimeIndex(data.times[(init_indices[:, None] + leads[None, :]).ravel()])
        day_index = valid_times.dayofyear.values - 1
        hour_index = np.array([self.hours.index(hour) for hour in valid_times.hour])
        forecast = self.values[day_index, hour_index]
        return forecast.reshape(len(init_indices), num_steps, *forecast.shape[1:])


class ModelForecaster:
    r"""
    Forecasts with a trained nanoWeather model, rolled out autoregressively from each initial state.

    Args:
        checkpoint (`str` or `os.PathLike`):
            Directory written by `train.py` (or a Hub model repository id) with `config.json`, `model.safetensors`
            and `preprocessor_config.json`.
        static (`np.ndarray` of shape `(num_static_variables, latitude, longitude)`):
            Normalized static fields, as in `WeatherData.static`.
        device (`str`, *optional*):
            Defaults to `"cuda"` if available.
    """

    def __init__(self, checkpoint: str | os.PathLike, static: np.ndarray, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = NanoWeatherForWeatherForecasting.from_pretrained(checkpoint).to(self.device)
        self.feature_extractor = NanoWeatherFeatureExtractor.from_pretrained(checkpoint, repo_type="model")
        self.static = torch.from_numpy(static).to(self.device)

    def __call__(self, data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
        # Normalize with the checkpoint's statistics: the ones the model was trained with.
        states = torch.from_numpy(self.feature_extractor.normalize(data.states[init_indices])).to(self.device)
        time_features = None
        if self.model.config.num_time_features > 0:
            # Input time of each step: init, init + 6h, ..., init + (num_steps - 1) * 6h
            step_times = data.times[init_indices][:, None] + np.arange(num_steps) * np.timedelta64(
                TIME_STEP_HOURS, "h"
            )
            features = get_time_features(step_times.ravel(), data.latitude, data.longitude, TIME_STEP_HOURS)
            features = features.reshape(len(init_indices), num_steps, *features.shape[1:])
            time_features = torch.from_numpy(features).to(self.device)
        forecast = self.model.rollout(states, static=self.static, num_steps=num_steps, time_features=time_features)
        return self.feature_extractor.denormalize(forecast.float().cpu().numpy())


def evaluate_forecaster(
    forecaster: Forecaster, data, num_steps: int = 20, batch_size: int = 64
) -> dict[str, np.ndarray]:
    """
    Scores a forecaster on all initializations of `data`.

    Args:
        forecaster (`Callable`):
            `forecaster(data, init_indices, num_steps)` returning forecasts in physical units.
        data ([`WeatherData`]):
            The test split.
        num_steps (`int`, *optional*, defaults to 20):
            Number of 6h steps to forecast.
        batch_size (`int`, *optional*, defaults to 64):
            Number of initializations forecast at once, to bound memory.

    Returns:
        `dict` with `lead_hours` of shape `(num_steps,)` and `rmse` of shape `(num_steps, num_variables)`.
    """
    init_indices = get_init_indices(data.times, num_steps)
    rmse_per_init = []
    for start in range(0, len(init_indices), batch_size):
        batch = init_indices[start : start + batch_size]
        forecast = forecaster(data, batch, num_steps)
        targets = get_targets(data, batch, num_steps)
        if forecast.shape != targets.shape:
            raise ValueError(f"Forecast shape {forecast.shape} does not match targets {targets.shape}.")
        rmse_per_init.append(latitude_weighted_rmse(forecast, targets, data.latitude_weights))
    rmse = np.concatenate(rmse_per_init).mean(axis=0)  # (num_steps, num_variables)
    return {"lead_hours": np.arange(1, num_steps + 1) * TIME_STEP_HOURS, "rmse": rmse}


def save_results(results: dict[str, np.ndarray], variables: list[str], path: str | os.PathLike):
    """Saves results as JSON: `{"lead_hours": [...], "rmse": {variable: [...]}}`."""
    output = {
        "lead_hours": results["lead_hours"].tolist(),
        "rmse": {name: results["rmse"][:, index].tolist() for index, name in enumerate(variables)},
    }
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(output, writer, indent=2)


def plot_rmse(all_results: dict[str, dict], variables: list[str], plot_variables=("z500", "t850", "t2m"), path=None):
    """Plots RMSE against lead time for each forecaster, one panel per variable."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(plot_variables), figsize=(5 * len(plot_variables), 4), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), plot_variables):
        index = variables.index(name)
        for forecaster_name, results in all_results.items():
            ax.plot(
                results["lead_hours"] / 24, results["rmse"][:, index], marker="o", markersize=3, label=forecaster_name
            )
        ax.set_title(name)
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def default_model_name(checkpoint: str) -> str:
    """`checkpoints/epochs50/best` -> `epochs50`, `user/repo` -> `repo`."""
    parts = [part for part in str(checkpoint).rstrip("/").split("/") if part not in ("best", "last")]
    return parts[-1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, default="data/nanoweather_v0")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument(
        "--checkpoint", type=str, nargs="+", default=[], help="One or more trained model directories or Hub repo ids."
    )
    parser.add_argument(
        "--model_name", type=str, nargs="+", default=None, help="Names in the results, one per checkpoint."
    )
    parser.add_argument("--no_baselines", action="store_true", help="Skip persistence and climatology.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    data = WeatherData(args.data_path, split="test")
    forecasters = {}
    if not args.no_baselines:
        forecasters["persistence"] = persistence_forecaster
        forecasters["climatology"] = ClimatologyForecaster(load_climatology(args.data_path, data.variables), data.variables)
    model_names = args.model_name or [default_model_name(checkpoint) for checkpoint in args.checkpoint]
    if len(model_names) != len(args.checkpoint):
        raise ValueError(f"Got {len(args.checkpoint)} checkpoints but {len(model_names)} model names.")
    for name, checkpoint in zip(model_names, args.checkpoint):
        forecasters[name] = ModelForecaster(checkpoint, data.static)

    all_results = {}
    for name, forecaster in forecasters.items():
        all_results[name] = evaluate_forecaster(forecaster, data, num_steps=args.num_steps)
        save_results(all_results[name], data.variables, os.path.join(args.output_dir, f"{name}.json"))
        day_index = min(args.num_steps, 4 * 3) - 1
        summary = ", ".join(
            f"{variable} {all_results[name]['rmse'][day_index, index]:.4g}"
            for index, variable in enumerate(data.variables)
        )
        logger.info(f"{name} RMSE at {all_results[name]['lead_hours'][day_index]}h: {summary}")

    plot_rmse(all_results, data.variables, path=os.path.join(args.output_dir, "rmse.png"))
    logger.info(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()