"""Forecasters: callables `forecaster(data, init_indices, num_steps)` returning forecasts in physical units.

The returned array has shape `(num_inits, num_steps, num_variables, latitude, longitude)`, where `init_indices` index
`data.times` and step `k` is valid at `data.times[init_index] + (k + 1) * data.time_step_hours`. Variables are those of
`forecaster.variables` if the forecaster has that attribute, otherwise those of `data.variables`.
"""

import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ..models.modeling_nanoweather import NanoWeatherForWeatherForecasting
from ..processing.feature_extraction_nanoweather import NanoWeatherFeatureExtractor, get_time_features


def persistence_forecaster(data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
    """Repeats the initial state at every lead time."""
    initial_states = data.states[init_indices]
    return np.repeat(initial_states[:, None], num_steps, axis=1)


def load_climatology(store_path: str, variables: list[str]) -> xr.Dataset:
    """Loads the `climatology` group of a prepared store, dims `(dayofyear, hour, latitude, longitude)`."""
    return xr.open_zarr(store_path, group="climatology")[variables]


class ClimatologyForecaster:
    """
    Forecasts the climatological value for the day of year and hour of each valid time.

    Args:
        climatology (`xr.Dataset`):
            Climatology with dims `(dayofyear, hour, latitude, longitude)`, with variables in the forecast's channel
            order.
    """

    def __init__(self, climatology: xr.Dataset):
        self.values = np.stack([climatology[name].values for name in climatology.data_vars], axis=2).astype(np.float32)
        self.hours = list(climatology.hour.values)

    def __call__(self, data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
        leads = np.arange(1, num_steps + 1)
        valid_times = pd.DatetimeIndex(data.times[(init_indices[:, None] + leads[None, :]).ravel()])
        day_index = valid_times.dayofyear.values - 1
        hour_index = np.array([self.hours.index(hour) for hour in valid_times.hour])
        forecast = self.values[day_index, hour_index]
        return forecast.reshape(len(init_indices), num_steps, *forecast.shape[1:])


class ModelForecaster:
    """
    Rolls out a trained model autoregressively from each initial state.

    Args:
        checkpoint (`str` or `os.PathLike`):
            Directory (or Hub model repository id) with `config.json`, `model.safetensors` and
            `preprocessor_config.json`.
        static (`np.ndarray` of shape `(num_static_variables, latitude, longitude)`):
            Normalized static fields.
        device (`str`, *optional*):
            Defaults to `"cuda"` if available.
    """

    def __init__(self, checkpoint: str | os.PathLike, static: np.ndarray, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = NanoWeatherForWeatherForecasting.from_pretrained(checkpoint).to(self.device)
        self.feature_extractor = NanoWeatherFeatureExtractor.from_pretrained(checkpoint, repo_type="model")
        self.variables = self.model.config.variables
        self.static = torch.from_numpy(static).to(self.device)

    def __call__(self, data, init_indices: np.ndarray, num_steps: int) -> np.ndarray:
        missing = [name for name in self.variables if name not in data.variables]
        if missing:
            raise ValueError(f"The model needs variables missing from the data: {missing}")
        channels = [data.variables.index(name) for name in self.variables]
        initial_states = data.states[init_indices][:, channels]
        states = torch.from_numpy(self.feature_extractor.normalize(initial_states)).to(self.device)
        time_features = None
        if self.model.config.num_time_features > 0:
            step = np.timedelta64(data.time_step_hours, "h")
            step_times = data.times[init_indices][:, None] + np.arange(num_steps) * step
            features = get_time_features(step_times.ravel(), data.latitude, data.longitude, data.time_step_hours)
            features = features.reshape(len(init_indices), num_steps, *features.shape[1:])
            time_features = torch.from_numpy(features).to(self.device)
        forecast = self.model.rollout(states, static=self.static, num_steps=num_steps, time_features=time_features)
        return self.feature_extractor.denormalize(forecast.float().cpu().numpy())
