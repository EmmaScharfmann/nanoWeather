"""Loading prepared splits in memory and serving training samples."""

import os

import numpy as np
import torch
import xarray as xr

from ..processing.feature_extraction_nanoweather import NanoWeatherFeatureExtractor, get_time_features


def get_store_path(data_path: str | os.PathLike, store_name: str) -> str:
    """Path of the Zarr store, for a local directory or a Hub dataset repository id."""
    if os.path.isdir(data_path):
        return os.path.join(data_path, store_name)
    return f"hf://datasets/{data_path}/{store_name}"


def latitude_weights(latitude: np.ndarray) -> np.ndarray:
    """`cos(latitude)` normalized to mean 1."""
    weights = np.cos(np.deg2rad(latitude))
    return weights / weights.mean()


class WeatherData:
    """
    One split of a prepared dataset, loaded in memory.

    Args:
        data_path (`str` or `os.PathLike`):
            Local directory written by `scripts/prepare_data.py`, or a Hub dataset repository id.
        split (`str`):
            Group of the Zarr store to load, e.g. `"train"`.
        store_name (`str`):
            Name of the Zarr store in `data_path`.
        feature_extractor (`NanoWeatherFeatureExtractor`, *optional*):
            Normalization to apply. Loaded from `data_path` if not given.
        time_indices (`np.ndarray`, *optional*):
            Positions of the time steps to load, in increasing order, instead of the whole split.

    Attributes:
        states (`np.ndarray` of shape `(time, num_variables, latitude, longitude)`):
            States in physical units.
        normalized_states (`np.ndarray` of shape `(time, num_variables, latitude, longitude)`):
            States normalized with `feature_extractor`.
        static (`np.ndarray` of shape `(num_static_variables, latitude, longitude)`):
            Normalized static fields.
        latitude_weights (`np.ndarray` of shape `(latitude,)`):
            Area weights with mean 1.
        time_step_hours (`int`):
            Time between consecutive states (the smallest gap, when only some time steps are loaded).
    """

    def __init__(
        self,
        data_path: str | os.PathLike,
        split: str,
        store_name: str,
        feature_extractor: NanoWeatherFeatureExtractor | None = None,
        time_indices: np.ndarray | None = None,
    ):
        self.data_path = data_path
        self.store_path = get_store_path(data_path, store_name)
        self.feature_extractor = feature_extractor or NanoWeatherFeatureExtractor.from_pretrained(data_path)
        self.variables = self.feature_extractor.variables

        dataset = xr.open_zarr(self.store_path, group=split)
        if time_indices is not None:
            dataset = dataset.isel(time=np.asarray(time_indices))
        self.times = dataset.time.values
        self.latitude = dataset.latitude.values
        self.longitude = dataset.longitude.values
        self.states = np.stack([dataset[name].values for name in self.variables], axis=1).astype(np.float32)
        self.normalized_states = self.feature_extractor.normalize(self.states)
        self.time_step_hours = int(np.diff(self.times).min() / np.timedelta64(1, "h"))

        static = xr.open_zarr(self.store_path, group="static")
        static = np.stack([static[name].values for name in self.feature_extractor.static_variables]).astype(np.float32)
        self.static = self.feature_extractor.normalize_static(static)
        self.latitude_weights = latitude_weights(self.latitude).astype(np.float32)


class ForecastPairs(torch.utils.data.Dataset):
    """
    Training samples: normalized input states and the states that follow, as dicts of float32 tensors.

    Each sample has:
        - `states`: the state at t, `(num_variables, latitude, longitude)`.
        - `labels`: the next state, `(num_variables, latitude, longitude)`, or with `num_rollout_steps > 1` the next
          `num_rollout_steps` states, `(num_rollout_steps, num_variables, latitude, longitude)`.
        - `previous_states` when `num_input_steps > 1`: the states before t, oldest first,
          `(num_input_steps - 1, num_variables, latitude, longitude)`.
        - `time_features` when enabled: at t, `(num_time_features, latitude, longitude)`, or at the input time of each
          rollout step, `(num_rollout_steps, num_time_features, latitude, longitude)`.

    Args:
        data (`WeatherData`):
            The split to draw samples from.
        num_rollout_steps (`int`, *optional*, defaults to 1):
            Number of consecutive target states.
        time_features (`bool`, *optional*, defaults to `False`):
            Whether to include time features (sunlight at t and t + one step, time of year).
        num_input_steps (`int`, *optional*, defaults to 1):
            Number of consecutive input states, the last one at t.
    """

    def __init__(
        self, data: WeatherData, num_rollout_steps: int = 1, time_features: bool = False, num_input_steps: int = 1
    ):
        self.states = torch.from_numpy(data.normalized_states)
        self.num_rollout_steps = num_rollout_steps
        self.num_history_steps = num_input_steps - 1
        self.time_features = None
        if time_features:
            features = get_time_features(data.times, data.latitude, data.longitude, data.time_step_hours)
            self.time_features = torch.from_numpy(features)
        first, last = self.num_history_steps, len(data.times) - num_rollout_steps
        span_hours = (data.times[first + num_rollout_steps :] - data.times[: last - first]) / np.timedelta64(1, "h")
        expected_hours = data.time_step_hours * (self.num_history_steps + num_rollout_steps)
        self.indices = first + np.flatnonzero(span_hours == expected_hours)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t = self.indices[index]
        sample = {"states": self.states[t]}
        if self.num_rollout_steps == 1:
            sample["labels"] = self.states[t + 1]
        else:
            sample["labels"] = self.states[t + 1 : t + 1 + self.num_rollout_steps]
        if self.num_history_steps > 0:
            sample["previous_states"] = self.states[t - self.num_history_steps : t]
        if self.time_features is not None:
            if self.num_rollout_steps == 1:
                sample["time_features"] = self.time_features[t]
            else:
                sample["time_features"] = self.time_features[t : t + self.num_rollout_steps]
        return sample
