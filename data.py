"""Loading the dataset for training and evaluation."""

import os

import numpy as np
import torch
import xarray as xr

from feature_extraction_nanoweather import NanoWeatherFeatureExtractor, get_time_features


STORE_NAME = "nanoweather_v0.zarr"
TIME_STEP_HOURS = 6


def get_store_path(data_path: str | os.PathLike) -> str:
    """Returns the Zarr store path for a local directory or a Hub dataset repository id."""
    if os.path.isdir(data_path):
        return os.path.join(data_path, STORE_NAME)
    return f"hf://datasets/{data_path}/{STORE_NAME}"


def latitude_weights(latitude: np.ndarray) -> np.ndarray:
    """Returns `cos(latitude)` normalized to mean 1, used in both the loss and the metrics."""
    weights = np.cos(np.deg2rad(latitude))
    return weights / weights.mean()


class WeatherData:
    """
    One split of the nanoWeather dataset, loaded in memory.

    At 64x32 with 6 variables, the train split is about 1 GB.

    Args:
        data_path (`str` or `os.PathLike`):
            Local directory written by `prepare_data.py`, or a Hub dataset repository id.
        split (`str`):
            One of `"train"`, `"val"` or `"test"`.
        feature_extractor ([`NanoWeatherFeatureExtractor`], *optional*):
            Normalization to apply. Loaded from `data_path` if not given.

    Attributes:
        states (`np.ndarray` of shape `(time, num_variables, latitude, longitude)`):
            States in physical units.
        normalized_states (`np.ndarray` of shape `(time, num_variables, latitude, longitude)`):
            States normalized with `feature_extractor`.
        static (`np.ndarray` of shape `(num_static_variables, latitude, longitude)`):
            Normalized static fields.
        latitude_weights (`np.ndarray` of shape `(latitude,)`):
            Area weights, mean 1.
    """

    def __init__(
        self,
        data_path: str | os.PathLike,
        split: str,
        feature_extractor: NanoWeatherFeatureExtractor | None = None,
    ):
        self.data_path = data_path
        self.feature_extractor = feature_extractor or NanoWeatherFeatureExtractor.from_pretrained(data_path)
        self.variables = self.feature_extractor.variables
        store_path = get_store_path(data_path)

        dataset = xr.open_zarr(store_path, group=split)
        self.times = dataset.time.values
        self.latitude = dataset.latitude.values
        self.longitude = dataset.longitude.values
        self.states = np.stack([dataset[name].values for name in self.variables], axis=1).astype(np.float32)
        self.normalized_states = self.feature_extractor.normalize(self.states)

        static = xr.open_zarr(store_path, group="static")
        static = np.stack([static[name].values for name in self.feature_extractor.static_variables]).astype(np.float32)
        self.static = self.feature_extractor.normalize_static(static)

        self.latitude_weights = latitude_weights(self.latitude).astype(np.float32)

    def climatology(self) -> xr.Dataset:
        """Returns the ERA5 climatology (1990-2019, from WeatherBench 2), dims `(dayofyear, hour, latitude, longitude)`."""
        return xr.open_zarr(get_store_path(self.data_path), group="climatology")[self.variables]


class ForecastPairs(torch.utils.data.Dataset):
    r"""
    Pairs of normalized states (state at t, state at t + lead), as dicts of float32 tensors.

    Each sample has `states` and `labels` of shape `(num_variables, latitude, longitude)`, and `time_features` of
    shape `(num_time_features, latitude, longitude)` when enabled. The keys match the arguments of
    [`NanoWeatherForWeatherForecasting.forward`], so a batch can be passed as `model(**batch, static=static)`.

    Args:
        data ([`WeatherData`]):
            The split to draw pairs from.
        lead_steps (`int`, *optional*, defaults to 1):
            Number of 6h steps between input and target.
        time_features (`bool`, *optional*, defaults to `False`):
            Whether to include time features (sunlight at t and t + 6h, time of year).
    """

    def __init__(self, data: WeatherData, lead_steps: int = 1, time_features: bool = False):
        self.states = torch.from_numpy(data.normalized_states)
        self.lead_steps = lead_steps
        self.time_features = None
        if time_features:
            features = get_time_features(data.times, data.latitude, data.longitude, TIME_STEP_HOURS)
            self.time_features = torch.from_numpy(features)
        
        # Only keep pairs exactly `lead_steps * 6h` apart, in case the time axis has gaps.
        hours_apart = (data.times[lead_steps:] - data.times[:-lead_steps]) / np.timedelta64(1, "h")
        self.indices = np.flatnonzero(hours_apart == TIME_STEP_HOURS * lead_steps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        time_index = self.indices[index]
        sample = {"states": self.states[time_index], "labels": self.states[time_index + self.lead_steps]}
        if self.time_features is not None:
            sample["time_features"] = self.time_features[time_index]
        return sample