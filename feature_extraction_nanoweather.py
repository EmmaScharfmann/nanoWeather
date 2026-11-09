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
"""Feature extractor for nanoWeather: normalization of weather states and static fields."""

import copy
import json
import logging
import os

import numpy as np


logger = logging.getLogger(__name__)

FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
NUM_TIME_FEATURES = 4


def get_time_features(
    times: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, time_step_hours: int = 6
) -> np.ndarray:
    """
    Time features for each input time, as fields on the grid.

    Channels:
        0. Incoming sunlight at time t: cosine of the solar zenith angle, clipped at 0 (night). Depends on latitude,
           longitude (local time) and day of year.
        1. Same at t + `time_step_hours`, the end of the step being predicted.
        2. sin(2 pi day_of_year / 365.25), constant over the grid.
        3. cos(2 pi day_of_year / 365.25), constant over the grid.

    The solar position uses a simple declination formula and ignores the equation of time (error under 20 minutes,
    negligible at 5.625 deg).

    Args:
        times (`np.ndarray` of `datetime64`, shape `(num_times,)`):
            Input times (UTC).
        latitude (`np.ndarray` of shape `(latitude,)`):
            Latitudes in degrees.
        longitude (`np.ndarray` of shape `(longitude,)`):
            Longitudes in degrees east.
        time_step_hours (`int`, *optional*, defaults to 6):
            Model time step.

    Returns:
        `np.ndarray` of shape `(num_times, 4, latitude, longitude)`, float32.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    latitude_radians = np.deg2rad(latitude)[None, :, None]

    def insolation(at_times: np.ndarray) -> np.ndarray:
        day_of_year = _day_of_year(at_times)[:, None, None]
        hour_utc = ((at_times - at_times.astype("datetime64[D]")) / np.timedelta64(1, "h"))[:, None, None]
        declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
        local_solar_hour = hour_utc + longitude[None, None, :] / 15.0
        hour_angle = np.deg2rad(15.0 * (local_solar_hour - 12.0))
        cos_zenith = np.sin(latitude_radians) * np.sin(declination) + np.cos(latitude_radians) * np.cos(
            declination
        ) * np.cos(hour_angle)
        return np.clip(cos_zenith, 0.0, None)

    num_times, grid_shape = len(times), (len(latitude), len(longitude))
    angle = 2 * np.pi * _day_of_year(times) / 365.25
    features = np.empty((num_times, NUM_TIME_FEATURES, *grid_shape), dtype=np.float32)
    features[:, 0] = insolation(times)
    features[:, 1] = insolation(times + np.timedelta64(time_step_hours, "h"))
    features[:, 2] = np.sin(angle)[:, None, None]
    features[:, 3] = np.cos(angle)[:, None, None]
    return features


def _day_of_year(times: np.ndarray) -> np.ndarray:
    """Day of year starting at 1, as float."""
    days = times.astype("datetime64[D]")
    return (days - days.astype("datetime64[Y]")).astype(np.float64) + 1.0


class NanoWeatherFeatureExtractor:
    r"""
    Normalizes weather states with per-variable statistics computed on the training years only.

    States are arrays or tensors of shape `(..., num_variables, latitude, longitude)`, with variables in the order of
    `variables`. Static fields are arrays of shape `(num_static_variables, latitude, longitude)`.

    Args:
        variables (`list[str]`):
            Names of the prognostic variables, in channel order.
        mean (`list[float]`):
            Mean of each variable over the training years.
        std (`list[float]`):
            Standard deviation of each variable over the training years.
        diff_std (`list[float]`, *optional*):
            Standard deviation of the 6h change of each variable, for tendency normalization.
        units (`list[str]`, *optional*):
            Physical units of each variable, for reference only.
        static_variables (`list[str]`, *optional*, defaults to `["land_sea_mask", "orography"]`):
            Names of the static fields, in channel order.
        static_mean (`list[float]`, *optional*):
            Mean of each static field. Defaults to 0 for every field.
        static_std (`list[float]`, *optional*):
            Standard deviation of each static field. Defaults to 1 for every field.
    """

    feature_extractor_type = "NanoWeatherFeatureExtractor"

    def __init__(
        self,
        variables: list[str],
        mean: list[float],
        std: list[float],
        diff_std: list[float] | None = None,
        units: list[str] | None = None,
        static_variables: list[str] | None = None,
        static_mean: list[float] | None = None,
        static_std: list[float] | None = None,
    ):
        num_variables = len(variables)
        if len(mean) != num_variables or len(std) != num_variables:
            raise ValueError(
                f"`mean` and `std` must have one value per variable ({num_variables}), got {len(mean)} and {len(std)}."
            )

        self.variables = list(variables)
        self.mean = list(mean)
        self.std = list(std)
        self.diff_std = list(diff_std) if diff_std is not None else None
        self.units = list(units) if units is not None else None

        self.static_variables = (
            list(static_variables) if static_variables is not None else ["land_sea_mask", "orography"]
        )
        num_static = len(self.static_variables)
        self.static_mean = list(static_mean) if static_mean is not None else [0.0] * num_static
        self.static_std = list(static_std) if static_std is not None else [1.0] * num_static

    def __call__(self, states):
        """Normalizes `states`. Alias of [`~NanoWeatherFeatureExtractor.normalize`]."""
        return self.normalize(states)

    def normalize(self, states):
        """Maps states in physical units to `(states - mean) / std`."""
        mean, std = self._as_channels(self.mean, states), self._as_channels(self.std, states)
        return (states - mean) / std

    def denormalize(self, states):
        """Maps normalized states back to physical units."""
        mean, std = self._as_channels(self.mean, states), self._as_channels(self.std, states)
        return states * std + mean

    def normalize_static(self, static):
        """Normalizes static fields of shape `(num_static_variables, latitude, longitude)`."""
        mean = self._as_channels(self.static_mean, static)
        std = self._as_channels(self.static_std, static)
        return (static - mean) / std

    @staticmethod
    def _as_channels(values, like):
        """Reshapes per-variable values to `(num_variables, 1, 1)`, matching the type, dtype and device of `like`."""
        array = np.asarray(values, dtype=np.float32)[:, None, None]
        if isinstance(like, np.ndarray):
            return array.astype(like.dtype)
        import torch

        return torch.as_tensor(array, dtype=like.dtype, device=like.device)

    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.feature_extractor_type
        return output

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def save_pretrained(self, save_directory: str | os.PathLike):
        """Saves the configuration to `save_directory/preprocessor_config.json`."""
        os.makedirs(save_directory, exist_ok=True)
        output_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)
        with open(output_file, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
        logger.info(f"Feature extractor saved in {output_file}")

    @classmethod
    def from_dict(cls, config: dict) -> "NanoWeatherFeatureExtractor":
        config = {key: value for key, value in config.items() if key != "feature_extractor_type"}
        return cls(**config)

    @classmethod
    def from_pretrained(
        cls, pretrained_path: str | os.PathLike, repo_type: str = "dataset", revision: str | None = None
    ) -> "NanoWeatherFeatureExtractor":
        """
        Loads a feature extractor from a local directory or a Hub repository.

        Args:
            pretrained_path (`str` or `os.PathLike`):
                A local directory containing `preprocessor_config.json`, or a Hub repository id.
            repo_type (`str`, *optional*, defaults to `"dataset"`):
                Type of the Hub repository, `"dataset"` for the nanoWeather data or `"model"` for a checkpoint.
            revision (`str`, *optional*):
                Branch, tag or commit of the Hub repository.
        """
        if os.path.isdir(pretrained_path):
            config_file = os.path.join(pretrained_path, FEATURE_EXTRACTOR_NAME)
        else:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(
                str(pretrained_path), FEATURE_EXTRACTOR_NAME, repo_type=repo_type, revision=revision
            )
        with open(config_file, encoding="utf-8") as reader:
            return cls.from_dict(json.load(reader))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"