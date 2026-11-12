"""Model classes by architecture, and loading a checkpoint of either architecture."""

import os

from .configuration_utils import load_config_dict
from .modeling_nanoweather import NanoWeatherForWeatherForecasting
from .modeling_nanoweather_mesh import NanoWeatherMeshForWeatherForecasting


MODEL_FOR_WEATHER_FORECASTING_MAPPING = {
    "nanoweather": NanoWeatherForWeatherForecasting,
    "nanoweather_mesh": NanoWeatherMeshForWeatherForecasting,
}


def model_class_for(model_type: str):
    if model_type not in MODEL_FOR_WEATHER_FORECASTING_MAPPING:
        raise ValueError(
            f"Unknown model type {model_type!r}, expected one of {list(MODEL_FOR_WEATHER_FORECASTING_MAPPING)}."
        )
    return MODEL_FOR_WEATHER_FORECASTING_MAPPING[model_type]


def load_model(pretrained_model_name_or_path: str | os.PathLike, **kwargs):
    """Loads a forecasting model of any architecture, using `model_type` in its `config.json`."""
    model_type = load_config_dict(pretrained_model_name_or_path, **kwargs).get("model_type", "nanoweather")
    return model_class_for(model_type).from_pretrained(pretrained_model_name_or_path, **kwargs)
