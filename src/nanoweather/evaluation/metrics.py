"""Metrics on a latitude/longitude grid."""

import numpy as np


def latitude_weighted_rmse(forecast: np.ndarray, truth: np.ndarray, latitude_weights: np.ndarray) -> np.ndarray:
    """
    Root mean squared error over the grid, with each latitude row weighted by its area.

    Args:
        forecast (`np.ndarray` of shape `(..., latitude, longitude)`):
            Forecast.
        truth (`np.ndarray` of shape `(..., latitude, longitude)`):
            Truth, in the same units as `forecast`.
        latitude_weights (`np.ndarray` of shape `(latitude,)`):
            Area weights with mean 1.

    Returns:
        `np.ndarray` of shape `(...)`: one value per field.
    """
    squared_error = (forecast.astype(np.float64) - truth) ** 2
    return np.sqrt((squared_error * latitude_weights[:, None]).mean(axis=(-2, -1)))
