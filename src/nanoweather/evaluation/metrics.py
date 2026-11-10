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


def anomaly_correlation(
    forecast: np.ndarray, truth: np.ndarray, climatology: np.ndarray, latitude_weights: np.ndarray
) -> np.ndarray:
    """
    Anomaly correlation coefficient over the grid, with each latitude row weighted by its area.

    Anomalies are differences from `climatology`. The result is undefined (NaN) for a forecast without anomalies,
    such as the climatology itself.

    Args:
        forecast (`np.ndarray` of shape `(..., latitude, longitude)`):
            Forecast.
        truth (`np.ndarray` of shape `(..., latitude, longitude)`):
            Truth, in the same units as `forecast`.
        climatology (`np.ndarray` of shape `(..., latitude, longitude)`):
            Climatology at the same valid times.
        latitude_weights (`np.ndarray` of shape `(latitude,)`):
            Area weights with mean 1.

    Returns:
        `np.ndarray` of shape `(...)`: one value per field.
    """
    weights = latitude_weights[:, None]
    forecast_anomaly = forecast.astype(np.float64) - climatology
    truth_anomaly = truth.astype(np.float64) - climatology
    covariance = (weights * forecast_anomaly * truth_anomaly).sum(axis=(-2, -1))
    forecast_variance = (weights * forecast_anomaly**2).sum(axis=(-2, -1))
    truth_variance = (weights * truth_anomaly**2).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return covariance / np.sqrt(forecast_variance * truth_variance)


def skill_horizon_hours(lead_hours: np.ndarray, acc: np.ndarray, threshold: float) -> float | None:
    """
    First lead time at which `acc` falls below `threshold`, linearly interpolated between lead times.

    Returns `None` if `acc` stays above the threshold over all lead times, or is undefined.
    """
    if np.isnan(acc).all():
        return None
    below = np.flatnonzero(acc < threshold)
    if len(below) == 0:
        return None
    index = below[0]
    if index == 0:
        return float(lead_hours[0])
    previous_hours, hours = lead_hours[index - 1], lead_hours[index]
    previous_acc, current_acc = acc[index - 1], acc[index]
    return float(previous_hours + (previous_acc - threshold) / (previous_acc - current_acc) * (hours - previous_hours))
