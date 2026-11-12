"""Loss settings computed from the data: per-variable weights and per-latitude change statistics."""

from collections import defaultdict

import numpy as np


def get_pressure_loss_weights(
    variables: list[str], specs: dict[str, dict], surface_weights: dict[str, float]
) -> list[float]:
    """
    Loss weights proportional to pressure for pressure-level variables, and given weights for surface variables.

    Pressure-level variables are grouped by source variable. Within a group, weights are proportional to the pressure
    of each level and sum to 1, so each source variable counts once in total, whatever its number of levels. Surface
    variables get their weight from `surface_weights`. The weights are then normalized to mean 1.

    Args:
        variables (`list[str]`):
            Variable names, in channel order.
        specs (`dict[str, dict]`):
            For each variable, `{"source": ..., "level": ...}` as returned by `expand_variables`. Variables without
            `level` are surface variables.
        surface_weights (`dict[str, float]`):
            Weight of each surface variable. Missing surface variables get 1.0.

    Returns:
        `list[float]` with one weight per variable.
    """
    levels_by_source = defaultdict(list)
    for name in variables:
        if specs[name].get("level") is not None:
            levels_by_source[specs[name]["source"]].append(specs[name]["level"])
    weights = []
    for name in variables:
        level = specs[name].get("level")
        if level is not None:
            weights.append(level / sum(levels_by_source[specs[name]["source"]]))
        else:
            weights.append(surface_weights.get(name, 1.0))
    weights = np.array(weights, dtype=np.float64)
    return (weights / weights.mean()).tolist()


def change_std_by_latitude(states: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    """
    Standard deviation of the change between consecutive time steps, per variable and latitude row.

    Computed over time and longitude, in chunks of `chunk_size` time steps to bound memory.

    Args:
        states (`np.ndarray` of shape `(time, num_variables, latitude, longitude)`):
            Consecutive states, without gaps.

    Returns:
        `np.ndarray` of shape `(num_variables, latitude)`.
    """
    total = np.zeros(states.shape[1:3])
    total_squares = np.zeros(states.shape[1:3])
    count = 0
    for start in range(0, len(states) - 1, chunk_size):
        changes = np.diff(states[start : start + chunk_size + 1].astype(np.float64), axis=0)
        total += changes.sum(axis=(0, 3))
        total_squares += (changes**2).sum(axis=(0, 3))
        count += changes.shape[0] * changes.shape[3]
    mean = total / count
    return np.sqrt(np.maximum(total_squares / count - mean**2, 0.0))
