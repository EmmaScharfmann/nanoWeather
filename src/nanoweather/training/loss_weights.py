"""Per-variable loss weights."""

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
