"""How evenly forecast errors are spread: over the globe, across types of surface, and across forecasts.

Errors are compared with the natural variability at each point (the RMSE of the climatology forecast), so that a
region counts as weak only if the model does worse there relative to what is achievable.
"""

import numpy as np


GRAVITY = 9.80665
HIGH_TERRAIN_METERS = 1500.0


def concentration_curve(squared_error: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Share of the total squared error against share of the area, from the worst grid points to the best.

    Args:
        squared_error (`np.ndarray` of shape `(latitude, longitude)`):
            Squared error (or normalized squared error) at each grid point.
        area (`np.ndarray` of shape `(latitude, longitude)`):
            Area of each grid point, in any units.

    Returns:
        `area_share` and `error_share`, both starting at 0 and ending at 1, and the Gini coefficient: 0 when the error
        is spread in proportion to area, approaching 1 when it is concentrated in a few points.
    """
    density = squared_error.ravel()
    weights = area.ravel()
    order = np.argsort(-density)
    area_share = np.concatenate([[0.0], np.cumsum(weights[order]) / weights.sum()])
    contribution = density[order] * weights[order]
    error_share = np.concatenate([[0.0], np.cumsum(contribution) / contribution.sum()])
    gini = 2 * np.trapezoid(error_share, area_share) - 1
    return area_share, error_share, float(gini)


def share_from_worst(values: np.ndarray, fraction: float, weights: np.ndarray | None = None) -> float:
    """Share of the total of `values` coming from the largest values covering `fraction` of the total weight."""
    values = values.ravel()
    weights = np.ones_like(values) if weights is None else weights.ravel()
    order = np.argsort(-values)
    covered = np.cumsum(weights[order]) / weights.sum()
    selected = order[: max(1, int(np.searchsorted(covered, fraction)) + 1)]
    return float((values[selected] * weights[selected]).sum() / (values * weights).sum())


def region_masks(land_sea_mask: np.ndarray, orography: np.ndarray, latitude: np.ndarray) -> dict[str, np.ndarray]:
    """
    Boolean masks of regions defined from static fields.

    Args:
        land_sea_mask (`np.ndarray` of shape `(latitude, longitude)`):
            Fraction of land in each grid cell.
        orography (`np.ndarray` of shape `(latitude, longitude)`):
            Geopotential of the surface, in m**2 s**-2.
        latitude (`np.ndarray` of shape `(latitude,)`):
            Latitudes in degrees.

    Returns:
        `dict` of masks: `ocean`, `land` (below 1500 m), `high terrain` (1500 m and above), `coast` (cells mixing land
        and sea, or next to a cell of the other kind), `tropics` (within 20 degrees of the equator), `mid-latitudes`
        (20 to 60 degrees) and `polar` (beyond 60 degrees). Regions can overlap.
    """
    land = land_sea_mask >= 0.5
    height = orography / GRAVITY
    neighbors_differ = np.zeros_like(land)
    for shift in (-1, 1):
        neighbors_differ |= land != np.roll(land, shift, axis=1)
    neighbors_differ[1:] |= land[1:] != land[:-1]
    neighbors_differ[:-1] |= land[:-1] != land[1:]
    absolute_latitude = np.abs(latitude)[:, None] * np.ones_like(land_sea_mask)
    return {
        "ocean": ~land,
        "land": land & (height < HIGH_TERRAIN_METERS),
        "high terrain": land & (height >= HIGH_TERRAIN_METERS),
        "coast": ((land_sea_mask > 0.1) & (land_sea_mask < 0.9)) | neighbors_differ,
        "tropics": absolute_latitude < 20,
        "mid-latitudes": (absolute_latitude >= 20) & (absolute_latitude < 60),
        "polar": absolute_latitude >= 60,
    }


def scores_by_region(
    rmse_map: np.ndarray, reference_map: np.ndarray, masks: dict[str, np.ndarray], area: np.ndarray
) -> dict[str, dict[str, float]]:
    """
    Area-weighted RMSE of the model and of the reference (natural variability) in each region, and their ratio.

    Args:
        rmse_map, reference_map (`np.ndarray` of shape `(latitude, longitude)`):
            RMSE of the model and of the climatology forecast at each grid point.
        masks (`dict[str, np.ndarray]`):
            Regions, see `region_masks`.
        area (`np.ndarray` of shape `(latitude, longitude)`):
            Area of each grid point.
    """
    scores = {}
    for name, mask in masks.items():
        weights = area * mask
        if weights.sum() == 0:
            continue
        model = np.sqrt((rmse_map**2 * weights).sum() / weights.sum())
        reference = np.sqrt((reference_map**2 * weights).sum() / weights.sum())
        scores[name] = {
            "rmse": float(model),
            "natural_variability": float(reference),
            "ratio": float(model / reference),
            "area_share": float(weights.sum() / area.sum()),
        }
    return scores


def systematic_share(bias: np.ndarray, rmse_map: np.ndarray) -> np.ndarray:
    """Share of the mean squared error coming from the bias (systematic error) at each point, between 0 and 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(np.nan_to_num(bias**2 / rmse_map**2), 0.0, 1.0)


def analyze_heterogeneity(
    diagnostics: dict, variables: list[str], latitude, land_sea_mask, orography
) -> dict[str, dict]:
    """
    Concentration, regional scores and systematic share for each variable and lead time.

    Args:
        diagnostics (`dict`):
            Output of `compute_diagnostics`, with `climatology_rmse_map` added from the climatology forecast.
        variables (`list[str]`):
            Variables, in the order of the diagnostics.
        latitude (`np.ndarray` of shape `(latitude,)`), land_sea_mask, orography (`np.ndarray` of shape
        `(latitude, longitude)`):
            Grid and static fields, in physical units.

    Returns:
        `dict` keyed by variable then lead time in days, with `gini` (concentration over the globe of the error
        normalized by natural variability), `worst_10_percent_area`, `worst_10_percent_forecasts` (shares of the total
        squared error), `systematic_share` (of the total squared error) and `regions` (see `scores_by_region`).
    """
    area = np.cos(np.deg2rad(latitude))[:, None] * np.ones_like(land_sea_mask)
    masks = region_masks(land_sea_mask, orography, latitude)
    lead_days = diagnostics["lead_hours"] / 24
    summary = {}
    for column, variable in enumerate(variables):
        summary[variable] = {}
        for lead, days in enumerate(lead_days):
            rmse_map = diagnostics["rmse_map"][lead, column]
            reference = diagnostics["climatology_rmse_map"][lead, column]
            normalized = rmse_map**2 / np.maximum(reference**2, 1e-12)
            _, _, gini = concentration_curve(normalized, area)
            bias = diagnostics["bias"][lead, column]
            summary[variable][f"{days:g}"] = {
                "gini": gini,
                "worst_10_percent_area": share_from_worst(normalized, 0.1, area),
                "worst_10_percent_forecasts": share_from_worst(
                    diagnostics["rmse_by_initialization"][:, lead, column] ** 2, 0.1
                ),
                "systematic_share": float((bias**2 * area).sum() / (rmse_map**2 * area).sum()),
                "regions": scores_by_region(rmse_map, reference, masks, area),
            }
    return summary


def plot_heterogeneity(diagnostics, variables, latitude, longitude, land_sea_mask, orography, init_times, units):
    """
    Four figures: normalized error maps, concentration curves, scores by region, and error over time.

    Returns:
        `dict` of figure name to `matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    lead_days = diagnostics["lead_hours"] / 24
    num_leads, num_variables = len(lead_days), len(variables)
    area = np.cos(np.deg2rad(latitude))[:, None] * np.ones_like(land_sea_mask)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figures = {}

    fig, axes = plt.subplots(
        num_variables, 4, figsize=(19, 2.9 * num_variables), squeeze=False, constrained_layout=True
    )
    last = num_leads - 1
    for row, name in enumerate(variables):
        rmse_map = diagnostics["rmse_map"][:, row]
        reference = diagnostics["climatology_rmse_map"][:, row]
        panels = [
            (rmse_map[last], f"{name} RMSE, day {lead_days[last]:g}", "viridis", None, units[row]),
            (
                rmse_map[0] / reference[0],
                f"{name} RMSE / natural variability, day {lead_days[0]:g}",
                "magma",
                (0, None),
                "",
            ),
            (
                rmse_map[last] / reference[last],
                f"{name} RMSE / natural variability, day {lead_days[last]:g}",
                "magma",
                (0, 1.2),
                "",
            ),
            (
                systematic_share(diagnostics["bias"][last, row], rmse_map[last]),
                f"{name} share of error from bias, day {lead_days[last]:g}",
                "Reds",
                (0, 1),
                "",
            ),
        ]
        for column, (values, title, cmap, limits, label) in enumerate(panels):
            ax = axes[row, column]
            vmin, vmax = limits if limits else (None, None)
            mesh = ax.pcolormesh(longitude, latitude, values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(
                longitude, latitude, land_sea_mask, levels=[0.5], colors="w" if cmap != "Reds" else "k", linewidths=0.5
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(mesh, ax=ax, label=label, shrink=0.85)
    figures["error_maps"] = fig

    fig, axes = plt.subplots(
        1, num_variables, figsize=(4.5 * num_variables, 4), squeeze=False, constrained_layout=True
    )
    for column, name in enumerate(variables):
        ax = axes[0, column]
        for lead in range(num_leads):
            normalized = diagnostics["rmse_map"][lead, column] ** 2 / np.maximum(
                diagnostics["climatology_rmse_map"][lead, column] ** 2, 1e-12
            )
            area_share, error_share, gini = concentration_curve(normalized, area)
            ax.plot(area_share, error_share, color=colors[lead], label=f"day {lead_days[lead]:g} (Gini {gini:.2f})")
        ax.plot([0, 1], [0, 1], "k:", label="evenly spread")
        ax.set_title(f"{name}: where the error is (worst points first)", fontsize=10)
        ax.set_xlabel("share of the globe's area")
        ax.set_ylabel("share of the normalized squared error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    figures["concentration"] = fig

    summary = analyze_heterogeneity(diagnostics, variables, latitude, land_sea_mask, orography)
    region_names = list(next(iter(next(iter(summary.values())).values()))["regions"])
    fig, axes = plt.subplots(
        1, num_variables, figsize=(4.5 * num_variables, 4), squeeze=False, constrained_layout=True
    )
    width = 0.8 / num_leads
    for column, name in enumerate(variables):
        ax = axes[0, column]
        for lead, days in enumerate(lead_days):
            regions = summary[name][f"{days:g}"]["regions"]
            ratios = [regions[region]["ratio"] if region in regions else np.nan for region in region_names]
            ax.bar(
                np.arange(len(region_names)) + lead * width, ratios, width, color=colors[lead], label=f"day {days:g}"
            )
        ax.set_xticks(
            np.arange(len(region_names)) + width * (num_leads - 1) / 2, region_names, rotation=30, ha="right"
        )
        ax.set_title(f"{name}: RMSE / natural variability by region", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    figures["regions"] = fig

    fig, axes = plt.subplots(
        num_variables, 1, figsize=(14, 2.6 * num_variables), squeeze=False, constrained_layout=True
    )
    for row, name in enumerate(variables):
        ax = axes[row, 0]
        for lead, days in enumerate(lead_days):
            values = diagnostics["rmse_by_initialization"][:, lead, row]
            share = summary[name][f"{days:g}"]["worst_10_percent_forecasts"]
            ax.plot(
                init_times,
                values,
                color=colors[lead],
                linewidth=0.8,
                label=f"day {days:g} (worst 10%: {share:.0%} of error)",
            )
        ax.set_title(f"{name}: RMSE of each forecast through the year", fontsize=10)
        ax.set_ylabel(units[row])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    figures["error_over_time"] = fig
    return figures
