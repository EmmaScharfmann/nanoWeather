"""Bias maps, RMSE by latitude and zonal spectra of a model's forecasts over many initializations.

RMSE by latitude is shown with the natural variability at each latitude (the RMSE of the climatology forecast) and
divided by it: forecast errors grow toward that level, which is itself larger at high latitudes.

Also maps where the error is relative to natural variability, how concentrated it is in space (concentration curves
and Gini coefficient), by region (ocean, land, high terrain, coasts, latitude bands) and over the forecasts of the year,
and which share comes from bias. Numbers are in heterogeneity.json.

    uv run scripts/diagnose.py --checkpoint checkpoints/rollout/best
    uv run scripts/diagnose.py --checkpoint checkpoints/rollout/best --output_dir results/diagnostics_winter \
        --set init_start=2020-01-01T00 init_end=2020-02-28T12

Writes results to <output_dir>/<model name>/. Settings are in configs/diagnostics.yaml.
"""

import argparse
import json
import logging
import os

import matplotlib


matplotlib.use("Agg")

import numpy as np

from nanoweather.config import load_config, parse_args_with_config
from nanoweather.data.datasets import WeatherData
from nanoweather.evaluation.diagnostics import compute_diagnostics, plot_diagnostics
from nanoweather.evaluation.evaluate import default_model_name, get_init_indices
from nanoweather.evaluation.forecasters import (
    ClimatologyForecaster,
    ModelForecaster,
    load_climatology,
    persistence_forecaster,
)
from nanoweather.evaluation.heterogeneity import analyze_heterogeneity, plot_heterogeneity
from nanoweather.visualization.forecast_figure import load_static_field


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parse_args_with_config(parser, default_config="configs/diagnostics.yaml")
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_config = load_config(args.data_config)
    data = WeatherData(
        args.data_path or data_config["output_dir"], split=args.split, store_name=data_config["store_name"]
    )
    lead_steps = [round(days * 24 / data.time_step_hours) for days in args.lead_days]
    forecaster = ModelForecaster(args.checkpoint, data.static)
    init_indices = get_init_indices(
        data.times,
        args.init_start,
        args.init_end,
        args.init_frequency_hours,
        max(lead_steps),
        forecaster.num_input_steps - 1,
    )
    diagnostics = compute_diagnostics(forecaster, data, init_indices, lead_steps, args.variables, args.batch_size)
    references = {
        "climatology": ClimatologyForecaster(load_climatology(data.store_path, data.variables)),
        "persistence": persistence_forecaster,
    }
    for name, reference in references.items():
        reference_diagnostics = compute_diagnostics(
            reference, data, init_indices, lead_steps, args.variables, args.batch_size
        )
        diagnostics[f"{name}_rmse_by_latitude"] = reference_diagnostics["rmse_by_latitude"]
        diagnostics[f"{name}_rmse_map"] = reference_diagnostics["rmse_map"]

    output_dir = os.path.join(args.output_dir, default_model_name(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)
    init_times = data.times[init_indices]
    np.savez(
        os.path.join(output_dir, "diagnostics.npz"),
        variables=args.variables,
        latitude=data.latitude,
        longitude=data.longitude,
        init_times=init_times,
        **diagnostics,
    )
    units = [data.feature_extractor.units[data.variables.index(name)] for name in args.variables]
    land_sea_mask = load_static_field(data.store_path, "land_sea_mask")
    orography = load_static_field(data.store_path, "orography")
    figures = plot_diagnostics(diagnostics, args.variables, data.latitude, data.longitude, land_sea_mask, units)
    figures.update(
        plot_heterogeneity(
            diagnostics, args.variables, data.latitude, data.longitude, land_sea_mask, orography, init_times, units
        )
    )
    for name, figure in figures.items():
        figure.savefig(os.path.join(output_dir, f"{name}.png"), dpi=130)

    summary = analyze_heterogeneity(diagnostics, args.variables, data.latitude, land_sea_mask, orography)
    with open(os.path.join(output_dir, "heterogeneity.json"), "w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2)
    for variable, by_lead in summary.items():
        text = "; ".join(
            f"day {days}: Gini {values['gini']:.2f}, worst 10% of area {values['worst_10_percent_area']:.0%}, "
            f"worst 10% of forecasts {values['worst_10_percent_forecasts']:.0%}, from bias {values['systematic_share']:.0%}"
            for days, values in by_lead.items()
        )
        logging.info(f"{variable}: {text}")
    logging.info(f"{len(init_indices)} initializations, figures and heterogeneity.json in {output_dir}")


if __name__ == "__main__":
    main()
