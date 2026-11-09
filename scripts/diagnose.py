"""Bias maps, RMSE by latitude and zonal spectra of a model's forecasts over many initializations.

uv run scripts/diagnose.py --checkpoint checkpoints/v0/best
uv run scripts/diagnose.py --checkpoint checkpoints/v0/best --init_start 2020-01-01T00 --init_end 2020-02-28T12
"""

import argparse
import logging
import os

import matplotlib


matplotlib.use("Agg")

import numpy as np

from nanoweather.config import load_yaml, parse_args_with_config
from nanoweather.data.datasets import WeatherData
from nanoweather.evaluation.diagnostics import compute_diagnostics, plot_diagnostics
from nanoweather.evaluation.evaluate import default_model_name, get_init_indices
from nanoweather.evaluation.forecasters import ModelForecaster
from nanoweather.visualization.forecast_figure import load_land_sea_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_config", type=str)
    parser.add_argument("--data_path", type=str, default=None, help="Defaults to `output_dir` of the data config.")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--init_start", type=str)
    parser.add_argument("--init_end", type=str)
    parser.add_argument("--lead_days", type=float, nargs="+")
    parser.add_argument("--variables", type=str, nargs="+")
    args = parse_args_with_config(parser, default_config="configs/diagnostics.yaml")
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_config = load_yaml(args.data_config)
    data = WeatherData(
        args.data_path or data_config["output_dir"], split=args.split, store_name=data_config["store_name"]
    )
    lead_steps = [round(days * 24 / data.time_step_hours) for days in args.lead_days]
    init_indices = get_init_indices(
        data.times, args.init_start, args.init_end, args.init_frequency_hours, max(lead_steps)
    )
    forecaster = ModelForecaster(args.checkpoint, data.static)
    diagnostics = compute_diagnostics(forecaster, data, init_indices, lead_steps, args.variables, args.batch_size)

    output_dir = os.path.join(args.output_dir, default_model_name(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "diagnostics.npz"), variables=args.variables, **diagnostics)
    units = [data.feature_extractor.units[data.variables.index(name)] for name in args.variables]
    figures = plot_diagnostics(
        diagnostics, args.variables, data.latitude, data.longitude, load_land_sea_mask(data.store_path), units
    )
    for name, figure in figures.items():
        figure.savefig(os.path.join(output_dir, f"{name}.png"), dpi=130)
    logging.info(f"{len(init_indices)} initializations, figures in {output_dir}")


if __name__ == "__main__":
    main()
