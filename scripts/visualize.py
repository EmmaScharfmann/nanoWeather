"""Animation and snapshots of one forecast against the truth.

    uv run scripts/visualize.py --checkpoint checkpoints/rollout/best
    uv run scripts/visualize.py --checkpoint checkpoints/rollout/best --init_time 2020-01-15T00 --set num_steps=40

Settings are in configs/visualization.yaml.
"""

import argparse
import logging
import os

import matplotlib


matplotlib.use("Agg")

import pandas as pd

from nanoweather.config import load_config, parse_args_with_config
from nanoweather.data.datasets import WeatherData
from nanoweather.visualization.forecast_figure import ForecastFigure, get_forecast_and_truth, load_land_sea_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--init_time", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parse_args_with_config(parser, default_config="configs/visualization.yaml")
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    data_config = load_config(args.data_config)
    data_path = args.data_path or data_config["output_dir"]
    data = WeatherData(data_path, split=args.split, store_name=data_config["store_name"])
    forecast, truth, array_variables = get_forecast_and_truth(data, args.checkpoint, args.init_time, args.num_steps)
    land_sea_mask = load_land_sea_mask(data.store_path)
    figure = ForecastFigure(
        data, forecast, truth, array_variables, args.variables, land_sea_mask, args.init_time, args.display
    )

    stem = os.path.join(args.output_dir, f"forecast_{pd.Timestamp(args.init_time):%Y%m%d%H}")
    figure.save_animation(f"{stem}.gif", fps=args.fps)
    steps = [round(days * 24 / data.time_step_hours) - 1 for days in args.snapshot_days]
    figure.save_snapshots(f"{stem}_snapshots.png", [step for step in steps if 0 <= step < args.num_steps])
    logging.info(f"Wrote {stem}.gif and {stem}_snapshots.png")


if __name__ == "__main__":
    main()
