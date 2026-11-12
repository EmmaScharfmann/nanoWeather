"""Where and why models fail by latitude: one-step error against persistence, on training and test years, and how the
normalization and loss weights treat each latitude.

    uv run scripts/diagnose_latitude.py --checkpoint checkpoints/vars29_both_rollout/best checkpoints/vars29_mesh_rollout/best

For each variable, three rows of plots:
    1. RMSE of the first forecast step by latitude, for each model and for persistence (whose error is the typical
       change over one step), on training years (dashed) and test years (solid).
    2. Model RMSE divided by persistence RMSE. Near or above 1 means the model adds little or nothing at that latitude.
       High on training years too means the model never learned that latitude (underfitting).
    3. Spread of the state and of its one-step change at each latitude, divided by the global values the normalization
       uses, and the weight of each latitude row in the loss.
"""

import argparse
import json
import logging
import os

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from nanoweather.config import load_config, parse_args_with_config
from nanoweather.data.datasets import WeatherData, get_store_path, latitude_weights
from nanoweather.evaluation.diagnostics import compute_diagnostics, statistics_by_latitude
from nanoweather.evaluation.evaluate import default_model_name
from nanoweather.evaluation.forecasters import ModelForecaster, persistence_forecaster


def sample_split(data_path, store_name, split, num_samples, num_history_steps, feature_extractor):
    """Loads `num_samples` evenly spaced initializations of a split, with their history and next step only."""
    num_times = xr.open_zarr(get_store_path(data_path, store_name), group=split).sizes["time"]
    inits = np.unique(np.linspace(num_history_steps, num_times - 2, num_samples).astype(int))
    window = np.arange(-num_history_steps, 2)
    time_indices = np.unique((inits[:, None] + window[None, :]).ravel())
    data = WeatherData(data_path, split, store_name, feature_extractor=feature_extractor, time_indices=time_indices)
    return data, np.searchsorted(time_indices, inits)


def plot(results, statistics, normalization, latitude, variables, model_names, path):
    colors = dict(zip(["persistence", *model_names], plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    styles = {"train": "--", "test": "-"}
    fig, axes = plt.subplots(
        3, len(variables), figsize=(5.2 * len(variables), 11), squeeze=False, constrained_layout=True
    )
    for column, variable in enumerate(variables):
        for split, by_forecaster in results.items():
            for name, rmse in by_forecaster.items():
                axes[0, column].plot(
                    latitude, rmse[column], styles[split], color=colors[name], label=f"{name} ({split})"
                )
                if name != "persistence":
                    ratio = rmse[column] / by_forecaster["persistence"][column]
                    axes[1, column].plot(latitude, ratio, styles[split], color=colors[name], label=f"{name} ({split})")
        test = statistics["test"]
        axes[2, column].plot(
            latitude, test["state_std"][column] / normalization["std"][column], label="state spread / global std"
        )
        axes[2, column].plot(
            latitude, test["change_std"][column] / normalization["diff_std"][column], label="6h change spread / global"
        )
        axes[2, column].plot(latitude, latitude_weights(latitude), "k:", label="loss weight of the row")
        axes[0, column].set_title(f"{variable}: RMSE of the first step")
        axes[1, column].set_title(f"{variable}: model / persistence")
        axes[1, column].axhline(1.0, color="k", linewidth=0.8)
        axes[2, column].set_title(f"{variable}: spread by latitude / normalization")
        axes[2, column].axhline(1.0, color="k", linewidth=0.8)
        for row in range(3):
            axes[row, column].set_xlabel("latitude")
            axes[row, column].grid(alpha=0.3)
    for row in range(3):
        axes[row, 0].legend(fontsize=7)
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None)
    parser.add_argument("--model_name", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parse_args_with_config(parser, default_config="configs/diagnostics.yaml")
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    num_samples = getattr(args, "latitude_num_samples", 400)
    output_dir = os.path.join(args.output_dir, "latitude")
    os.makedirs(output_dir, exist_ok=True)

    data_config = load_config(args.data_config)
    data_path, store_name = args.data_path or data_config["output_dir"], data_config["store_name"]
    model_names = args.model_name or [default_model_name(checkpoint) for checkpoint in args.checkpoint]
    first = WeatherData(data_path, "test", store_name, time_indices=np.arange(3))
    forecasters = {
        name: ModelForecaster(checkpoint, first.static) for name, checkpoint in zip(model_names, args.checkpoint)
    }
    num_history_steps = max(forecaster.num_input_steps for forecaster in forecasters.values()) - 1

    results, statistics, summary = {}, {}, {}
    for split in ("train", "test"):
        data, inits = sample_split(
            data_path, store_name, split, num_samples, num_history_steps, first.feature_extractor
        )
        logging.info(f"{split}: {len(inits)} initializations")
        results[split] = {}
        for name, forecaster in {"persistence": persistence_forecaster, **forecasters}.items():
            diagnostics = compute_diagnostics(forecaster, data, inits, [1], args.variables, args.batch_size)
            results[split][name] = diagnostics["rmse_by_latitude"][0]
        statistics[split] = statistics_by_latitude(data, inits, args.variables)

    latitude = first.latitude
    channels = [first.variables.index(name) for name in args.variables]
    normalization = {key: np.array(getattr(first.feature_extractor, key))[channels] for key in ("std", "diff_std")}
    plot(
        results,
        statistics,
        normalization,
        latitude,
        args.variables,
        model_names,
        os.path.join(output_dir, "latitude.png"),
    )

    polar, middle = np.abs(latitude) > 80, (np.abs(latitude) > 40) & (np.abs(latitude) < 60)
    for split in results:
        for name in model_names:
            ratio = results[split][name] / results[split]["persistence"]
            summary[f"{name} {split}"] = {
                variable: {"polar": float(ratio[index, polar].mean()), "40-60": float(ratio[index, middle].mean())}
                for index, variable in enumerate(args.variables)
            }
            text = ", ".join(
                f"{variable} polar {values['polar']:.2f} / 40-60 {values['40-60']:.2f}"
                for variable, values in summary[f"{name} {split}"].items()
            )
            logging.info(f"model / persistence, first step, {name} {split}: {text}")
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2)
    np.savez(
        os.path.join(output_dir, "latitude.npz"),
        latitude=latitude,
        variables=args.variables,
        **{f"rmse_{split}_{name}": rmse for split in results for name, rmse in results[split].items()},
        **{f"{key}_{split}": value for split in statistics for key, value in statistics[split].items()},
    )
    logging.info(f"Figure and summary in {output_dir}")


if __name__ == "__main__":
    main()
