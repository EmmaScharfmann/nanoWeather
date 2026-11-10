"""Scores persistence, climatology and trained models with latitude-weighted RMSE by lead time.

uv run scripts/evaluate.py
uv run scripts/evaluate.py --checkpoint checkpoints/v0/best checkpoints/epochs50/best
uv run scripts/evaluate.py --checkpoint checkpoints/v0/best --num_steps 40 --output_dir results/10days
"""

import argparse
import logging
import os

import numpy as np

from nanoweather.config import load_yaml, parse_args_with_config
from nanoweather.data.datasets import WeatherData
from nanoweather.evaluation.evaluate import (
    default_model_name,
    evaluate_forecaster,
    get_common_variables,
    get_init_indices,
    get_num_history_steps,
    plot_scores,
    save_results,
)
from nanoweather.evaluation.forecasters import (
    ClimatologyForecaster,
    ModelForecaster,
    load_climatology,
    persistence_forecaster,
)
from nanoweather.evaluation.metrics import skill_horizon_hours


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_config", type=str)
    parser.add_argument("--data_path", type=str, default=None, help="Defaults to `output_dir` of the data config.")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--init_start", type=str)
    parser.add_argument("--init_end", type=str)
    parser.add_argument("--checkpoint", type=str, nargs="+", default=[], help="Model directories or Hub repo ids.")
    parser.add_argument("--model_name", type=str, nargs="+", default=None, help="One name per checkpoint.")
    parser.add_argument("--no_baselines", action="store_true", help="Skip persistence and climatology.")
    args = parse_args_with_config(parser, default_config="configs/evaluation.yaml")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    data_config = load_yaml(args.data_config)
    data_path = args.data_path or data_config["output_dir"]
    data = WeatherData(data_path, split=args.split, store_name=data_config["store_name"])
    climatology = ClimatologyForecaster(load_climatology(data.store_path, data.variables))
    forecasters = {}
    if not args.no_baselines:
        forecasters["persistence"] = persistence_forecaster
        forecasters["climatology"] = climatology
    model_names = args.model_name or [default_model_name(checkpoint) for checkpoint in args.checkpoint]
    if len(model_names) != len(args.checkpoint):
        raise ValueError(f"Got {len(args.checkpoint)} checkpoints but {len(model_names)} model names.")
    for name, checkpoint in zip(model_names, args.checkpoint):
        forecasters[name] = ModelForecaster(checkpoint, data.static)

    init_indices = get_init_indices(
        data.times,
        args.init_start,
        args.init_end,
        args.init_frequency_hours,
        args.num_steps,
        get_num_history_steps(list(forecasters.values())),
    )
    variables = get_common_variables(list(forecasters.values()), data)
    logging.info(f"Scoring {len(variables)} variables forecast by all models: {', '.join(variables)}")

    all_results = {}
    summary_index = min(args.num_steps, args.summary_lead_hours // data.time_step_hours) - 1
    for name, forecaster in forecasters.items():
        results = evaluate_forecaster(
            forecaster, data, init_indices, args.num_steps, args.batch_size, variables, climatology=climatology
        )
        all_results[name] = results
        save_results(results, variables, os.path.join(args.output_dir, f"{name}.json"), args.acc_threshold)
        summary = []
        for variable in [plot_variable for plot_variable in args.plot_variables if plot_variable in variables]:
            index = variables.index(variable)
            text = f"{variable} RMSE {results['rmse'][summary_index, index]:.4g}"
            if results["acc"] is not None and not np.isnan(results["acc"][:, index]).all():
                horizon = skill_horizon_hours(results["lead_hours"], results["acc"][:, index], args.acc_threshold)
                text += f", ACC > {args.acc_threshold} for " + (
                    f"{horizon / 24:.1f} days" if horizon is not None else f"over {args.num_steps} steps"
                )
            summary.append(text)
        summary = "; ".join(summary)
        logging.info(f"{name} at {results['lead_hours'][summary_index]}h: {summary}")

    plot_scores(
        all_results,
        variables,
        args.plot_variables,
        args.acc_threshold,
        path=os.path.join(args.output_dir, "scores.png"),
    )
    logging.info(f"{len(init_indices)} initializations, results written to {args.output_dir}")


if __name__ == "__main__":
    main()
