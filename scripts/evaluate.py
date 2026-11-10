"""Scores persistence, climatology and trained models: RMSE, ACC and skill horizon by lead time, to 10 days.

    uv run scripts/evaluate.py --checkpoint checkpoints/rollout/best
    uv run scripts/evaluate.py --experiments experiments/experiments.yaml --output_dir results/experiments
    uv run scripts/evaluate.py --checkpoint checkpoints/rollout/best --set num_steps=20 baselines=false

All models are scored on the variables and initializations they have in common. Settings are in
configs/evaluation.yaml; --set overrides any of them.
"""

import argparse
import logging
import os

import numpy as np

from nanoweather.config import load_config, parse_args_with_config
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
from nanoweather.experiments import load_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None, help="Model directories or Hub repo ids.")
    parser.add_argument("--model_name", type=str, nargs="+", default=None, help="One name per checkpoint.")
    parser.add_argument("--experiments", type=str, default=None, help="Registry: evaluate all its trained models.")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parse_args_with_config(parser, default_config="configs/evaluation.yaml")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = list(args.checkpoint or [])
    model_names = list(args.model_name or [default_model_name(checkpoint) for checkpoint in checkpoints])
    if len(model_names) != len(checkpoints):
        raise ValueError(f"Got {len(checkpoints)} checkpoints but {len(model_names)} model names.")
    if args.experiments:
        missing = []
        for experiment in load_experiments(args.experiments):
            if os.path.isdir(experiment.checkpoint):
                checkpoints.append(experiment.checkpoint)
                model_names.append(experiment.name)
            else:
                missing.append(experiment.name)
        if missing:
            logging.warning(f"No checkpoint for {', '.join(missing)}: not evaluated")

    data_config = load_config(args.data_config)
    data = WeatherData(
        args.data_path or data_config["output_dir"], split=args.split, store_name=data_config["store_name"]
    )
    climatology = ClimatologyForecaster(load_climatology(data.store_path, data.variables))
    forecasters = {}
    if args.baselines:
        forecasters["persistence"] = persistence_forecaster
        forecasters["climatology"] = climatology
    for name, checkpoint in zip(model_names, checkpoints):
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
