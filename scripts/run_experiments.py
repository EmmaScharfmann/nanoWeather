"""Trains the experiments of a registry that have no checkpoint yet, in registry order.

    uv run scripts/run_experiments.py --dry_run
    uv run scripts/run_experiments.py
    uv run scripts/run_experiments.py --only vars29_both vars29_both_rollout --retrain

Both datasets must be prepared first (see the README).
"""

import argparse
import logging
import os
import subprocess
import sys

from nanoweather.experiments import load_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=str, default="experiments/experiments.yaml")
    parser.add_argument("--only", type=str, nargs="+", default=None, help="Names of the experiments to train.")
    parser.add_argument("--retrain", action="store_true", help="Train even if a checkpoint exists.")
    parser.add_argument("--dry_run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    experiments = load_experiments(args.experiments)
    by_name = {experiment.name: experiment for experiment in experiments}
    for experiment in experiments:
        if args.only and experiment.name not in args.only:
            continue
        if os.path.exists(os.path.join(experiment.checkpoint, "config.json")) and not args.retrain:
            logging.info(f"{experiment.name}: checkpoint exists, skipped")
            continue
        command = [sys.executable, "scripts/train.py", "--config", experiment.config]
        command += ["--output_dir", os.path.dirname(experiment.checkpoint)]
        if experiment.init_from:
            command += ["--init_from", by_name[experiment.init_from].checkpoint]
        logging.info(f"{experiment.name}: {' '.join(command[1:])}")
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
