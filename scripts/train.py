"""Trains a model (step 1 of the recipe), or fine-tunes it on multi-step rollouts (step 2).

    uv run scripts/train.py --overfit_one_batch
    uv run scripts/train.py
    uv run scripts/train.py --config configs/train/rollout.yaml
    uv run scripts/train.py --output_dir checkpoints/epochs50 --set num_train_epochs=50

All settings are in the config (configs/train/pretrain.yaml by default); --set overrides any of them.
"""

import argparse
import logging

from nanoweather.config import load_config, parse_args_with_config
from nanoweather.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None, help="Trained model to fine-tune.")
    parser.add_argument("--overfit_one_batch", action="store_true", default=None, help="Sanity check on one batch.")
    args = parse_args_with_config(parser, default_config="configs/train/pretrain.yaml")
    args.overfit_one_batch = bool(args.overfit_one_batch)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_config = load_config(args.data_config)
    args.data_path = args.data_path or data_config["output_dir"]
    train(args, data_config)


if __name__ == "__main__":
    main()
