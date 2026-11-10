"""Trains a model. Settings come from a YAML config; any of them can be overridden on the command line.

uv run scripts/train.py --overfit_one_batch
uv run scripts/train.py --config configs/train/v0.yaml
uv run scripts/train.py --output_dir checkpoints/epochs50 --num_train_epochs 50
"""

import argparse
import logging

from nanoweather.config import load_yaml, parse_args_with_config
from nanoweather.training.trainer import train


def str_to_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_config", type=str)
    parser.add_argument("--data_path", type=str, default=None, help="Defaults to `output_dir` of the data config.")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--push_to_hub", type=str, default=None, help="Model repository id for the best checkpoint.")
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_hidden_layers", type=int)
    parser.add_argument("--num_attention_heads", type=int)
    parser.add_argument("--intermediate_size", type=int)
    parser.add_argument("--num_input_steps", type=int)
    parser.add_argument("--use_time_features", type=str_to_bool, nargs="?", const=True)
    parser.add_argument("--tendency_normalization", type=str_to_bool, nargs="?", const=True)
    parser.add_argument("--loss_weighting", type=str, choices=["uniform", "pressure"])
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--bf16", type=str_to_bool, nargs="?", const=True)
    parser.add_argument("--compile", type=str_to_bool, nargs="?", const=True)
    parser.add_argument("--overfit_one_batch", action="store_true")
    parser.add_argument("--overfit_steps", type=int)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parse_args_with_config(parser, default_config="configs/train/v0.yaml")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_config = load_yaml(args.data_config)
    args.data_path = args.data_path or data_config["output_dir"]
    train(args, data_config)


if __name__ == "__main__":
    main()
