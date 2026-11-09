"""Loading YAML configs and merging them with command-line arguments."""

import argparse
import os

import yaml


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as reader:
        return yaml.safe_load(reader) or {}


def parse_args_with_config(parser: argparse.ArgumentParser, default_config: str | None = None) -> argparse.Namespace:
    """
    Parses command-line arguments, using the values of a YAML config file as defaults.

    The config file is given with `--config`. Values on the command line override values in the config, which
    override the defaults of the parser. Config keys without a matching argument (nested settings, for example) are
    added to the returned namespace unchanged.

    Args:
        parser (`argparse.ArgumentParser`):
            Parser with all arguments except `--config`, which is added here.
        default_config (`str`, *optional*):
            Config file used when `--config` is not given.
    """
    parser.add_argument("--config", type=str, default=default_config, help="YAML config file.")
    known_args, _ = parser.parse_known_args()
    config = load_yaml(known_args.config) if known_args.config else {}

    destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in config.items() if key in destinations})
    args = parser.parse_args()
    for key, value in config.items():
        if key not in destinations:
            setattr(args, key, value)
    return args
