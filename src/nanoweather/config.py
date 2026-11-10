"""Loading YAML configs and merging them with command-line arguments."""

import argparse
import os

import yaml


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as reader:
        return yaml.safe_load(reader) or {}


def load_config(path: str | os.PathLike) -> dict:
    """
    Loads a YAML config, resolving inheritance.

    A config can name another one with `base: <path>`: the base is loaded first (recursively), then the keys of the
    config override it. Paths are relative to the working directory, i.e. the repo root.
    """
    config = load_yaml(path)
    base = config.pop("base", None)
    if base is None:
        return config
    return {**load_config(base), **config}


def parse_overrides(overrides: list[str]) -> dict:
    """
    Parses `key=value` strings, reading each value as YAML so that `true`, `null` and `[2, 4]` get their types.

    Numbers in scientific notation without a decimal point, such as `5e-4`, are strings in YAML 1.1; they are
    converted to floats.
    """
    parsed = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected key=value, got {override!r}.")
        key, value = override.split("=", 1)
        value = yaml.safe_load(value)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        parsed[key.strip()] = value
    return parsed


def parse_args_with_config(parser: argparse.ArgumentParser, default_config: str | None = None) -> argparse.Namespace:
    """
    Parses command-line arguments on top of a YAML config.

    Adds `--config` (the config file) and `--set key=value ...` (overrides of any config key) to `parser`. Priority,
    highest first: explicit command-line arguments, `--set`, the config, its bases. Arguments of `parser` should
    default to `None`, so that unset arguments do not override the config.

    Returns:
        `argparse.Namespace` with all keys of the config and all arguments of the parser.
    """
    parser.add_argument("--config", type=str, default=default_config, help="YAML config file.")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="Override config values.")
    args = parser.parse_args()

    values = load_config(args.config) if args.config else {}
    values.update(parse_overrides(args.set))
    for key, value in vars(args).items():
        if key in ("config", "set"):
            continue
        if value is not None or key not in values:
            values[key] = value
    values["config"] = args.config
    return argparse.Namespace(**values)
