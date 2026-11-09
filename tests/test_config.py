"""Tests for merging YAML configs with command-line arguments."""

import argparse
import sys

from nanoweather.config import parse_args_with_config


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=8)
    return parser


def test_command_line_overrides_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("learning_rate: 0.5\nhidden_size: 64\ndisplay:\n  z500: {units: m}\n")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config), "--hidden_size", "128"])
    args = parse_args_with_config(make_parser())
    assert args.learning_rate == 0.5
    assert args.hidden_size == 128
    assert args.display == {"z500": {"units": "m"}}


def test_parser_defaults_without_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args_with_config(make_parser())
    assert args.learning_rate == 1.0
