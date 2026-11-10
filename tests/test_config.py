"""Tests for configs: inheritance, overrides, command-line merging, and the experiment registry."""

import argparse
import sys

import pytest

from nanoweather.config import load_config, parse_args_with_config, parse_overrides
from nanoweather.experiments import load_experiments


def write(path, text):
    path.write_text(text)
    return str(path)


def test_base_inheritance(tmp_path):
    base = write(tmp_path / "base.yaml", "learning_rate: 0.1\nhidden_size: 8\nseed: 1\n")
    middle = write(tmp_path / "middle.yaml", f"base: {base}\nhidden_size: 16\n")
    child = write(tmp_path / "child.yaml", f"base: {middle}\nseed: 2\n")
    assert load_config(child) == {"learning_rate": 0.1, "hidden_size": 16, "seed": 2}


def test_overrides_are_typed():
    assert parse_overrides(["a=5e-4", "b=true", "c=[2, 4]", "d=text"]) == {
        "a": 5e-4,
        "b": True,
        "c": [2, 4],
        "d": "text",
    }
    with pytest.raises(ValueError):
        parse_overrides(["no_equal_sign"])


def test_priority(tmp_path, monkeypatch):
    config = write(tmp_path / "config.yaml", "learning_rate: 0.5\nhidden_size: 64\noutput_dir: a\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None)
    argv = ["train.py", "--config", config, "--set", "hidden_size=128", "--output_dir", "b"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args_with_config(parser)
    assert (args.learning_rate, args.hidden_size, args.output_dir) == (0.5, 128, "b")


def test_unset_arguments_keep_config_values(tmp_path, monkeypatch):
    config = write(tmp_path / "config.yaml", "output_dir: a\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", config])
    assert parse_args_with_config(parser).output_dir == "a"


def test_repository_configs_load():
    for path in ("configs/train/pretrain.yaml", "configs/train/rollout.yaml", "configs/evaluation.yaml"):
        assert "base" not in load_config(path)
    rollout = load_config("configs/train/rollout.yaml")
    assert rollout["rollout_steps"] == [2, 4, 8] and rollout["tendency_normalization"] is True


def test_experiment_registry():
    experiments = load_experiments("experiments/experiments.yaml")
    by_name = {experiment.name: experiment for experiment in experiments}
    pretrain_keys = set(load_config("configs/train/pretrain.yaml"))
    for experiment in experiments:
        assert set(load_config(experiment.config)) == pretrain_keys, experiment.name
        assert experiment.checkpoint == f"checkpoints/{experiment.name}/best"
    progression = [experiment.name for experiment in experiments if experiment.group == "progression"]
    assert progression[0] == "v0" and progression[-1] == "vars29_both_rollout"
    assert by_name["vars29"].compare_to == "v0"
    assert by_name["vars29_both_rollout"].init_from == "vars29_both"


def test_registry_rejects_unknown_references(tmp_path):
    registry = write(
        tmp_path / "experiments.yaml",
        "progression:\n  - {name: a, config: x.yaml, description: d, init_from: missing}\n",
    )
    with pytest.raises(ValueError):
        load_experiments(registry)
