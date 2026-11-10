"""The list of trained models of a study, read from a registry file (see `experiments/experiments.yaml`)."""

import os
from dataclasses import dataclass

from .config import load_yaml


GROUPS = ("progression", "ablations")


@dataclass
class Experiment:
    """
    One trained model.

    Attributes:
        name (`str`): Name in results and plots. The checkpoint is `<checkpoint_root>/<name>/best`.
        group (`str`): `"progression"` (each step adds one change to the previous one) or `"ablations"`.
        config (`str`): Training config.
        description (`str`): What this model changes.
        init_from (`str`, *optional*): Name of the experiment this one is fine-tuned from.
        compare_to (`str`, *optional*): Name of the experiment this one should be compared with.
        checkpoint (`str`): Directory of the best checkpoint.
    """

    name: str
    group: str
    config: str
    description: str
    init_from: str | None = None
    compare_to: str | None = None
    checkpoint: str = ""


def load_experiments(path: str | os.PathLike, checkpoint_root: str = "checkpoints") -> list[Experiment]:
    """
    Reads and checks a registry: names are unique, and `init_from` and `compare_to` name other experiments.

    In the progression, each experiment is compared with the previous one unless `compare_to` says otherwise.
    """
    registry = load_yaml(path)
    experiments = []
    for group in GROUPS:
        previous = None
        for entry in registry.get(group, []):
            experiment = Experiment(group=group, **entry)
            if group == "progression" and experiment.compare_to is None:
                experiment.compare_to = previous
            experiment.checkpoint = os.path.join(checkpoint_root, experiment.name, "best")
            experiments.append(experiment)
            previous = experiment.name

    names = [experiment.name for experiment in experiments]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Experiment names must be unique: {sorted(duplicates)}")
    for experiment in experiments:
        for reference in (experiment.init_from, experiment.compare_to):
            if reference is not None and reference not in names:
                raise ValueError(f"`{experiment.name}` refers to unknown experiment `{reference}`.")
    return experiments
