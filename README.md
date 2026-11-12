# nanoWeather

Train and evaluate small global weather forecasting models. v0 is a vision transformer of about 5M parameters on
ERA5 at 5.625° (a 32 × 64 grid): it predicts the atmosphere 6 hours ahead and is run step by step to forecast 10 days.
One training run takes minutes on a single GPU.

Results and explanations are in `notebooks/results.ipynb`.

## Setup

```bash
uv sync
```

## The recipe

```bash
uv run scripts/prepare_data.py                                 # ERA5, 29 variables, from WeatherBench 2 (~11 GB)
uv run scripts/train.py --overfit_one_batch                    # sanity check: the loss should keep decreasing
uv run scripts/train.py                                        # step 1: one-step training   -> checkpoints/pretrain
uv run scripts/train.py --config configs/train/rollout.yaml    # step 2: rollout fine-tuning -> checkpoints/rollout
uv run scripts/evaluate.py --checkpoint checkpoints/rollout/best
```

- **Data**: z, t, u, v and specific humidity at 250, 500, 700, 850 and 925 hPa, plus 2 m temperature, 10 m wind and
  mean sea level pressure. Training years 2000-2015, validation 2016, test 2020.
- **Step 1**: the model predicts the change over 6 hours. The loss compares predicted and true changes, each variable
  scaled by its typical 6-hour change, with weights proportional to pressure for upper-air levels and a higher weight
  for 2 m temperature among surface variables.
- **Step 2**: the model is fine-tuned on 2, then 4, then 8 consecutive steps, feeding its own predictions back as
  inputs, so it learns to handle its own errors.
- **Evaluation**: forecasts from 00 and 12 UTC every day of 2020, to 10 days, against ERA5. RMSE, anomaly correlation
  (ACC), and skill horizon (lead time at which ACC falls below 0.6), compared with persistence and climatology.

## Latitude settings of the loss

Two options in the training config address errors that grow toward the poles:

- `latitude_weight_exponent`: rows are weighted by `cos(latitude) ** exponent` in the training loss. 1 (default)
  weights by area; 0.5 gives polar rows about 3 times more weight; 0 weights all rows equally. Evaluation always stays
  area-weighted, so scores remain comparable.
- `tendency_normalization_by_latitude`: the predicted change is scaled per variable and latitude row, by the typical
  6-hour change of that row in the training data, so every row's errors count relative to its own variability.

## Architectures

Set `architecture` in the training config:

- `vit` (default): a vision transformer on 2 × 2 patches of the latitude/longitude grid, with attention over all
  patches.
- `mesh`: encode-process-decode on an icosahedral mesh, as in GraphCast and WeatherNext 2. Grid inputs are encoded
  onto mesh nodes spread evenly over the sphere, processed by transformer layers where each node attends to its
  nearest nodes, and decoded back to the grid. The processor no longer sees the crowding of the grid near the poles.
  Inputs and outputs stay on the grid, so data, evaluation and diagnostics are unchanged.

The mesh model computes attention over neighbors only (`attn_implementation: sparse`) or over all pairs with a mask
(`dense`), whichever is faster for each part of the model (`auto`, the default). The crossover depends on the
hardware: `uv run scripts/benchmark_models.py` measures both on your machine.

```bash
uv run scripts/train.py --config experiments/configs/vars29_mesh.yaml --output_dir checkpoints/vars29_mesh
```

## Configs

Every script reads a YAML config from `configs/`, and `--set key=value` overrides any setting:

```bash
uv run scripts/train.py --output_dir checkpoints/test --set num_train_epochs=5 learning_rate=1e-3
```

| Config | Used by |
|---|---|
| `configs/data.yaml` | `prepare_data.py`, and all scripts through `data_config` |
| `configs/train/pretrain.yaml` | `train.py` (default), step 1 |
| `configs/train/rollout.yaml` | `train.py --config ...`, step 2, inherits from `pretrain.yaml` |
| `configs/evaluation.yaml` | `evaluate.py` |
| `configs/diagnostics.yaml` | `diagnose.py`: bias maps, RMSE by latitude (raw and relative to natural variability), spectra |
| `configs/visualization.yaml` | `visualize.py`: maps and animation of one forecast |

A config can start from another with `base: <path>` and list only what differs.

## Experiments

`experiments/experiments.yaml` lists every model trained in the v0 study, in two groups: the **progression** (each
model adds one change to the previous one, from the original v0 to the recipe) and the **ablations** (one change
compared with a reference). Each experiment has a small config in `experiments/configs/` listing what differs from the
recipe.

```bash
uv run scripts/prepare_data.py --config experiments/configs/data_6vars.yaml   # dataset of the original v0
uv run scripts/run_experiments.py --dry_run                                    # what would be trained
uv run scripts/run_experiments.py                                              # trains missing checkpoints
uv run scripts/evaluate.py --experiments experiments/experiments.yaml --output_dir results/experiments
uv run scripts/diagnose.py --checkpoint checkpoints/v0/best
uv run scripts/diagnose.py --checkpoint checkpoints/vars29_both_rollout/best
uv run scripts/diagnose_latitude.py --checkpoint checkpoints/vars29_both_rollout/best checkpoints/vars29_mesh_rollout/best
uv run scripts/visualize.py --checkpoint checkpoints/vars29_both_rollout/best
```

Checkpoints go to `checkpoints/<experiment>/`, results to `results/`; neither is versioned. All experiments are
evaluated together, on the variables and forecasts they have in common, so their scores are directly comparable.
Then open `notebooks/results.ipynb`.

## Layout

```
configs/              settings of the recipe
experiments/          the experiments of the v0 study: registry and configs
notebooks/            results.ipynb (results and explanations), check_data.ipynb (checks of a prepared dataset)
scripts/              command-line entry points
src/nanoweather/
    processing/       dataset preparation, normalization, time features
    data/             loading splits and training samples
    models/           model configurations and code: vit (modeling_nanoweather.py), mesh, shared code, auto loading
    training/         training loop, loss weights
    evaluation/       metrics, baselines, evaluation, diagnostics
    visualization/    forecast maps and animations
    experiments.py    reading the experiment registry
tests/
```

## Tests

```bash
uv run pytest
```

Decisions and findings are logged in `notes.md`.
