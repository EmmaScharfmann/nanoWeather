# nanoWeather

Train and evaluate small global weather forecasting models.

## Setup

```bash
uv sync
```

## Layout

```
configs/                      YAML settings: data source and splits, training, evaluation, visualization
scripts/                      command-line entry points
src/nanoweather/
    processing/               dataset preparation, normalization, time features
    data/                     loading splits and training samples
    models/                   model configuration and code
    training/                 training loop
    evaluation/               metrics, baselines, evaluation protocol
    visualization/            forecast maps and animations
tests/
```

Loss options (`configs/train/v0.yaml`): `tendency_normalization` scales each variable's predicted change by its
typical change over one time step (`diff_std / std`), so all variables contribute comparably to the loss.
`loss_weighting: pressure` weights pressure-level variables in proportion to pressure, each source variable counting
once in total, and surface variables by `surface_loss_weights`. Validation losses are not comparable between runs
with different loss settings; compare models with `scripts/evaluate.py`.

`num_input_steps` sets how many consecutive states the model takes as input (the current one and previous ones,
one time step apart). Evaluation scores all models on the initializations that have enough previous steps for the
model that needs the most, so results stay comparable.

Rollout fine-tuning (`configs/train/rollout.yaml`) continues training a model from `init_from` with the loss
averaged over several consecutive forecast steps, each step taking the previous prediction as input, with gradients
through the whole rollout. Training runs in stages, one per entry of `rollout_steps` (e.g. `[2, 4, 8]`), each for
`num_train_epochs` epochs. The val loss is computed over the largest number of steps, so it is comparable across
stages. `gradient_checkpointing` reduces memory for long rollouts at the cost of extra compute.

The evaluation reports RMSE and the anomaly correlation coefficient (ACC, correlation of forecast and truth anomalies
from the climatology, area-weighted). The skill horizon is the lead time at which ACC falls below `acc_threshold`
(0.6 by default, the usual limit of a useful forecast). Results are in `<output_dir>/<model>.json` and
`<output_dir>/scores.png`.

Every script reads its settings from a config in `configs/`. Any setting can be overridden on the command line, and
`--config` selects another file.

Two datasets are configured: `configs/data.yaml` (6 variables, v0) and `configs/data_29vars.yaml` (z, t, u, v and
specific humidity at 5 pressure levels, plus 4 surface variables). Select one with `--data_config`. Evaluation scores
all models on the variables they share, so 6-variable models can be compared with 29-variable models on the
29-variable dataset.

## Usage

```bash
# Data: ERA5 at 5.625 deg from WeatherBench 2 (configs/data.yaml)
uv run scripts/prepare_data.py

# Training (configs/train/v0.yaml). Run the overfit check first.
uv run scripts/train.py --overfit_one_batch
uv run scripts/train.py
uv run scripts/train.py --output_dir checkpoints/epochs50 --num_train_epochs 50
uv run scripts/train.py --tendency_normalization --loss_weighting pressure
uv run scripts/train.py --num_input_steps 2
uv run scripts/train.py --config configs/train/rollout.yaml --init_from checkpoints/vars29_both/best

# Evaluation against persistence and climatology: RMSE, ACC and skill horizon (configs/evaluation.yaml)
uv run scripts/evaluate.py --checkpoint checkpoints/v0/best
uv run scripts/evaluate.py --checkpoint checkpoints/v0/best checkpoints/epochs50/best --output_dir results/compare

# Bias maps, RMSE by latitude and zonal spectra over all initializations (configs/diagnostics.yaml)
uv run scripts/diagnose.py --checkpoint checkpoints/v0/best

# Forecast maps and animation (configs/visualization.yaml)
uv run scripts/visualize.py --checkpoint checkpoints/v0/best --init_time 2020-01-15T00
```

## Using the library

```python
import torch

from nanoweather.data.datasets import ForecastPairs, WeatherData
from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting


data = WeatherData("data/nanoweather_v0", split="train", store_name="nanoweather_v0.zarr")
loader = torch.utils.data.DataLoader(ForecastPairs(data), batch_size=4, shuffle=True)
batch = next(iter(loader))
static = torch.from_numpy(data.static)

model = NanoWeatherForWeatherForecasting(NanoWeatherConfig())
output = model(**batch, static=static)
forecast = model.rollout(batch["states"], static=static, num_steps=20)
forecast = data.feature_extractor.denormalize(forecast)
```

- `states`: the 6 variables on the 32x64 grid at time t, normalized. `labels`: the same one time step later.
- `static`: land-sea mask and orography, shared by all samples.
- Model inputs and outputs are normalized; `feature_extractor.denormalize` converts back to physical units.

## Tests

```bash
uv run pytest
```

Findings and results are logged in `notes.md`.
