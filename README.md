# nanoWeather

Train and evaluate small global weather forecasting models.

Code follows the transformers conventions: `<file>_nanoweather.py` naming, `NanoWeather<Class>` class names,
explicit code over indirection, Google-style docstrings, Apache 2.0 headers, `logging` instead of `print`,
optional dependencies imported lazily, and ruff with the transformers settings (`uv run ruff check . && uv run ruff format .`).

## Setup

```bash
uv sync                  # creates .venv and installs dependencies from uv.lock
uv sync --extra plot     # also installs matplotlib
uv run jupyter lab       # notebooks (jupyterlab is in the dev group, installed by default)
```

## Step 1: data

```bash
uv run prepare_data.py --output_dir data/nanoweather_v0
uv run prepare_data.py --output_dir data/nanoweather_v0 --push_to_hub <user>/nanoweather-era5-5.625deg
```

Then run `experiments/01_check_data.ipynb` to check the data: every check is an `assert`, followed by plots.

## Step 2: evaluation and baselines

```bash
uv run evaluate.py --data_path data/nanoweather_v0 --output_dir results
```

Scores persistence and climatology on 2020 (initializations every 12h, lead times 6h to 5 days) with
latitude-weighted RMSE. Writes `results/<forecaster>.json` and `results/rmse_baselines.png`.

## Step 3: model

```python
import torch

from configuration_nanoweather import NanoWeatherConfig
from data import ForecastPairs, WeatherData
from modeling_nanoweather import NanoWeatherForWeatherForecasting


data = WeatherData("data/nanoweather_v0", split="train")
pairs = ForecastPairs(data, lead_steps=1)
loader = torch.utils.data.DataLoader(pairs, batch_size=4, shuffle=True)

batch = next(iter(loader))  # dict with "states" and "labels", each (4, 6, 32, 64), normalized
states, targets = batch["states"], batch["labels"]
static = torch.from_numpy(data.static)  # (2, 32, 64): normalized land-sea mask and orography

model = NanoWeatherForWeatherForecasting(NanoWeatherConfig())  # 4.9M parameters
output = model(states, static=static, labels=targets)  # output.prediction (4, 6, 32, 64), output.loss
forecast = model.rollout(states, static=static, num_steps=20)  # (4, 20, 6, 32, 64): 5 days
forecast_physical = data.feature_extractor.denormalize(forecast)  # back to K, Pa, m**2 s**-2
model.save_pretrained("checkpoints/v0")  # config.json + model.safetensors
```

- `states`: the atmosphere at time t, the 6 variables on the 32x64 grid, normalized. Model input.
- `targets`: the same at t + 6h, passed as `labels` to get the loss.
- `static`: fields constant in time, shared by all samples, so no batch dimension.

Everything the model sees and returns is normalized. Use `data.feature_extractor.denormalize` to get physical units.

Tests (no data needed): `uv run pytest`

## Step 4: training

```bash
# 1. Sanity check: the loss on one fixed batch should drop below 10% of its initial value
uv run train.py --data_path data/nanoweather_v0 --overfit_one_batch

# 2. Full run: 20 epochs, about 15k steps
uv run train.py --data_path data/nanoweather_v0 --output_dir checkpoints/v0
```

Writes `checkpoints/v0/best/` (lowest val loss), `checkpoints/v0/last/` and `checkpoints/v0/training_history.json`.
Each checkpoint has `config.json`, `model.safetensors` and `preprocessor_config.json`.

## Step 5: evaluate the trained model

```bash
uv run evaluate.py --data_path data/nanoweather_v0 --output_dir results --checkpoint checkpoints/v0/best
```

Scores the model, persistence and climatology with the same protocol as step 2, and writes `results/rmse.png`.

## Step 6: visualize and evaluate to 10 days

```bash
uv run visualize.py --checkpoint checkpoints/v0/best --init_time 2020-07-01T00
uv run evaluate.py --data_path data/nanoweather_v0 --output_dir results/10days --checkpoint checkpoints/v0/best --num_steps 40
```

`visualize.py` writes a GIF (ERA5, forecast and error for each lead time) and a PNG with snapshots at 1, 3 and 5 days.
Findings are logged in `notes.md`.

## Ablations

Round 1 (training length, model size, time features) is in `experiments/round1_epochs_and_size.sh`. Options:

- `train.py --use_time_features`: adds the incoming sunlight at each grid point (at t and t + 6h) and the time of
  year as input channels.
- `evaluate.py --checkpoint a/best b/best --model_name a b`: scores several models on one plot. Names default to the
  checkpoint directory names.

## Files

| File | Content |
|---|---|
| `prepare_data.py` | Downloads WeatherBench 2 ERA5 at 5.625 deg, writes splits, static fields, climatology, normalization |
| `feature_extraction_nanoweather.py` | `NanoWeatherFeatureExtractor`: normalization, saved as `preprocessor_config.json` |
| `data.py` | `WeatherData` (one split in memory) and `ForecastPairs` (torch dataset of (t, t + 6h) pairs) |
| `evaluate.py` | Latitude-weighted RMSE by lead time, persistence and climatology baselines, plots |
| `configuration_nanoweather.py` | `NanoWeatherConfig` |
| `modeling_nanoweather.py` | `NanoWeatherModel` (backbone) and `NanoWeatherForWeatherForecasting` (head, loss, rollout) |
| `train.py` | Training: overfit-one-batch check, AdamW + cosine schedule, bf16, val loss per epoch, checkpoints |
| `visualize.py` | Forecast vs ERA5 animation and snapshots |
| `notes.md` | Decisions, results and findings |
| `tests/test_modeling_nanoweather.py` | Model tests: shapes, persistence at init, gradients, attention backends, save/load, rollout |
| `experiments/01_check_data.ipynb` | Sanity checks and plots for the prepared data |