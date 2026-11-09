# nanoWeather notes

Running log of decisions, results, surprises and failed attempts. Raw material for the "what worked / what didn't"
write-up. Newest entries at the top of each section.

## v0 baseline (2026-11-09)

### Setup

- Data: WeatherBench 2 ERA5, 5.625 deg (64x32), 6-hourly. Train 2000-2015, val 2016, test 2020.
- Variables: z500, t850, t2m, u10, v10, msl. Static: land-sea mask, orography.
- Model: ViT, 2x2 patches (512 tokens), 6 layers, hidden 256, 8 heads, 4.88M parameters. Predicts the 6h change.
  Head initialized at zero, so the untrained model is the persistence forecast.
- Training: one 6h step, latitude-weighted MSE, AdamW (lr 5e-4, wd 0.05), 1000 warmup steps then cosine, batch 32,
  20 epochs (~14.6k steps), bf16.
- Evaluation: 732 initializations (00 and 12 UTC through 2020), rollout to 5 days, latitude-weighted RMSE vs ERA5.

### Results

RMSE on 2020, read from `results/rmse.png` (replace with exact values from `results/*.json`):

| variable | lead | nanoweather-v0 | persistence | climatology |
|---|---|---|---|---|
| z500 (m2/s2) | 1 day | ~165 | ~575 | ~800 |
| z500 | 3 days | ~455 | ~925 | ~800 |
| z500 | 5 days | ~705 | ~1020 | ~800 |
| t850 (K) | 1 day | ~0.95 | ~2.7 | ~3.2 |
| t850 | 3 days | ~2.0 | ~4.0 | ~3.2 |
| t850 | 5 days | ~3.0 | ~4.3 | ~3.2 |
| t2m (K) | 1 day | ~0.85 | ~1.7 | ~2.4 |
| t2m | 3 days | ~1.55 | ~2.6 | ~2.4 |
| t2m | 5 days | ~2.2 | ~2.9 | ~2.4 |

v0 success criterion met: beats persistence and climatology on all variables up to day 5.

Rough comparison (to verify): the original WeatherBench at 5.625 deg reported z500 RMSE at 3 / 5 days of about
626 / 757 for a direct CNN, 489 / 743 for IFS at T42, and ~270-370 at day 3 for the best deep learning baselines.
Different test years (2017-2018), so only indicative.

### Findings

- **Training is cheap**: 13 s per epoch, ~5 min for the full run on one GPU. Ablations can be run in batches.
- **Still improving at epoch 20**: the best val loss (0.00390) was the last epoch. Undertrained.
- **Persistence at step 1 is already a decent baseline**: overfit check started at loss 0.051 (persistence error, in
  normalized units) because the 6h change is small relative to each variable's spread. The first overfit check with
  300 steps reached only 14.7% of that, still decreasing; 1000 steps needed. The 10% threshold was too strict for
  300 steps.
- **Diurnal cycle**: persistence t2m RMSE zigzags with a 24h period (day vs night at 12h leads). The model shows no
  zigzag despite having no time-of-day input: it infers the phase of the day from the temperature field itself.
- **Error growth is roughly linear** in lead time up to 5 days; z500 should cross climatology around day 6-7.
  To check: does the forecast get smoother with lead time (MSE training rewards blurring when uncertain)?

### Engineering notes

- WeatherBench 2 stores arrays as (time, longitude, latitude), latitude south to north. Transposed in `prepare_data.py`.
- z500 and t850 chunks contain all 13 pressure levels: selecting one level downloads all 13 (7.7 MB per chunk vs
  0.58 MB for a surface variable). Most of the ~5 GB download.
- gcsfs failed behind a proxy; plain HTTPS on `storage.googleapis.com` works without credentials.
- `.chunk()` in xarray requires dask; setting chunks through the Zarr `encoding` does not.

## Ablation round 1 (in progress)

- Training length: 50, 100 epochs. Model size: 3 / 12 layers, hidden 128 / 384, at 20 epochs.
  Caveat: if longer training helps a lot, redo the size comparison at the better length.
- Time features: sunlight (cosine of the solar zenith angle, clipped at 0) at t and t + 6h, plus sin/cos of day of
  year. Sunlight rather than UTC hour because local time depends on longitude; close to GraphCast's use of
  top-of-atmosphere solar radiation.

## Next

- [ ] Animation and snapshots (`visualize.py`) for a summer and a winter initialization.
- [ ] 10-day evaluation (`--num_steps 40`).
- [ ] Ablations: 50 / 100 epochs, time-of-day and time-of-year inputs, tendency normalization, rollout fine-tuning
      (2, 4, 8 steps), model size.