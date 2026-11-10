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

## Round 1 results (training length, model size, time features)

- Smaller models (hidden 128, 3 layers) are clearly worse at all lead times.
- 50 / 100 epochs, 12 layers and hidden 384 cluster a few percent below v0 (z500 day 5: ~680 vs ~705). To check
  against seed-to-seed variation.
- Time features (sunlight, time of year): no gain, including for t2m.
- Error growth rate is the same for all variants: differences are set in the first day and carried along.
- Interpretation: the limit is not capacity or training time.

## Single forecast, 2020-01-15 (v0)

- Largest t2m errors over northern continents in winter (Canada, Siberia), already +-10 C at day 3.
- Error bands along the pole rows at day 5 (z500, t2m). Pole rows have near-zero weight in the loss.
- z500 day 5: dipoles in the storm tracks, i.e. systems in the wrong place.
- Hypothesis: the 6 input variables do not describe the atmosphere well enough (no winds aloft, no humidity, two
  pressure levels). Consistent with the round 1 plateau.

## Round 2 (in progress)

- Diagnostics over all of 2020: bias maps, RMSE by latitude, zonal spectra (`scripts/diagnose.py`).
  On fake noise data, the spectrum of a small model shows a spike at wavenumber 32 (period of 2 grid points, the
  patch size): check whether v0 has this patch artifact.
- 29 variables: z, t, u, v, q at 250/500/700/850/925 hPa plus t2m, u10, v10, msl. Each source chunk holds all 13
  levels and is downloaded once whatever the number of levels selected (~11 GB in total).

## Round 2 results (29 variables, patch size, diagnostics)

- 29 variables: z500 better by 14-28%, with a lower error growth rate (first change of slope so far). t2m worse at
  all leads (+25-30% at day 1), worst near the poles (warm drift up to +7.5 K over Antarctica at day 5). t850 worse up
  to day 2, then better.
- Patch size 1: no change in RMSE compared with v0.
- A first set of 29-variable diagnostics was inconsistent with the evaluation (RMSE several times too high, row
  stripes); a re-run was consistent. Cause not identified, not reproduced.
- Comparison with GraphCast, Stormer and WeatherNext 2: all normalize the predicted change per variable by the
  spread of its change over one step, weight variables in the loss (pressure-proportional for levels; GraphCast uses
  t2m 1.0 and other surface variables 0.1), and train on multi-step rollouts. GraphCast and WeatherNext 2 take two
  input time steps. We had none of these.

## Round 3 (in progress): loss

- Tendency normalization (`residual_scale = diff_std / std`) and pressure loss weights, separately and together,
  on 29 variables; both together on 6 variables. With tendency normalization, the overfit check starts at a loss
  of about 1 (persistence error in units of the typical change).

## Round 3 results (loss)

- Tendency normalization is the main effect: on 29 variables it brings t2m back to v0's level and gives the best
  z500 and t850 (about -23% and -17% vs v0 at day 5), with a lower error growth rate.
- Pressure weights alone fix t2m but do not improve z500 or t850. With tendency normalization, they trade some
  upper-air skill for the best t2m (about -10% vs v0).
- vars29_both beats v0 on all three variables at day 5 (z500 -17%, t850 -13%, t2m -10%). New baseline.
- Both options also improve the 6-variable model by 6-9%.
- Extrapolated, the baseline still reaches the climatology RMSE around 6-7 days (z500, t850) and 6 days (t2m).
  State-of-the-art models keep useful z500 skill to about 9-10 days.

## Round 4 (in progress): ACC and two input time steps

- ACC and skill horizon (lead time where ACC < 0.6) added to the evaluation; evaluation extended to 10 days.
- Two input time steps (t - 6h, t), as GraphCast and WeatherNext 2.
- Second seed of the baseline, for run-to-run variation.

## Round 4 results (ACC, two input steps, seeds)

- Seed-to-seed variation: under 1% in RMSE, about 0.1 day in skill horizon.
- Skill horizon (ACC > 0.6): v0 4.9 / 4.6 / 4.7 days (z500 / t850 / t2m); vars29_both 6.1-6.2 / 5.6-5.7 / 5.5-5.6.
- Two input steps: better at 6h by 3-7% on all variables, but worse once both inputs are model predictions (from
  about 12-18h for z500 and msl, about 2 days for t850 and t2m). Skill horizon unchanged. Interpretation: the model is
  not robust to its own predictions as history, expected to be fixed by rollout training.

## Round 5 results (rollout fine-tuning)

- Fine-tuning on 2, 4 and 8 step rollouts (one epoch each, learning rate 1e-4) improves day-3 RMSE by 12-16% and the
  skill horizon by 0.8-0.9 days: z500 6.1 -> 7.0 days, t850 5.6 -> 6.4, t2m 5.5 -> 6.3.
- The two-step model gains slightly more but stays slightly behind (z500 horizon 6.9 vs 7.0 days).
- Errors still rise toward both poles after fine-tuning, though lower than before. Literature: the fix is a spherical
  representation (mesh as GraphCast and WeatherNext 2, spherical harmonics as SFNO, cubed sphere, reduced Gaussian
  grids); transformers on lat-lon grids (ClimaX, Stormer) only use area-weighted losses. Left for v1.

## Reorganization

- Recipe = vars29_both + rollout fine-tuning, as defaults (`configs/train/pretrain.yaml`, `configs/train/rollout.yaml`).
- Every trained model is listed in `experiments/experiments.yaml`, with a config per experiment listing only its
  differences from the recipe. All experiments are evaluated together with the final protocol (10 days, ACC).
- Results and explanations: `notebooks/results.ipynb`.

## Next

- [ ] Animation and snapshots (`visualize.py`) for a summer and a winter initialization.
- [ ] 10-day evaluation (`--num_steps 40`).
- [ ] Ablations: 50 / 100 epochs, time-of-day and time-of-year inputs, tendency normalization, rollout fine-tuning
      (2, 4, 8 steps), model size.