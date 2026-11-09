"""Prepares dataset from ERA5 (5.625 deg, 6-hourly).

Output directory:
    nanoweather_v0.zarr/        Zarr store with groups:
        train/ val/ test/       variables with dims (time, latitude, longitude), float32
        static/                 land_sea_mask, orography, dims (latitude, longitude)
        climatology/            WeatherBench 2 ERA5 climatology 1990-2019, dims (dayofyear, hour, latitude, longitude)
    preprocessor_config.json    `NanoWeatherFeatureExtractor` with normalization statistics (train years only)
    README.md                   dataset card

Conventions: dims are (..., latitude, longitude), latitude south to north, longitude 0 to 354.375.
WeatherBench 2 stores (..., longitude, latitude), so the arrays are transposed.

Usage:
    uv run prepare_data.py --output_dir data/nanoweather_v0
    uv run prepare_data.py --output_dir data/nanoweather_v0 --push_to_hub <user>/nanoweather-era5-5.625deg
"""

import argparse
import logging
import os
import time

import numpy as np
import xarray as xr

from feature_extraction_nanoweather import NanoWeatherFeatureExtractor


logger = logging.getLogger(__name__)

WEATHERBENCH2_URL = "https://storage.googleapis.com/weatherbench2/datasets"
ERA5_URL = f"{WEATHERBENCH2_URL}/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
CLIMATOLOGY_URL = f"{WEATHERBENCH2_URL}/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"
STORE_NAME = "nanoweather_v0.zarr"

# Short name to (WeatherBench 2 variable name, pressure level in hPa, or None for surface fields)
VARIABLES = {
    "z500": ("geopotential", 500),
    "t850": ("temperature", 850),
    "t2m": ("2m_temperature", None),
    "u10": ("10m_u_component_of_wind", None),
    "v10": ("10m_v_component_of_wind", None),
    "msl": ("mean_sea_level_pressure", None),
}

# Short name to WeatherBench 2 variable name
STATIC_VARIABLES = {
    "land_sea_mask": "land_sea_mask",
    "orography": "geopotential_at_surface",  # m**2 s**-2, i.e. surface height times g
}

SPLITS = {
    "train": ("2000-01-01T00", "2015-12-31T18"),
    "val": ("2016-01-01T00", "2016-12-31T18"),
    "test": ("2020-01-01T00", "2021-01-10T18"),
}

TIME_CHUNK_SIZE = 1464  # ~ one year of 6h steps/chunk


def open_weatherbench2(url: str) -> xr.Dataset:
    """Opens a public WeatherBench 2 Zarr store over plain HTTPS (no Google Cloud credentials needed)."""
    return xr.open_zarr(url, consolidated=True, zarr_format=2)


def to_standard_layout(dataset: xr.Dataset) -> xr.Dataset:
    """Transposes `(..., longitude, latitude)` to `(..., latitude, longitude)`, with latitude ascending."""
    return dataset.transpose(..., "latitude", "longitude").sortby("latitude").sortby("longitude")


def select_variables(dataset: xr.Dataset) -> xr.Dataset:
    """Selects the variables and pressure levels in `VARIABLES`, renamed to their short names."""
    selected = {}
    for name, (source_name, level) in VARIABLES.items():
        variable = dataset[source_name]
        if level is not None:
            variable = variable.sel(level=level, drop=True)
        variable = variable.astype("float32")
        variable.attrs = {
            "long_name": dataset[source_name].attrs.get("long_name", source_name),
            "units": dataset[source_name].attrs.get("units", ""),
            "source_variable": source_name,
            "level_hpa": level if level is not None else "surface",
        }
        selected[name] = variable
    return to_standard_layout(xr.Dataset(selected))


def clear_encoding(dataset: xr.Dataset) -> xr.Dataset:
    """Drops the source chunking and compression encodings, which conflict with our own chunking on write."""
    for name in dataset.variables:
        dataset[name].encoding = {}
    return dataset


def compute_statistics(train: xr.Dataset) -> dict[str, list]:
    """
    Computes per-variable statistics on the training years only, so no information leaks from val or test.

    Returns a dict with `mean` and `std` (for normalizing states), `diff_std` (standard deviation of the 6h change,
    for tendency normalization) and `units`, each a list in the order of `VARIABLES`.
    """
    statistics = {"mean": [], "std": [], "diff_std": [], "units": []}
    for name in VARIABLES:
        values = train[name].values.astype(np.float64)  # (time, latitude, longitude)
        statistics["mean"].append(float(values.mean()))
        statistics["std"].append(float(values.std()))
        statistics["diff_std"].append(float(np.diff(values, axis=0).std()))
        statistics["units"].append(train[name].attrs["units"])
    return statistics


def write_dataset_card(output_dir: str, feature_extractor: NanoWeatherFeatureExtractor, num_steps: dict[str, int]):
    variable_rows = "\n".join(
        f"| `{name}` | {VARIABLES[name][0]} | {VARIABLES[name][1] or 'surface'} | {units} | "
        f"{mean:.4g} | {std:.4g} | {diff_std:.4g} |"
        for name, units, mean, std, diff_std in zip(
            feature_extractor.variables,
            feature_extractor.units,
            feature_extractor.mean,
            feature_extractor.std,
            feature_extractor.diff_std,
        )
    )
    split_rows = "\n".join(
        f"| {split} | {start} | {end} | {num_steps[split]} |" for split, (start, end) in SPLITS.items()
    )
    card = f"""---
tags: [weather, climate, era5, weatherbench2, nanoweather]
---

# nanoWeather v0: ERA5 at 5.625 deg, 6-hourly

Small, ML-ready subset of ERA5 for training and evaluating small global weather models.
Derived from the WeatherBench 2 regridded ERA5 (64x32 equiangular grid, conservative regridding).

## Layout

- `{STORE_NAME}`: Zarr store with groups `train`, `val`, `test`, `static` and `climatology`.
- `preprocessor_config.json`: normalization statistics, computed on the train split only.

Dims are `(time, latitude, longitude)`, latitude south to north, longitude 0 to 354.375.
The test split includes 10 days of 2021 so that 2020 initializations can be evaluated up to 10 days.
`climatology` is the WeatherBench 2 ERA5 climatology (1990-2019), dims `(dayofyear, hour, latitude, longitude)`.

```python
import xarray as xr

train = xr.open_zarr("hf://datasets/<repo_id>/{STORE_NAME}", group="train")
```

## Variables (statistics from the train split)

| name | source | level (hPa) | units | mean | std | 6h change std |
|---|---|---|---|---|---|---|
{variable_rows}

## Splits

| split | start | end | time steps |
|---|---|---|---|
{split_rows}

## Source and license

ERA5 (Hersbach et al., 2020, Copernicus Climate Change Service), as regridded by WeatherBench 2 (Rasp et al., 2024).
Check the Copernicus licence terms before redistributing.
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as writer:
        writer.write(card)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=str, default="data/nanoweather_v0")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Dataset repository id, e.g. user/repo.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)
    store_path = os.path.join(args.output_dir, STORE_NAME)
    era5 = open_weatherbench2(ERA5_URL)

    num_steps = {}
    statistics = None
    for split, (start, end) in SPLITS.items():
        start_time = time.time()
        logger.info(f"Downloading {split} ({start} to {end})")
        dataset = select_variables(era5.sel(time=slice(start, end))).load()
        for name in dataset.data_vars:
            if not np.isfinite(dataset[name].values).all():
                raise ValueError(f"Found non-finite values in `{name}` for split `{split}`.")
        num_steps[split] = dataset.sizes["time"]
        if split == "train":
            statistics = compute_statistics(dataset)
        # Chunk sizes are set through the encoding rather than `.chunk()`, which would require dask.
        encoding = {
            name: {"chunks": (min(TIME_CHUNK_SIZE, dataset.sizes["time"]), *dataset[name].shape[1:])}
            for name in dataset.data_vars
        }
        clear_encoding(dataset).to_zarr(store_path, group=split, mode="w", zarr_format=2, encoding=encoding)
        logger.info(f"Wrote {split}: {num_steps[split]} time steps in {time.time() - start_time:.0f}s")

    static = xr.Dataset({name: era5[source] for name, source in STATIC_VARIABLES.items()}).astype("float32")
    static = to_standard_layout(static).load()
    clear_encoding(static).to_zarr(store_path, group="static", mode="w", zarr_format=2)

    climatology = select_variables(open_weatherbench2(CLIMATOLOGY_URL))
    climatology = climatology.transpose("dayofyear", "hour", "latitude", "longitude").load()
    clear_encoding(climatology).to_zarr(store_path, group="climatology", mode="w", zarr_format=2)
    logger.info("Wrote static fields and climatology")

    feature_extractor = NanoWeatherFeatureExtractor(
        variables=list(VARIABLES),
        static_variables=list(STATIC_VARIABLES),
        static_mean=[float(static[name].mean()) for name in STATIC_VARIABLES],
        static_std=[float(static[name].std()) for name in STATIC_VARIABLES],
        **statistics,
    )
    feature_extractor.save_pretrained(args.output_dir)
    write_dataset_card(args.output_dir, feature_extractor, num_steps)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="dataset")
        logger.info(f"Pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
