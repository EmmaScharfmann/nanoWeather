"""Building the training dataset from a source Zarr store: variable selection, layout, splits and statistics."""

import logging
import os
import time

import numpy as np
import xarray as xr

from .feature_extraction_nanoweather import NanoWeatherFeatureExtractor


logger = logging.getLogger(__name__)


def open_zarr_store(path_or_url: str) -> xr.Dataset:
    """Opens a consolidated Zarr store (format 2) from a local path or a URL."""
    return xr.open_zarr(path_or_url, consolidated=True, zarr_format=2)


def to_standard_layout(dataset: xr.Dataset) -> xr.Dataset:
    """Puts latitude and longitude as the last two dimensions, both sorted in ascending order."""
    return dataset.transpose(..., "latitude", "longitude").sortby("latitude").sortby("longitude")


def select_variables(dataset: xr.Dataset, variables: dict[str, dict]) -> xr.Dataset:
    """
    Selects variables and pressure levels from `dataset` and renames them.

    Args:
        dataset (`xr.Dataset`):
            Source dataset.
        variables (`dict[str, dict]`):
            Maps each new name to `{"source": source_name}` for a surface field, or
            `{"source": source_name, "level": level}` for one pressure level of a 3D field.

    Returns:
        `xr.Dataset` with float32 variables in the standard layout.
    """
    selected = {}
    for name, spec in variables.items():
        source_name, level = spec["source"], spec.get("level")
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
    """Removes the chunking and compression settings inherited from the source store."""
    for name in dataset.variables:
        dataset[name].encoding = {}
    return dataset


def compute_statistics(dataset: xr.Dataset) -> dict[str, list]:
    """
    Per-variable statistics over all time steps and grid points.

    Returns:
        `dict` with lists in the order of `dataset.data_vars`: `mean`, `std`, `diff_std` (standard deviation of the
        change between consecutive time steps) and `units`.
    """
    statistics = {"mean": [], "std": [], "diff_std": [], "units": []}
    for name, variable in dataset.data_vars.items():
        values = variable.values.astype(np.float64)
        statistics["mean"].append(float(values.mean()))
        statistics["std"].append(float(values.std()))
        statistics["diff_std"].append(float(np.diff(values, axis=0).std()))
        statistics["units"].append(variable.attrs.get("units", ""))
    return statistics


def write_split(dataset: xr.Dataset, store_path: str, group: str, time_chunk_size: int):
    encoding = {
        name: {"chunks": (min(time_chunk_size, dataset.sizes["time"]), *dataset[name].shape[1:])}
        for name in dataset.data_vars
    }
    clear_encoding(dataset).to_zarr(store_path, group=group, mode="w", zarr_format=2, encoding=encoding)


def prepare_dataset(config: dict, output_dir: str | os.PathLike) -> tuple[NanoWeatherFeatureExtractor, dict[str, int]]:
    """
    Downloads, selects and writes the splits, static fields and climatology, and fits the normalization.

    Statistics are computed on the `train` split only.

    Args:
        config (`dict`):
            Data config, see `configs/data.yaml`.
        output_dir (`str` or `os.PathLike`):
            Directory for the Zarr store and `preprocessor_config.json`.

    Returns:
        The fitted feature extractor, and the number of time steps of each split.
    """
    os.makedirs(output_dir, exist_ok=True)
    store_path = os.path.join(output_dir, config["store_name"])
    source = open_zarr_store(config["source"]["era5_url"])

    num_steps, statistics = {}, None
    for split, (start, end) in config["splits"].items():
        start_time = time.time()
        logger.info(f"Preparing {split} ({start} to {end})")
        dataset = select_variables(source.sel(time=slice(start, end)), config["variables"]).load()
        for name in dataset.data_vars:
            if not np.isfinite(dataset[name].values).all():
                raise ValueError(f"Found non-finite values in `{name}` for split `{split}`.")
        num_steps[split] = dataset.sizes["time"]
        if split == "train":
            statistics = compute_statistics(dataset)
        write_split(dataset, store_path, split, config["time_chunk_size"])
        logger.info(f"Wrote {split}: {num_steps[split]} time steps in {time.time() - start_time:.0f}s")

    static_names = config["static_variables"]
    static = xr.Dataset({name: source[source_name] for name, source_name in static_names.items()}).astype("float32")
    static = to_standard_layout(static).load()
    clear_encoding(static).to_zarr(store_path, group="static", mode="w", zarr_format=2)

    climatology = select_variables(open_zarr_store(config["source"]["climatology_url"]), config["variables"])
    climatology = climatology.transpose("dayofyear", "hour", "latitude", "longitude").load()
    clear_encoding(climatology).to_zarr(store_path, group="climatology", mode="w", zarr_format=2)
    logger.info("Wrote static fields and climatology")

    feature_extractor = NanoWeatherFeatureExtractor(
        variables=list(config["variables"]),
        static_variables=list(static_names),
        static_mean=[float(static[name].mean()) for name in static_names],
        static_std=[float(static[name].std()) for name in static_names],
        **statistics,
    )
    feature_extractor.save_pretrained(output_dir)
    return feature_extractor, num_steps


def write_dataset_card(
    output_dir: str | os.PathLike,
    card_config: dict,
    data_config: dict,
    feature_extractor: NanoWeatherFeatureExtractor,
    num_steps: dict[str, int],
):
    """Fills the dataset card template (see `configs/dataset_card.yaml`) and writes it to `README.md`."""
    variable_rows = [
        card_config["variable_row"].format(
            name=name,
            source=data_config["variables"][name]["source"],
            level=data_config["variables"][name].get("level", "surface"),
            units=units,
            mean=mean,
            std=std,
            diff_std=diff_std,
        )
        for name, units, mean, std, diff_std in zip(
            feature_extractor.variables,
            feature_extractor.units,
            feature_extractor.mean,
            feature_extractor.std,
            feature_extractor.diff_std,
        )
    ]
    split_rows = [
        card_config["split_row"].format(split=split, start=start, end=end, num_steps=num_steps[split])
        for split, (start, end) in data_config["splits"].items()
    ]
    card = card_config["template"].format(
        store_name=data_config["store_name"],
        variable_rows="\n".join(variable_rows),
        split_rows="\n".join(split_rows),
        era5_url=data_config["source"]["era5_url"],
    )
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as writer:
        writer.write(card)
