"""Training loop for one-step forecasting with a latitude-weighted MSE loss."""

import json
import logging
import math
import os
import random
import time
from argparse import Namespace

import numpy as np
import torch

from ..data.datasets import ForecastPairs, WeatherData
from ..models.configuration_nanoweather import NanoWeatherConfig
from ..models.modeling_nanoweather import NanoWeatherForWeatherForecasting
from ..processing.feature_extraction_nanoweather import NUM_TIME_FEATURES
from ..processing.prepare import expand_variables
from .loss_weights import get_pressure_loss_weights


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup to the peak learning rate, then cosine decay to zero."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW, without weight decay on biases, normalization weights and position embeddings."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or "position_embeddings" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    parameter_groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(parameter_groups, lr=learning_rate, betas=(0.9, 0.95))


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def training_step(model, batch, static, optimizer, scheduler, max_grad_norm: float, use_bf16: bool) -> float:
    with torch.autocast(device_type=static.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        loss = model(**batch, static=static).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


@torch.no_grad()
def evaluate_loss(model, dataloader, static, device, use_bf16: bool) -> float:
    """Mean loss over a dataloader, weighted by batch size."""
    model.eval()
    total_loss, num_samples = 0.0, 0
    for batch in dataloader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch, static=static).loss
        batch_size = batch["states"].shape[0]
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    model.train()
    return total_loss / num_samples


def get_loss_settings(args: Namespace, data_config: dict, feature_extractor) -> dict:
    """`residual_scale` and `loss_weights` for the model config, from the training arguments."""
    variables = feature_extractor.variables
    residual_scale = None
    if args.tendency_normalization:
        residual_scale = (np.array(feature_extractor.diff_std) / np.array(feature_extractor.std)).tolist()
    if args.loss_weighting == "uniform":
        loss_weights = None
    elif args.loss_weighting == "pressure":
        specs = expand_variables(data_config)
        loss_weights = get_pressure_loss_weights(variables, specs, args.surface_loss_weights or {})
    else:
        raise ValueError(f"Unknown loss_weighting {args.loss_weighting!r}, expected 'uniform' or 'pressure'.")
    return {"residual_scale": residual_scale, "loss_weights": loss_weights}


def log_loss_settings(variables: list[str], settings: dict):
    if settings["loss_weights"] is not None:
        logger.info("Loss weights: " + ", ".join(f"{n} {w:.3f}" for n, w in zip(variables, settings["loss_weights"])))
    if settings["residual_scale"] is not None:
        logger.info(
            "Residual scale (diff_std / std): "
            + ", ".join(f"{n} {r:.3f}" for n, r in zip(variables, settings["residual_scale"]))
        )


def save_checkpoint(model, feature_extractor, output_dir: str):
    """Saves the model and its feature extractor, unwrapping a compiled model first."""
    getattr(model, "_orig_mod", model).save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)


def overfit_one_batch(model, train_pairs, static, device, args: Namespace, use_bf16: bool) -> float:
    """
    Trains on a single fixed batch and returns the final loss as a fraction of the loss at the first step.

    With a working model and data pipeline the loss keeps decreasing towards zero. With `predict_residual`, the loss at
    the first step is the error of the persistence forecast.
    """
    batch = next(iter(torch.utils.data.DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True)))
    batch = to_device(batch, device)
    optimizer = get_optimizer(model, args.learning_rate, weight_decay=0.0)
    initial_loss = None
    for step in range(1, args.overfit_steps + 1):
        loss = training_step(model, batch, static, optimizer, None, args.max_grad_norm, use_bf16)
        initial_loss = initial_loss if initial_loss is not None else loss
        if step == 1 or step % args.logging_steps == 0:
            logger.info(f"overfit step {step:5d} | loss {loss:.6f}")
    ratio = loss / initial_loss
    logger.info(f"Final loss {loss:.6f}, {ratio:.1%} of the loss at step 1")
    if ratio > 0.1:
        logger.warning(
            "The loss did not drop below 10% of its value at step 1. If it was still decreasing, increase "
            "overfit_steps; if it had flattened out, check the model and data pipeline."
        )
    return ratio


def train(args: Namespace, data_config: dict):
    """
    Trains a model as configured in `args` (see `configs/train/v0.yaml`).

    Without `args.overfit_one_batch`, trains on the train split, computes the val loss after each epoch, and writes to
    `args.output_dir`: `best/` (lowest val loss), `last/` and `training_history.json`. Checkpoints contain the model and
    the feature extractor used for training.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and args.bf16
    logger.info(f"Device {device}, bf16 {use_bf16}")

    store_name = data_config["store_name"]
    train_data = WeatherData(args.data_path, split="train", store_name=store_name)
    train_pairs = ForecastPairs(
        train_data, lead_steps=1, time_features=args.use_time_features, num_input_steps=args.num_input_steps
    )
    static = torch.from_numpy(train_data.static).to(device)

    loss_settings = get_loss_settings(args, data_config, train_data.feature_extractor)
    log_loss_settings(train_data.variables, loss_settings)
    config = NanoWeatherConfig(
        variables=train_data.variables,
        num_input_steps=args.num_input_steps,
        num_static_channels=static.shape[0],
        num_time_features=NUM_TIME_FEATURES if args.use_time_features else 0,
        grid_size=train_data.states.shape[-2:],
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        **loss_settings,
    )
    model = NanoWeatherForWeatherForecasting(config).to(device)
    logger.info(f"Model with {model.num_parameters() / 1e6:.2f}M parameters")
    if args.compile:
        model = torch.compile(model)

    if args.overfit_one_batch:
        overfit_one_batch(model, train_pairs, static, device, args, use_bf16)
        return

    val_data = WeatherData(
        args.data_path, split="val", store_name=store_name, feature_extractor=train_data.feature_extractor
    )
    val_pairs = ForecastPairs(
        val_data, lead_steps=1, time_features=args.use_time_features, num_input_steps=args.num_input_steps
    )
    pin_memory = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_pairs, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=args.eval_batch_size, pin_memory=pin_memory)

    num_training_steps = args.num_train_epochs * len(train_loader)
    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    logger.info(
        f"{len(train_pairs)} train pairs, {len(val_pairs)} val pairs, {len(train_loader)} steps per epoch, "
        f"{num_training_steps} steps in total"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_dir = os.path.join(args.output_dir, "best")
    history = {"args": vars(args), "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    global_step = 0
    model.train()

    for epoch in range(1, args.num_train_epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        for batch in train_loader:
            loss = training_step(
                model, to_device(batch, device), static, optimizer, scheduler, args.max_grad_norm, use_bf16
            )
            running_loss += loss
            global_step += 1
            if global_step % args.logging_steps == 0:
                mean_loss = running_loss / args.logging_steps
                history["train_loss"].append({"step": global_step, "loss": mean_loss})
                logger.info(
                    f"epoch {epoch:3d} | step {global_step:7d} | loss {mean_loss:.5f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0

        val_loss = evaluate_loss(model, val_loader, static, device, use_bf16)
        history["val_loss"].append({"epoch": epoch, "step": global_step, "loss": val_loss})
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, train_data.feature_extractor, best_dir)
        logger.info(
            f"epoch {epoch:3d} done in {time.time() - epoch_start:.0f}s | val loss {val_loss:.5f}"
            + (" | new best, saved" if is_best else "")
        )
        with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as writer:
            json.dump(history, writer, indent=2)

    save_checkpoint(model, train_data.feature_extractor, os.path.join(args.output_dir, "last"))
    logger.info(f"Best val loss {best_val_loss:.5f}, checkpoint in {best_dir}")

    if args.push_to_hub:
        best_model = NanoWeatherForWeatherForecasting.from_pretrained(best_dir)
        best_model.push_to_hub(args.push_to_hub)
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=os.path.join(best_dir, "preprocessor_config.json"),
            path_in_repo="preprocessor_config.json",
            repo_id=args.push_to_hub,
        )
        logger.info(f"Pushed to https://huggingface.co/{args.push_to_hub}")
