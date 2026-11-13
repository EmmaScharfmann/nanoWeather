"""Training loop: one-step training from scratch, or rollout fine-tuning of a trained model."""

import json
import logging
import math
import os
import random
import time
from argparse import Namespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ..data.datasets import ForecastPairs, WeatherData
from ..models.auto import load_model, model_class_for
from ..models.configuration_nanoweather import NanoWeatherConfig
from ..models.configuration_nanoweather_mesh import NanoWeatherMeshConfig
from ..processing.feature_extraction_nanoweather import NUM_TIME_FEATURES
from ..processing.prepare import expand_variables
from .loss_weights import change_std_by_latitude, get_pressure_loss_weights


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


def rollout_loss(model, batch: dict[str, torch.Tensor], static: torch.Tensor, gradient_checkpointing: bool = False):
    """
    Mean loss over consecutive forecast steps, each step taking the previous prediction as input.

    Gradients flow through the whole rollout, so the loss at later steps also trains the earlier ones. With one target
    step (`labels` of shape `(batch_size, num_variables, latitude, longitude)`), this is the one-step loss.

    Args:
        model (`NanoWeatherForecastingModel`):
            Model to train.
        batch (`dict[str, torch.Tensor]`):
            Sample from `ForecastPairs`, with `labels` of shape `(batch_size, num_rollout_steps, ...)` for a rollout.
        static (`torch.Tensor`):
            Normalized static fields.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Recompute each step's activations during the backward pass instead of storing them, so memory does not
            grow with the number of steps, at the cost of about one more forward pass.
    """
    labels = batch["labels"]
    time_features = batch.get("time_features")
    if labels.dim() == 4:
        labels = labels[:, None]
        time_features = time_features[:, None] if time_features is not None else None
    states, previous_states = batch["states"], batch.get("previous_states")

    def step(states, previous_states, step_time_features, step_labels):
        output = model(
            states,
            static=static,
            time_features=step_time_features,
            previous_states=previous_states,
            labels=step_labels,
        )
        return output.prediction, output.loss

    losses = []
    for index in range(labels.shape[1]):
        step_time_features = time_features[:, index] if time_features is not None else None
        inputs = (states, previous_states, step_time_features, labels[:, index])
        if gradient_checkpointing and torch.is_grad_enabled():
            prediction, loss = checkpoint(step, *inputs, use_reentrant=False)
        else:
            prediction, loss = step(*inputs)
        losses.append(loss)
        if previous_states is not None:
            previous_states = torch.cat([previous_states[:, 1:], states[:, None]], dim=1)
        states = prediction
    return torch.stack(losses).mean()


def training_step(
    model,
    batch,
    static,
    optimizer,
    scheduler,
    max_grad_norm: float,
    use_bf16: bool,
    gradient_checkpointing: bool = False,
) -> float:
    with torch.autocast(device_type=static.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        loss = rollout_loss(model, batch, static, gradient_checkpointing)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


@torch.no_grad()
def evaluate_loss(model, dataloader, static, device, use_bf16: bool) -> float:
    """Mean rollout loss over a dataloader, weighted by batch size."""
    model.eval()
    total_loss, num_samples = 0.0, 0
    for batch in dataloader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = rollout_loss(model, batch, static)
        batch_size = batch["states"].shape[0]
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    model.train()
    return total_loss / num_samples


def get_loss_settings(args: Namespace, data_config: dict, train_data: WeatherData) -> dict:
    """`residual_scale`, `loss_weights` and `latitude_weight_exponent` for the model config, from the arguments."""
    feature_extractor = train_data.feature_extractor
    variables = feature_extractor.variables
    residual_scale = None
    if args.tendency_normalization:
        state_std = np.array(feature_extractor.std)
        if getattr(args, "tendency_normalization_by_latitude", False):
            residual_scale = change_std_by_latitude(train_data.normalized_states).tolist()
        else:
            residual_scale = (np.array(feature_extractor.diff_std) / state_std).tolist()
    if args.loss_weighting == "uniform":
        loss_weights = None
    elif args.loss_weighting == "pressure":
        specs = expand_variables(data_config)
        loss_weights = get_pressure_loss_weights(variables, specs, args.surface_loss_weights or {})
    else:
        raise ValueError(f"Unknown loss_weighting {args.loss_weighting!r}, expected 'uniform' or 'pressure'.")
    return {
        "residual_scale": residual_scale,
        "loss_weights": loss_weights,
        "latitude_weight_exponent": getattr(args, "latitude_weight_exponent", 1.0),
    }


def log_loss_settings(variables: list[str], settings: dict):
    if settings["loss_weights"] is not None:
        logger.info("Loss weights: " + ", ".join(f"{n} {w:.3f}" for n, w in zip(variables, settings["loss_weights"])))
    if settings["residual_scale"] is not None:
        scale = np.array(settings["residual_scale"])
        if scale.ndim == 2:
            polar_ratio = scale[:, [0, -1]].mean(axis=1) / scale.mean(axis=1)
            logger.info(
                "Residual scale per latitude; polar rows / mean: "
                + ", ".join(f"{n} {r:.2f}" for n, r in zip(variables, polar_ratio))
            )
            scale = scale.mean(axis=1)
        logger.info("Residual scale (diff_std / std): " + ", ".join(f"{n} {r:.3f}" for n, r in zip(variables, scale)))
    logger.info(f"Latitude weight exponent {settings['latitude_weight_exponent']}")


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
        loss = training_step(
            model, batch, static, optimizer, None, args.max_grad_norm, use_bf16, args.gradient_checkpointing
        )
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


def build_model(args: Namespace, data_config: dict, train_data: WeatherData, num_static_channels: int):
    """
    A new model configured from `args`, or with `args.init_from` the trained model to fine-tune.

    `args.architecture` is `"vit"` (patches of the latitude/longitude grid) or `"mesh"` (encode-process-decode on an
    icosahedral mesh). When fine-tuning, the architecture, inputs and loss settings come from the checkpoint, and the
    corresponding arguments are ignored.
    """
    if args.init_from:
        model = load_model(args.init_from)
        if model.config.variables != train_data.variables:
            raise ValueError(f"The model in {args.init_from} was trained on other variables than this dataset.")
        logger.info(f"Fine-tuning {args.init_from}; model and loss settings come from its config")
        return model.train()

    missing = [
        name for name in ("hidden_size", "num_hidden_layers", "architecture") if getattr(args, name, None) is None
    ]
    if missing:
        raise ValueError(f"Set {', '.join(missing)} to train a new model, or init_from to fine-tune a trained one.")
    loss_settings = get_loss_settings(args, data_config, train_data)
    log_loss_settings(train_data.variables, loss_settings)
    common = {
        "variables": train_data.variables,
        "num_input_steps": args.num_input_steps,
        "num_static_channels": num_static_channels,
        "num_time_features": NUM_TIME_FEATURES if args.use_time_features else 0,
        "grid_size": train_data.normalized_states.shape[-2:],
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size,
        **loss_settings,
    }
    if args.architecture == "vit":
        config = NanoWeatherConfig(patch_size=args.patch_size, **common)
    elif args.architecture == "mesh":
        config = NanoWeatherMeshConfig(
            mesh_refinement=args.mesh_refinement,
            encoder_num_neighbors=args.encoder_num_neighbors,
            processor_num_neighbors=args.processor_num_neighbors,
            decoder_num_neighbors=args.decoder_num_neighbors,
            dense_attention_max_pairs=args.dense_attention_max_pairs,
            **common,
        )
    else:
        raise ValueError(f"Unknown architecture {args.architecture!r}, expected 'vit' or 'mesh'.")
    return model_class_for(config.model_type)(config)


def make_dataset(data: WeatherData, config: NanoWeatherConfig, num_rollout_steps: int) -> ForecastPairs:
    return ForecastPairs(
        data,
        num_rollout_steps=num_rollout_steps,
        time_features=config.num_time_features > 0,
        num_input_steps=config.num_input_steps,
    )


def train(args: Namespace, data_config: dict):
    """
    Trains a model as configured in `args` (see `configs/train/pretrain.yaml` and `configs/train/rollout.yaml`).

    Training runs in stages, one per entry of `args.rollout_steps`, each for `args.num_train_epochs` epochs with the
    loss averaged over that many consecutive forecast steps. `[1]` is one-step training. For rollout fine-tuning, start
    from a trained model with `args.init_from` and increase the number of steps, e.g. `[2, 4, 8]`.

    The val loss is computed after each epoch over the largest number of steps of all stages, so it is comparable
    across stages. Writes to `args.output_dir`: `best/` (lowest val loss), `last/` and `training_history.json`.
    Checkpoints contain the model and the feature extractor used for training.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and args.bf16
    logger.info(f"Device {device}, bf16 {use_bf16}")

    store_name = data_config["store_name"]
    train_data = WeatherData(args.data_path, split="train", store_name=store_name, keep_physical_states=False)
    static = torch.from_numpy(train_data.static).to(device)
    model = build_model(args, data_config, train_data, static.shape[0]).to(device)
    config = model.config
    logger.info(f"Model with {model.num_parameters() / 1e6:.2f}M parameters")
    if args.compile:
        model = torch.compile(model)

    stages = list(args.rollout_steps)
    if args.overfit_one_batch:
        overfit_one_batch(model, make_dataset(train_data, config, stages[0]), static, device, args, use_bf16)
        return

    val_data = WeatherData(
        args.data_path,
        split="val",
        store_name=store_name,
        feature_extractor=train_data.feature_extractor,
        keep_physical_states=False,
    )
    pin_memory = device.type == "cuda"
    val_steps = max(stages)
    val_loader = torch.utils.data.DataLoader(
        make_dataset(val_data, config, val_steps), batch_size=args.eval_batch_size, pin_memory=pin_memory
    )

    steps_per_epoch = {
        num_steps: len(ForecastPairs(train_data, num_steps, num_input_steps=config.num_input_steps)) // args.batch_size
        for num_steps in stages
    }
    num_training_steps = args.num_train_epochs * sum(steps_per_epoch.values())
    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    logger.info(
        f"Stages (forecast steps per sample): {stages}, {args.num_train_epochs} epoch(s) each, "
        f"{num_training_steps} optimizer steps in total; val loss over {val_steps} steps"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_dir = os.path.join(args.output_dir, "best")
    history = {"args": vars(args), "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    global_step, epoch = 0, 0
    model.train()

    for num_steps in stages:
        train_loader = torch.utils.data.DataLoader(
            make_dataset(train_data, config, num_steps),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
        )
        for _ in range(args.num_train_epochs):
            epoch += 1
            epoch_start = time.time()
            running_loss = 0.0
            for batch in train_loader:
                loss = training_step(
                    model,
                    to_device(batch, device),
                    static,
                    optimizer,
                    scheduler,
                    args.max_grad_norm,
                    use_bf16,
                    args.gradient_checkpointing,
                )
                running_loss += loss
                global_step += 1
                if global_step % args.logging_steps == 0:
                    mean_loss = running_loss / args.logging_steps
                    history["train_loss"].append({"step": global_step, "rollout_steps": num_steps, "loss": mean_loss})
                    logger.info(
                        f"epoch {epoch:3d} | {num_steps} steps | step {global_step:7d} | loss {mean_loss:.5f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0

            val_loss = evaluate_loss(model, val_loader, static, device, use_bf16)
            history["val_loss"].append(
                {"epoch": epoch, "step": global_step, "rollout_steps": num_steps, "loss": val_loss}
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(model, train_data.feature_extractor, best_dir)
            logger.info(
                f"epoch {epoch:3d} ({num_steps} steps) done in {time.time() - epoch_start:.0f}s | "
                f"val loss over {val_steps} steps {val_loss:.5f}" + (" | new best, saved" if is_best else "")
            )
            with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as writer:
                json.dump(history, writer, indent=2)

    save_checkpoint(model, train_data.feature_extractor, os.path.join(args.output_dir, "last"))
    logger.info(f"Best val loss {best_val_loss:.5f}, checkpoint in {best_dir}")

    if args.push_to_hub:
        best_model = load_model(best_dir)
        best_model.push_to_hub(args.push_to_hub)
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=os.path.join(best_dir, "preprocessor_config.json"),
            path_in_repo="preprocessor_config.json",
            repo_id=args.push_to_hub,
        )
        logger.info(f"Pushed to https://huggingface.co/{args.push_to_hub}")
