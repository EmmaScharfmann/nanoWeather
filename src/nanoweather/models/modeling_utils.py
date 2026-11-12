"""Shared model code: outputs, loss, saving and loading, and the forecasting logic common to all architectures.

An architecture subclasses `NanoWeatherForecastingModel` and implements `predict_change`, which maps the input fields
on the grid to the predicted change (or next state) on the grid. Input assembly, the residual connection, the loss and
the autoregressive rollout are shared.
"""

import logging
import os
import tempfile
from dataclasses import dataclass

import torch
from torch import nn

from .configuration_utils import NanoWeatherBaseConfig


logger = logging.getLogger(__name__)

WEIGHTS_NAME = "model.safetensors"


@dataclass
class NanoWeatherForecastOutput:
    """
    Output of `NanoWeatherForWeatherForecasting`.

    Args:
        loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is provided):
            Latitude-weighted mean squared error between `prediction` and `labels`, weighted per variable by
            `config.loss_weights`, in units of `config.residual_scale` if set, otherwise in normalized units.
        prediction (`torch.FloatTensor` of shape `(batch_size, num_variables, latitude, longitude)`):
            Predicted state one time step ahead, normalized.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Hidden states of the backbone.
    """

    loss: torch.FloatTensor | None = None
    prediction: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


def get_latitude_weights(num_latitudes: int) -> torch.Tensor:
    """`cos(latitude)` on an equiangular grid without poles (as in WeatherBench 2), normalized to mean 1."""
    latitude = -90 + (torch.arange(num_latitudes, dtype=torch.float64) + 0.5) * 180 / num_latitudes
    weights = torch.cos(torch.deg2rad(latitude))
    return (weights / weights.mean()).float()


def latitude_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
    variable_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean squared error with each latitude row weighted by its area, and each variable by `variable_weights`.

    Args:
        prediction (`torch.Tensor` of shape `(batch_size, num_variables, latitude, longitude)`):
            Prediction.
        target (`torch.Tensor` of shape `(batch_size, num_variables, latitude, longitude)`):
            Target.
        latitude_weights (`torch.Tensor` of shape `(latitude,)`):
            Area weights with mean 1.
        variable_weights (`torch.Tensor` of shape `(num_variables,)`, *optional*):
            Weights with mean 1. All variables count equally if not given.
    """
    squared_error = (prediction - target) ** 2 * latitude_weights[:, None]
    per_variable = squared_error.mean(dim=(0, 2, 3))
    if variable_weights is not None:
        per_variable = per_variable * variable_weights
    return per_variable.mean()


class NanoWeatherPreTrainedModel(nn.Module):
    """
    Base class for weight initialization, saving and loading.

    Saved checkpoints contain `config.json` and `model.safetensors`.
    """

    config_class = None

    def __init__(self, config: NanoWeatherBaseConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(getattr(module, "position_embeddings", None), nn.Parameter):
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)

    def post_init(self):
        """Initializes the weights. Called at the end of `__init__` of the models."""
        self.apply(self._init_weights)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_pretrained(self, save_directory: str | os.PathLike):
        """Saves the configuration and the weights to `save_directory`."""
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = {name: tensor.contiguous() for name, tensor in self.state_dict().items()}
        save_file(state_dict, os.path.join(save_directory, WEIGHTS_NAME), metadata={"format": "pt"})
        logger.info(f"Model saved in {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        """
        Loads a model from a local directory or a Hub model repository, in eval mode.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Local directory written by `save_pretrained`, or a Hub model repository id.
            kwargs (*optional*):
                Passed to `huggingface_hub.hf_hub_download` (e.g. `revision`).
        """
        from safetensors.torch import load_file

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if os.path.isdir(pretrained_model_name_or_path):
            weights_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            from huggingface_hub import hf_hub_download

            weights_file = hf_hub_download(str(pretrained_model_name_or_path), WEIGHTS_NAME, **kwargs)
        model = cls(config)
        model.load_state_dict(load_file(weights_file), strict=True)
        return model.eval()

    def push_to_hub(self, repo_id: str, private: bool | None = None, commit_message: str = "Upload model"):
        """Saves the model to a temporary directory and uploads it to the Hub model repository `repo_id`."""
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as save_directory:
            self.save_pretrained(save_directory)
            api.upload_folder(folder_path=save_directory, repo_id=repo_id, commit_message=commit_message)


class NanoWeatherForecastingModel(NanoWeatherPreTrainedModel):
    """
    Forecasting logic shared by all architectures: assembles the inputs, adds the predicted change to the input state,
    computes the loss, and rolls out several steps.

    Subclasses build their modules, call `post_init()`, then `init_output_head(head)`, and implement `predict_change`.
    """

    def __init__(self, config: NanoWeatherBaseConfig):
        super().__init__(config)
        self.register_buffer("latitude_weights", get_latitude_weights(config.grid_size[0]), persistent=False)
        self.register_buffer("residual_scale", self._per_variable_buffer(config.residual_scale), persistent=False)
        self.register_buffer("variable_weights", self._per_variable_buffer(config.loss_weights), persistent=False)

    @staticmethod
    def _per_variable_buffer(values: list[float] | None) -> torch.Tensor | None:
        return torch.tensor(values, dtype=torch.float32) if values is not None else None

    def init_output_head(self, head: nn.Linear):
        """With `config.predict_residual`, zeroes the output head so the untrained model predicts no change."""
        if self.config.predict_residual:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def predict_change(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_input_channels, latitude, longitude)`):
                Input states, static fields and time features.

        Returns:
            The predicted change of shape `(batch_size, num_variables, latitude, longitude)` (the next state if
            `config.predict_residual` is `False`), and the last hidden state of the architecture.
        """
        raise NotImplementedError

    def forward(
        self,
        states: torch.Tensor,
        static: torch.Tensor | None = None,
        time_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        previous_states: torch.Tensor | None = None,
    ) -> NanoWeatherForecastOutput:
        r"""
        Args:
            states (`torch.FloatTensor` of shape `(batch_size, num_variables, latitude, longitude)`):
                Normalized state at time t.
            static (`torch.FloatTensor` of shape `(num_static_channels, latitude, longitude)` or with a batch dimension, *optional*):
                Normalized static fields. Required when `config.num_static_channels > 0`.
            time_features (`torch.FloatTensor` of shape `(batch_size, num_time_features, latitude, longitude)`, *optional*):
                Time features at the input time. Required when `config.num_time_features > 0`.
            labels (`torch.FloatTensor` of shape `(batch_size, num_variables, latitude, longitude)`, *optional*):
                Normalized state one time step after `states`. If given, the loss is returned (see `latitude_weighted_mse`,
                `config.residual_scale` and `config.loss_weights`).
            previous_states (`torch.FloatTensor` of shape `(batch_size, num_input_steps - 1, num_variables, latitude, longitude)`, *optional*):
                Normalized states before `states`, oldest first, one time step apart. Required when
                `config.num_input_steps > 1`.
        """
        inputs = [states]
        if self.config.num_input_steps > 1:
            expected = self.config.num_input_steps - 1
            if previous_states is None or previous_states.shape[1] != expected:
                raise ValueError(f"`previous_states` with {expected} time steps is required.")
            inputs.insert(0, previous_states.flatten(1, 2).to(states.dtype))
        if self.config.num_static_channels > 0:
            if static is None:
                raise ValueError(f"`static` is required: the model expects {self.config.num_static_channels} fields.")
            if static.dim() == 3:
                static = static.unsqueeze(0).expand(states.shape[0], -1, -1, -1)
            inputs.append(static.to(states.dtype))
        if self.config.num_time_features > 0:
            if time_features is None:
                raise ValueError(
                    f"`time_features` is required: the model expects {self.config.num_time_features} channels."
                )
            inputs.append(time_features.to(states.dtype))
        pixel_values = torch.cat(inputs, dim=1)

        prediction, last_hidden_state = self.predict_change(pixel_values)
        if self.residual_scale is not None:
            prediction = prediction * self.residual_scale[:, None, None].to(prediction.dtype)
        if self.config.predict_residual:
            prediction = states + prediction

        loss = None
        if labels is not None:
            error_scale = self.residual_scale[:, None, None] if self.residual_scale is not None else 1.0
            loss = latitude_weighted_mse(
                prediction.float() / error_scale,
                labels.float() / error_scale,
                self.latitude_weights,
                self.variable_weights,
            )

        return NanoWeatherForecastOutput(loss=loss, prediction=prediction, last_hidden_state=last_hidden_state)

    @torch.no_grad()
    def rollout(
        self,
        states: torch.Tensor,
        static: torch.Tensor | None = None,
        num_steps: int = 1,
        time_features: torch.Tensor | None = None,
        previous_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forecasts `num_steps` time steps autoregressively, feeding each prediction back as the next input.

        Args:
            states (`torch.FloatTensor` of shape `(batch_size, num_variables, latitude, longitude)`):
                Normalized initial state.
            static (`torch.FloatTensor`, *optional*):
                Normalized static fields, as in `forward`.
            num_steps (`int`, *optional*, defaults to 1):
                Number of time steps.
            time_features (`torch.FloatTensor` of shape `(batch_size, num_steps, num_time_features, latitude, longitude)`, *optional*):
                Time features at the input time of each step. Required when `config.num_time_features > 0`.
            previous_states (`torch.FloatTensor` of shape `(batch_size, num_input_steps - 1, num_variables, latitude, longitude)`, *optional*):
                Normalized states before `states`, oldest first. Required when `config.num_input_steps > 1`. At each
                step, the oldest is dropped and the current state appended.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_steps, num_variables, latitude, longitude)`, normalized.
        """
        predictions = []
        current_states = states
        for step in range(num_steps):
            step_time_features = time_features[:, step] if time_features is not None else None
            prediction = self(
                current_states, static=static, time_features=step_time_features, previous_states=previous_states
            ).prediction
            if previous_states is not None:
                previous_states = torch.cat([previous_states[:, 1:], current_states[:, None]], dim=1)
            current_states = prediction
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)
