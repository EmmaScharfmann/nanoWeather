# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch nanoWeather model: a small Vision Transformer that forecasts the global atmospheric state 6h ahead.

The whole forward pass is in this file, following the "one model, one file" convention of transformers.

    states (B, C, H, W) + static (S, H, W) + time features (B, F, H, W, optional)
      -> patch embeddings + position embeddings      (B, N, D)   N = (H / p) * (W / p)
      -> transformer layers                          (B, N, D)
      -> linear head + unpatchify                    (B, C, H, W)
      -> + states (if `predict_residual`)            next state, normalized

No padding is needed for the periodic longitude: patches do not overlap and attention is global, so patches on
either side of the 0/360 meridian attend to each other like any other pair.
"""

import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from configuration_nanoweather import NanoWeatherConfig


logger = logging.getLogger(__name__)

WEIGHTS_NAME = "model.safetensors"


@dataclass
class NanoWeatherModelOutput:
    """
    Output of [`NanoWeatherModel`]. Mirrors `transformers.modeling_outputs.BaseModelOutput`.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Hidden states after the last transformer layer and the final layer norm.
    """

    last_hidden_state: torch.FloatTensor | None = None


@dataclass
class NanoWeatherForecastOutput:
    """
    Output of [`NanoWeatherForWeatherForecasting`].

    Args:
        loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is provided):
            Latitude-weighted mean squared error between `prediction` and `labels`, in normalized units.
        prediction (`torch.FloatTensor` of shape `(batch_size, num_variables, latitude, longitude)`):
            Predicted state 6h ahead, normalized.
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


def latitude_weighted_mse(prediction: torch.Tensor, target: torch.Tensor, latitude_weights: torch.Tensor):
    """Mean squared error with each latitude row weighted by its area. Weights have mean 1."""
    return ((prediction - target) ** 2 * latitude_weights[:, None]).mean()


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, None]:
    attn_output = nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout if module.training else 0.0, scale=scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# Same pattern as transformers' attention dispatch: layers look the backend up by `config.attn_implementation`.
ALL_ATTENTION_FUNCTIONS: dict[str, Callable] = {
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class NanoWeatherEmbeddings(nn.Module):
    """Embeds the input grid as non-overlapping patches, plus learned position embeddings."""

    def __init__(self, config: NanoWeatherConfig):
        super().__init__()
        num_channels = config.num_variables + config.num_static_channels + config.num_time_features
        self.patch_embeddings = nn.Conv2d(
            num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.num_patches, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, latitude, longitude) -> (batch_size, num_patches, hidden_size), row-major patches
        embeddings = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        return embeddings + self.position_embeddings


class NanoWeatherAttention(nn.Module):
    def __init__(self, config: NanoWeatherConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_probs_dropout_prob
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config.attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            scaling=self.scaling,
            dropout=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class NanoWeatherMLP(nn.Module):
    def __init__(self, config: NanoWeatherConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        return self.dropout(self.fc2(hidden_states))


class NanoWeatherLayer(nn.Module):
    """Pre-norm transformer layer: attention and MLP, each with a residual connection."""

    def __init__(self, config: NanoWeatherConfig):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = NanoWeatherAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NanoWeatherMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(self.layernorm_before(hidden_states))
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = hidden_states + self.mlp(self.layernorm_after(hidden_states))
        return hidden_states


class NanoWeatherPreTrainedModel(nn.Module):
    """
    Base class handling weight initialization, saving and loading, like `transformers.PreTrainedModel`.

    Saved checkpoints contain `config.json` and `model.safetensors`.
    """

    config_class = NanoWeatherConfig
    base_model_prefix = "model"

    def __init__(self, config: NanoWeatherConfig):
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
        elif isinstance(module, NanoWeatherEmbeddings):
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


class NanoWeatherModel(NanoWeatherPreTrainedModel):
    """Backbone: patch embeddings and transformer layers, without the forecasting head."""

    def __init__(self, config: NanoWeatherConfig):
        super().__init__(config)
        self.embeddings = NanoWeatherEmbeddings(config)
        self.layers = nn.ModuleList([NanoWeatherLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> NanoWeatherModelOutput:
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, latitude, longitude)`):
                Normalized prognostic variables concatenated with static fields and time features.
        """
        hidden_states = self.embeddings(pixel_values)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return NanoWeatherModelOutput(last_hidden_state=self.layernorm(hidden_states))


class NanoWeatherForWeatherForecasting(NanoWeatherPreTrainedModel):
    """nanoWeather backbone with a linear head that predicts the normalized state 6h ahead."""

    def __init__(self, config: NanoWeatherConfig):
        super().__init__(config)
        self.model = NanoWeatherModel(config)
        self.head = nn.Linear(config.hidden_size, config.num_variables * config.patch_size**2)
        self.register_buffer("latitude_weights", get_latitude_weights(config.grid_size[0]), persistent=False)
        self.post_init()
        if config.predict_residual:
            # Zero head: the untrained model predicts no change, i.e. starts as the persistence forecast.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(batch_size, num_patches, num_variables * patch_size**2) -> (batch_size, num_variables, latitude, longitude)"""
        batch_size = patches.shape[0]
        patch_size = self.config.patch_size
        num_rows = self.config.grid_size[0] // patch_size
        num_columns = self.config.grid_size[1] // patch_size
        patches = patches.reshape(batch_size, num_rows, num_columns, self.config.num_variables, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(batch_size, self.config.num_variables, *self.config.grid_size)

    def forward(
        self,
        states: torch.Tensor,
        static: torch.Tensor | None = None,
        time_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
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
                Normalized state at t + 6h. If given, the latitude-weighted MSE is returned as `loss`.
        """
        inputs = [states]
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

        last_hidden_state = self.model(pixel_values).last_hidden_state
        prediction = self.unpatchify(self.head(last_hidden_state))
        if self.config.predict_residual:
            prediction = states + prediction

        loss = None
        if labels is not None:
            loss = latitude_weighted_mse(prediction.float(), labels.float(), self.latitude_weights)

        return NanoWeatherForecastOutput(loss=loss, prediction=prediction, last_hidden_state=last_hidden_state)

    @torch.no_grad()
    def rollout(
        self,
        states: torch.Tensor,
        static: torch.Tensor | None = None,
        num_steps: int = 1,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forecasts `num_steps` 6h steps autoregressively, feeding each prediction back as the next input.

        Args:
            states (`torch.FloatTensor` of shape `(batch_size, num_variables, latitude, longitude)`):
                Normalized initial state.
            static (`torch.FloatTensor`, *optional*):
                Normalized static fields, as in `forward`.
            num_steps (`int`, *optional*, defaults to 1):
                Number of 6h steps.
            time_features (`torch.FloatTensor` of shape `(batch_size, num_steps, num_time_features, latitude, longitude)`, *optional*):
                Time features at the input time of each step. Required when `config.num_time_features > 0`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_steps, num_variables, latitude, longitude)`, normalized.
        """
        predictions = []
        current_states = states
        for step in range(num_steps):
            step_time_features = time_features[:, step] if time_features is not None else None
            current_states = self(current_states, static=static, time_features=step_time_features).prediction
            predictions.append(current_states)
        return torch.stack(predictions, dim=1)