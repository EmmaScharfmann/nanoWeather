"""A small Vision Transformer that forecasts the global atmospheric state one time step ahead.

    previous states (B, K - 1, C, H, W, optional) + states (B, C, H, W) + static (S, H, W) + time features (B, F, H, W, optional)
      -> patch embeddings + position embeddings      (B, N, D)   N = (H / p) * (W / p)
      -> transformer layers                          (B, N, D)
      -> linear head + unpatchify                    (B, C, H, W)
      -> + states (if `predict_residual`)            next state, normalized

No padding is needed for the periodic longitude: patches do not overlap and attention is global, so patches on
either side of the 0/360 meridian attend to each other like any other pair.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .configuration_nanoweather import NanoWeatherConfig
from .modeling_utils import (  # noqa: F401  (latitude_weighted_mse and get_latitude_weights are re-exported)
    NanoWeatherForecastingModel,
    NanoWeatherForecastOutput,
    NanoWeatherPreTrainedModel,
    get_latitude_weights,
    latitude_weighted_mse,
)


@dataclass
class NanoWeatherModelOutput:
    """
    Output of `NanoWeatherModel`.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Hidden states after the last transformer layer and the final layer norm.
    """

    last_hidden_state: torch.FloatTensor | None = None


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


ALL_ATTENTION_FUNCTIONS: dict[str, Callable] = {
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class NanoWeatherEmbeddings(nn.Module):
    """Embeds the input grid as non-overlapping patches, plus learned position embeddings."""

    def __init__(self, config: NanoWeatherConfig):
        super().__init__()
        num_channels = config.num_input_channels
        self.patch_embeddings = nn.Conv2d(
            num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.num_patches, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
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


class NanoWeatherForWeatherForecasting(NanoWeatherForecastingModel):
    """
    Vision Transformer with a linear head that predicts the normalized state one time step ahead.

    With `config.predict_residual`, the head is initialized at zero, so the untrained model predicts no change.
    """

    config_class = NanoWeatherConfig

    def __init__(self, config: NanoWeatherConfig):
        super().__init__(config)
        self.model = NanoWeatherModel(config)
        self.head = nn.Linear(config.hidden_size, config.num_variables * config.patch_size**2)
        self.post_init()
        self.init_output_head(self.head)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(batch_size, num_patches, num_variables * patch_size**2) -> (batch_size, num_variables, latitude, longitude)"""
        batch_size = patches.shape[0]
        patch_size = self.config.patch_size
        num_rows = self.config.grid_size[0] // patch_size
        num_columns = self.config.grid_size[1] // patch_size
        patches = patches.reshape(batch_size, num_rows, num_columns, self.config.num_variables, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(batch_size, self.config.num_variables, *self.config.grid_size)

    def predict_change(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_hidden_state = self.model(pixel_values).last_hidden_state
        return self.unpatchify(self.head(last_hidden_state)), last_hidden_state
