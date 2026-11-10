"""nanoWeather model configuration."""

import copy
import json
import logging
import os


logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"


class NanoWeatherConfig:
    r"""
    Configuration of a `NanoWeatherForWeatherForecasting` model: a small Vision Transformer on a global
    latitude/longitude grid that predicts the state of the atmosphere one time step ahead.

    Args:
        variables (`list[str]`, *optional*, defaults to `["z500", "t850", "t2m", "u10", "v10", "msl"]`):
            Prognostic variables, in channel order. Must match the feature extractor.
        num_static_channels (`int`, *optional*, defaults to 2):
            Number of static fields concatenated to the input (land-sea mask and orography).
        num_time_features (`int`, *optional*, defaults to 0):
            Number of time feature channels concatenated to the input (sunlight and time of year, see
            `feature_extraction_nanoweather.get_time_features`). 0 disables them, 4 enables them.
        grid_size (`tuple[int, int]`, *optional*, defaults to `(32, 64)`):
            Number of grid points in latitude and longitude.
        patch_size (`int`, *optional*, defaults to 2):
            Size of the square patches. Must divide both dimensions of `grid_size`.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the transformer layers.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in each layer.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the MLP in each layer.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout probability after the attention output and in the MLP.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout probability on the attention weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon of the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        predict_residual (`bool`, *optional*, defaults to `True`):
            Whether the model predicts the change over one time step, added to the input state, rather than the next
            state directly. The output head then starts at zero, so the untrained model is the persistence forecast.
        residual_scale (`list[float]`, *optional*):
            Per variable, the standard deviation of the change over one time step divided by the standard deviation
            of the state (`diff_std / std`). When set, the output of the head is multiplied by it before being added to
            the input state, so the head predicts changes of unit variance for every variable, and the loss is
            computed on changes in these units. Requires `predict_residual`. `None` disables it.
        loss_weights (`list[float]`, *optional*):
            Weight of each variable in the loss, normalized to mean 1. `None` weights all variables equally.
        attn_implementation (`str`, *optional*, defaults to `"sdpa"`):
            Attention backend, `"sdpa"` (`torch.nn.functional.scaled_dot_product_attention`) or `"eager"`.

    Example:

    ```python
    >>> from nanoweather.models.configuration_nanoweather import NanoWeatherConfig
    >>> from nanoweather.models.modeling_nanoweather import NanoWeatherForWeatherForecasting

    >>> configuration = NanoWeatherConfig(num_hidden_layers=4)
    >>> model = NanoWeatherForWeatherForecasting(configuration)
    ```
    """

    model_type = "nanoweather"

    def __init__(
        self,
        variables: list[str] | None = None,
        num_static_channels: int = 2,
        num_time_features: int = 0,
        grid_size: tuple[int, int] | list[int] = (32, 64),
        patch_size: int = 2,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        predict_residual: bool = True,
        residual_scale: list[float] | None = None,
        loss_weights: list[float] | None = None,
        attn_implementation: str = "sdpa",
        **kwargs,
    ):
        self.variables = list(variables) if variables is not None else ["z500", "t850", "t2m", "u10", "v10", "msl"]
        self.num_static_channels = num_static_channels
        self.num_time_features = num_time_features
        self.grid_size = tuple(grid_size)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.predict_residual = predict_residual
        self.residual_scale = list(residual_scale) if residual_scale is not None else None
        self.loss_weights = list(loss_weights) if loss_weights is not None else None
        self.attn_implementation = attn_implementation

        if any(size % patch_size != 0 for size in self.grid_size):
            raise ValueError(
                f"`patch_size` ({patch_size}) must divide both dimensions of `grid_size` {self.grid_size}."
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_attention_heads` ({num_attention_heads})."
            )
        for name, values in (("residual_scale", self.residual_scale), ("loss_weights", self.loss_weights)):
            if values is not None and len(values) != len(self.variables):
                raise ValueError(
                    f"`{name}` must have one value per variable ({len(self.variables)}), got {len(values)}."
                )
        if self.residual_scale is not None and not predict_residual:
            raise ValueError("`residual_scale` requires `predict_residual=True`.")
        if attn_implementation not in ("sdpa", "eager"):
            raise ValueError(f"`attn_implementation` must be `'sdpa'` or `'eager'`, got {attn_implementation!r}.")
        for key in kwargs:
            if key != "model_type":
                logger.warning(f"Ignoring unknown config attribute `{key}`.")

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @property
    def num_patches(self) -> int:
        return (self.grid_size[0] // self.patch_size) * (self.grid_size[1] // self.patch_size)

    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        output["grid_size"] = list(self.grid_size)
        output["model_type"] = self.model_type
        return output

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory: str | os.PathLike):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, CONFIG_NAME), "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NanoWeatherConfig":
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> "NanoWeatherConfig":
        """
        Loads a configuration from a local directory or a Hub model repository.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Local directory containing `config.json`, or a Hub model repository id.
            kwargs (*optional*):
                Passed to `huggingface_hub.hf_hub_download` (e.g. `revision`).
        """
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(str(pretrained_model_name_or_path), CONFIG_NAME, **kwargs)
        with open(config_file, encoding="utf-8") as reader:
            return cls.from_dict(json.load(reader))

    def __eq__(self, other) -> bool:
        return isinstance(other, NanoWeatherConfig) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"
