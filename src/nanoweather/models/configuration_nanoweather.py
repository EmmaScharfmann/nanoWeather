"""Configuration of the nanoWeather Vision Transformer."""

from .configuration_utils import COMMON_ARGS_DOC, NanoWeatherBaseConfig


class NanoWeatherConfig(NanoWeatherBaseConfig):
    __doc__ = (
        r"""
    Configuration of a `NanoWeatherForWeatherForecasting` model: a small Vision Transformer on a global
    latitude/longitude grid that predicts the state of the atmosphere one time step ahead.

    Args:"""
        + COMMON_ARGS_DOC
        + r"""
        patch_size (`int`, *optional*, defaults to 2):
            Size of the square patches. Must divide both dimensions of `grid_size`.
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
    )

    model_type = "nanoweather"
    attn_implementations = ("sdpa", "eager")

    def __init__(self, patch_size: int = 2, **kwargs):
        self.patch_size = patch_size
        super().__init__(**kwargs)
        if any(size % patch_size != 0 for size in self.grid_size):
            raise ValueError(
                f"`patch_size` ({patch_size}) must divide both dimensions of `grid_size` {self.grid_size}."
            )

    @property
    def num_patches(self) -> int:
        return (self.grid_size[0] // self.patch_size) * (self.grid_size[1] // self.patch_size)
