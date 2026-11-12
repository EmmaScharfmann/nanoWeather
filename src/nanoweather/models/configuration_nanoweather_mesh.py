"""Configuration of the nanoWeather encode-process-decode model on an icosahedral mesh."""

from .configuration_utils import COMMON_ARGS_DOC, NanoWeatherBaseConfig


class NanoWeatherMeshConfig(NanoWeatherBaseConfig):
    __doc__ = (
        r"""
    Configuration of a `NanoWeatherMeshForWeatherForecasting` model: inputs on a latitude/longitude grid are encoded
    onto an icosahedral mesh, processed by a transformer whose attention is restricted to nearby mesh nodes, and
    decoded back to the grid. `hidden_size`, `num_hidden_layers` and the other transformer settings apply to the
    processor; the encoder and decoder are one attention layer each.

    Args:"""
        + COMMON_ARGS_DOC
        + r"""
        mesh_refinement (`int`, *optional*, defaults to 3):
            Number of times the faces of an icosahedron are split into 4. The mesh has `10 * 4**mesh_refinement + 2`
            nodes: 642 for 3 (about 900 km apart), 2562 for 4 (about 450 km), 10242 for 5 (about 220 km).
        encoder_num_neighbors (`int`, *optional*, defaults to 4):
            Number of nearest mesh nodes each grid point sends its inputs to. Each node attends to the grid points it
            receives, weighted by their area, so every grid point is used whatever its latitude.
        processor_num_neighbors (`int`, *optional*, defaults to 16):
            Number of nearest mesh nodes, itself included, each node attends to in the processor.
        decoder_num_neighbors (`int`, *optional*, defaults to 4):
            Number of nearest mesh nodes each grid point attends to in the decoder.
        attn_implementation (`str`, *optional*, defaults to `"auto"`):
            `"sparse"` computes attention over the neighbors only: gathered by index where every receiver has the same
            number of neighbors, and over an edge list otherwise. Its cost grows with the number of edges. `"dense"`
            computes attention over all pairs with a mask, using fused kernels; its cost grows with receivers ×
            senders, so it is faster for small meshes and impractical for large ones. `"auto"` uses `"dense"` for edge
            sets with at most `dense_attention_max_pairs` receiver-sender pairs, and `"sparse"` otherwise.
        dense_attention_max_pairs (`int`, *optional*, defaults to 1000000):
            Threshold of `"auto"`. The best value depends on the hardware: measure with `scripts/benchmark_models.py`.

    Example:

    ```python
    >>> from nanoweather.models.configuration_nanoweather_mesh import NanoWeatherMeshConfig
    >>> from nanoweather.models.modeling_nanoweather_mesh import NanoWeatherMeshForWeatherForecasting

    >>> configuration = NanoWeatherMeshConfig(mesh_refinement=4)
    >>> model = NanoWeatherMeshForWeatherForecasting(configuration)
    ```
    """
    )

    model_type = "nanoweather_mesh"
    attn_implementations = ("auto", "sparse", "dense")

    def __init__(
        self,
        mesh_refinement: int = 3,
        encoder_num_neighbors: int = 4,
        processor_num_neighbors: int = 16,
        decoder_num_neighbors: int = 4,
        dense_attention_max_pairs: int = 1_000_000,
        **kwargs,
    ):
        self.mesh_refinement = mesh_refinement
        self.encoder_num_neighbors = encoder_num_neighbors
        self.processor_num_neighbors = processor_num_neighbors
        self.decoder_num_neighbors = decoder_num_neighbors
        self.dense_attention_max_pairs = dense_attention_max_pairs
        super().__init__(**kwargs)
        if min(encoder_num_neighbors, processor_num_neighbors, decoder_num_neighbors) < 1:
            raise ValueError("The numbers of neighbors must be at least 1.")
        if processor_num_neighbors > self.num_mesh_nodes:
            raise ValueError(f"`processor_num_neighbors` exceeds the {self.num_mesh_nodes} mesh nodes.")

    @property
    def num_mesh_nodes(self) -> int:
        return 10 * 4**self.mesh_refinement + 2
