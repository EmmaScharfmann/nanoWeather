"""An encode-process-decode forecaster on an icosahedral mesh.

    grid inputs (B, C, H, W) -> one embedding per grid point                          (B, G, D)    G = H * W
      -> encoder: each mesh node attends to the grid points sent to it, by area       (B, N, D)    N mesh nodes
      -> processor: transformer layers, each node attending to its nearest nodes      (B, N, D)
      -> decoder: each grid point attends to its nearest mesh nodes                   (B, G, D)
      -> linear head                                                                  (B, C_out, H, W)

The mesh spreads nodes evenly over the sphere, so the processor does not see the crowding of the latitude/longitude
grid near the poles, and points on either side of a pole are neighbors. Attention only covers neighbors, so its cost
grows with the number of nodes times the number of neighbors rather than the number of nodes squared (for small meshes, a masked full attention with fused
kernels is faster, see `attn_implementation`). A learned bias from
each neighbor's relative position (east, north, distance) tells the attention where the neighbor is.

The mesh and all neighbor lists are rebuilt from the config when the model is created; they are not saved.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from . import mesh_utils
from .configuration_nanoweather_mesh import NanoWeatherMeshConfig
from .modeling_utils import NanoWeatherForecastingModel, NanoWeatherPreTrainedModel


NUM_POSITION_FEATURES = 4


class NanoWeatherMeshEdges(nn.Module):
    """
    Edges from senders to receivers, held as buffers so they move with the model across devices.

    Args:
        receivers (`np.ndarray` of shape `(num_edges,)`):
            Index of the receiver of each edge.
        senders (`np.ndarray` of shape `(num_edges,)`):
            Index of the sender of each edge.
        positions (`np.ndarray` of shape `(num_edges, 4)`):
            Position of each sender relative to its receiver, see `mesh_utils.relative_positions`.
        num_receivers (`int`):
            Number of receivers.
        num_senders (`int`):
            Number of senders.
        num_neighbors (`int`, *optional*):
            Set when every receiver has exactly this many edges and edges are grouped by receiver, in order. Attention
            can then gather neighbors by index instead of working on the edge list.
        log_weights (`np.ndarray` of shape `(num_edges,)`, *optional*):
            Added to the attention scores, e.g. the log of the area of each sender.
    """

    def __init__(
        self,
        receivers: np.ndarray,
        senders: np.ndarray,
        positions: np.ndarray,
        num_receivers: int,
        num_senders: int,
        num_neighbors: int | None = None,
        log_weights: np.ndarray | None = None,
    ):
        super().__init__()
        self.num_receivers = num_receivers
        self.num_senders = num_senders
        self.num_neighbors = num_neighbors
        self.register_buffer("receivers", torch.as_tensor(receivers, dtype=torch.long), persistent=False)
        self.register_buffer("senders", torch.as_tensor(senders, dtype=torch.long), persistent=False)
        self.register_buffer("positions", torch.as_tensor(positions, dtype=torch.float32), persistent=False)
        log_weights = torch.as_tensor(log_weights, dtype=torch.float32) if log_weights is not None else None
        self.register_buffer("log_weights", log_weights, persistent=False)

    @classmethod
    def from_neighbor_index(
        cls, neighbor_index: np.ndarray, positions: np.ndarray, num_senders: int
    ) -> "NanoWeatherMeshEdges":
        """Edges from a `(num_receivers, num_neighbors)` index of senders and `(num_receivers, num_neighbors, 4)` positions."""
        num_receivers, num_neighbors = neighbor_index.shape
        receivers = np.repeat(np.arange(num_receivers), num_neighbors)
        senders = neighbor_index.reshape(-1)
        positions = positions.reshape(-1, NUM_POSITION_FEATURES)
        return cls(receivers, senders, positions, num_receivers, num_senders, num_neighbors)


def build_mesh_edges(config: NanoWeatherMeshConfig) -> dict:
    """
    Mesh nodes and the three edge sets of the model, from the config.

    Returns:
        `dict` with `nodes` (unit vectors of the mesh nodes) and `encoder`, `processor` and `decoder` edges.
    """
    nodes = mesh_utils.icosahedral_mesh(config.mesh_refinement)
    points, area_weights = mesh_utils.grid_points(config.grid_size)
    spacing = mesh_utils.mesh_spacing(nodes)

    nearest_nodes = mesh_utils.nearest_neighbors(points, nodes, config.encoder_num_neighbors)
    grid_senders = np.repeat(np.arange(len(points)), config.encoder_num_neighbors)
    node_receivers = nearest_nodes.reshape(-1)
    encoder = NanoWeatherMeshEdges(
        receivers=node_receivers,
        senders=grid_senders,
        positions=mesh_utils.relative_positions(nodes[node_receivers], points[grid_senders], spacing),
        num_receivers=len(nodes),
        num_senders=len(points),
        log_weights=np.log(area_weights[grid_senders]),
    )

    neighbor_index = mesh_utils.nearest_neighbors(nodes, nodes, config.processor_num_neighbors)
    processor = NanoWeatherMeshEdges.from_neighbor_index(
        neighbor_index, mesh_utils.relative_positions(nodes[:, None], nodes[neighbor_index], spacing), len(nodes)
    )

    neighbor_index = mesh_utils.nearest_neighbors(points, nodes, config.decoder_num_neighbors)
    decoder = NanoWeatherMeshEdges.from_neighbor_index(
        neighbor_index, mesh_utils.relative_positions(points[:, None], nodes[neighbor_index], spacing), len(nodes)
    )
    return {"nodes": nodes, "encoder": encoder, "processor": processor, "decoder": decoder}


def sparse_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    edges: NanoWeatherMeshEdges,
    bias: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Attention of each receiver over its senders only.

    Args:
        query (`torch.Tensor` of shape `(batch_size, num_receivers, num_heads, head_dim)`)
        key, value (`torch.Tensor` of shape `(batch_size, num_senders, num_heads, head_dim)`)
        edges (`NanoWeatherMeshEdges`)
        bias (`torch.Tensor` of shape `(num_edges, num_heads)`): added to the attention scores.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_receivers, num_heads, head_dim)`.
    """
    if edges.num_neighbors is not None:
        batch_size, num_receivers, num_heads, head_dim = query.shape
        neighbors = torch.cat([key, value], dim=-1).index_select(1, edges.senders)
        neighbors = neighbors.view(batch_size, num_receivers, edges.num_neighbors, num_heads, 2 * head_dim)
        neighbor_keys, neighbor_values = neighbors.split(head_dim, dim=-1)
        scores = torch.einsum("brhd,brkhd->brhk", query, neighbor_keys) * scaling
        scores = scores + bias.view(edges.num_receivers, edges.num_neighbors, -1).transpose(1, 2)
        weights = scores.float().softmax(dim=-1).to(value.dtype)
        return torch.einsum("brhk,brkhd->brhd", weights, neighbor_values)

    batch_size, num_receivers, num_heads, head_dim = query.shape
    receivers, senders = edges.receivers, edges.senders
    scores = (query.index_select(1, receivers) * key.index_select(1, senders)).sum(-1).float() * scaling + bias
    index = receivers[None, :, None].expand_as(scores)
    maximum = torch.full((batch_size, num_receivers, num_heads), float("-inf"), device=scores.device)
    maximum = maximum.scatter_reduce(1, index, scores, reduce="amax").detach()
    exponentials = torch.exp(scores - maximum[:, receivers])
    totals = torch.zeros_like(maximum).index_add(1, receivers, exponentials)
    weights = (exponentials / totals[:, receivers]).to(value.dtype)
    output = torch.zeros(batch_size, num_receivers, num_heads, head_dim, device=value.device, dtype=value.dtype)
    return output.index_add(1, receivers, weights[..., None] * value.index_select(1, senders))


def dense_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    edges: NanoWeatherMeshEdges,
    bias: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Same result as `sparse_attention_forward`, with full attention and a mask. Memory grows with receivers × senders."""
    num_receivers, num_senders, num_heads = query.shape[1], key.shape[1], query.shape[2]
    flat_index = edges.receivers * num_senders + edges.senders
    mask = torch.zeros(num_heads, num_receivers * num_senders, device=query.device)
    mask = mask.scatter(1, flat_index.expand(num_heads, -1), bias.t().float())
    is_edge = torch.zeros(num_receivers * num_senders, dtype=torch.bool, device=query.device)
    is_edge[flat_index] = True
    mask = mask.masked_fill(~is_edge, float("-inf")).view(num_heads, num_receivers, num_senders)
    output = nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask[None].to(query.dtype),
        scale=scaling,
    )
    return torch.nan_to_num(output.transpose(1, 2), nan=0.0)


ALL_MESH_ATTENTION_FUNCTIONS: dict[str, Callable] = {
    "sparse": sparse_attention_forward,
    "dense": dense_attention_forward,
}


class NanoWeatherMeshAttention(nn.Module):
    """Attention of receivers over their senders, with a learned bias from each sender's relative position."""

    def __init__(self, config: NanoWeatherMeshConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.position_bias = nn.Linear(NUM_POSITION_FEATURES, config.num_attention_heads)

    def forward(self, receivers: torch.Tensor, senders: torch.Tensor, edges: NanoWeatherMeshEdges) -> torch.Tensor:
        batch_size, num_receivers, _ = receivers.shape
        query = self.q_proj(receivers).view(batch_size, num_receivers, self.num_heads, self.head_dim)
        key = self.k_proj(senders).view(batch_size, senders.shape[1], self.num_heads, self.head_dim)
        value = self.v_proj(senders).view(batch_size, senders.shape[1], self.num_heads, self.head_dim)
        bias = self.position_bias(edges.positions)
        if edges.log_weights is not None:
            bias = bias + edges.log_weights[:, None]

        implementation = self.config.attn_implementation
        if implementation == "auto":
            num_pairs = edges.num_receivers * edges.num_senders
            implementation = "dense" if num_pairs <= self.config.dense_attention_max_pairs else "sparse"
        attention_interface = ALL_MESH_ATTENTION_FUNCTIONS[implementation]
        output = attention_interface(query, key, value, edges, bias, self.scaling)
        return self.o_proj(output.reshape(batch_size, num_receivers, -1))


class NanoWeatherMeshMLP(nn.Module):
    def __init__(self, config: NanoWeatherMeshConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation_fn(self.fc1(hidden_states))))


class NanoWeatherMeshLayer(nn.Module):
    """
    Pre-norm transformer layer where receivers attend to senders along edges, then go through an MLP.

    With `cross_attention`, receivers and senders are different sets (grid to mesh, mesh to grid) with their own
    normalization; otherwise the layer is self-attention over the mesh.
    """

    def __init__(self, config: NanoWeatherMeshConfig, cross_attention: bool = False):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_senders = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if cross_attention else None
        )
        self.attention = NanoWeatherMeshAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NanoWeatherMeshMLP(config)

    def forward(
        self, receivers: torch.Tensor, edges: NanoWeatherMeshEdges, senders: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.layernorm_before(receivers)
        senders = self.layernorm_senders(senders) if senders is not None else normalized
        receivers = receivers + self.dropout(self.attention(normalized, senders, edges))
        return receivers + self.mlp(self.layernorm_after(receivers))


class NanoWeatherMeshModel(NanoWeatherPreTrainedModel):
    """Encoder, processor and decoder, without the forecasting head. Returns one hidden state per grid point."""

    def __init__(self, config: NanoWeatherMeshConfig):
        super().__init__(config)
        mesh = build_mesh_edges(config)
        self.encoder_edges, self.processor_edges, self.decoder_edges = (
            mesh["encoder"],
            mesh["processor"],
            mesh["decoder"],
        )
        self.register_buffer("node_positions", torch.as_tensor(mesh["nodes"], dtype=torch.float32), persistent=False)

        self.grid_embeddings = nn.Linear(config.num_input_channels, config.hidden_size)
        self.node_embeddings = nn.Sequential(
            nn.Linear(3, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.encoder = NanoWeatherMeshLayer(config, cross_attention=True)
        self.processor = nn.ModuleList([NanoWeatherMeshLayer(config) for _ in range(config.num_hidden_layers)])
        self.decoder = NanoWeatherMeshLayer(config, cross_attention=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_input_channels, latitude, longitude)`):
                Input states, static fields and time features on the grid.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, latitude * longitude, hidden_size)`, grid points row by row.
        """
        batch_size = pixel_values.shape[0]
        grid = self.grid_embeddings(pixel_values.flatten(2).transpose(1, 2))
        nodes = self.node_embeddings(self.node_positions).expand(batch_size, -1, -1)
        nodes = self.encoder(nodes, self.encoder_edges, senders=grid)
        for layer in self.processor:
            nodes = layer(nodes, self.processor_edges)
        grid = self.decoder(grid, self.decoder_edges, senders=nodes)
        return self.layernorm(grid)


class NanoWeatherMeshForWeatherForecasting(NanoWeatherForecastingModel):
    """
    Mesh model with a linear head that predicts the normalized state one time step ahead, at every grid point.

    With `config.predict_residual`, the head is initialized at zero, so the untrained model predicts no change.
    """

    config_class = NanoWeatherMeshConfig

    def __init__(self, config: NanoWeatherMeshConfig):
        super().__init__(config)
        self.model = NanoWeatherMeshModel(config)
        self.head = nn.Linear(config.hidden_size, config.num_variables)
        self.post_init()
        self.init_output_head(self.head)

    def predict_change(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_hidden_state = self.model(pixel_values)
        change = self.head(last_hidden_state).transpose(1, 2)
        return change.reshape(
            pixel_values.shape[0], self.config.num_variables, *self.config.grid_size
        ), last_hidden_state
