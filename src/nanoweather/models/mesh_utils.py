"""Geometry for models on an icosahedral mesh: mesh nodes, grid points, neighbor lists and relative positions.

Everything here is deterministic and computed from the model config at load time, so none of it is stored in
checkpoints. Points are unit vectors in 3D; distances are chord lengths, which order points like geodesic distances.
"""

import numpy as np
from scipy.spatial import cKDTree


def icosahedral_mesh(refinement: int) -> np.ndarray:
    """
    Vertices of an icosahedron whose faces are split into 4 triangles `refinement` times, projected on the sphere.

    Returns:
        `np.ndarray` of shape `(10 * 4**refinement + 2, 3)`: unit vectors.
    """
    phi = (1 + 5**0.5) / 2
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    vertices = [np.array(vertex, dtype=np.float64) / np.linalg.norm(vertex) for vertex in vertices]

    for _ in range(refinement):
        midpoints = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                middle = vertices[a] + vertices[b]
                vertices.append(middle / np.linalg.norm(middle))
                midpoints[key] = len(vertices) - 1
            return midpoints[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.stack(vertices)


def grid_latitudes_longitudes(grid_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes (south to north, no points at the poles) and longitudes (from 0) of an equiangular grid, in degrees."""
    num_latitudes, num_longitudes = grid_size
    latitude = -90 + (np.arange(num_latitudes) + 0.5) * 180 / num_latitudes
    longitude = np.arange(num_longitudes) * 360 / num_longitudes
    return latitude, longitude


def latitude_longitude_to_xyz(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Unit vectors for points given in degrees; inputs broadcast together."""
    latitude, longitude = np.broadcast_arrays(np.deg2rad(latitude), np.deg2rad(longitude))
    return np.stack(
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=-1
    )


def grid_points(grid_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Unit vectors and area weights of the points of an equiangular grid, flattened row by row (latitude-major).

    Returns:
        `points` of shape `(num_latitudes * num_longitudes, 3)` and `area_weights` of the same length, proportional
        to cos(latitude) and with mean 1.
    """
    latitude, longitude = grid_latitudes_longitudes(grid_size)
    points = latitude_longitude_to_xyz(latitude[:, None], longitude[None, :]).reshape(-1, 3)
    weights = np.repeat(np.cos(np.deg2rad(latitude)), len(longitude))
    return points, weights / weights.mean()


def nearest_neighbors(receivers: np.ndarray, senders: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Indices in `senders` of the `num_neighbors` nearest senders of each receiver, nearest first."""
    _, indices = cKDTree(senders).query(receivers, k=num_neighbors)
    return indices.reshape(len(receivers), num_neighbors)


def relative_positions(receivers: np.ndarray, senders: np.ndarray, scale: float) -> np.ndarray:
    """
    Position of each sender relative to its receiver, in the receiver's local frame.

    Args:
        receivers (`np.ndarray` of shape `(..., 3)`):
            Unit vectors of the receiving points.
        senders (`np.ndarray` of shape `(..., 3)`):
            Unit vectors of the sending points, paired with `receivers` (shapes broadcast together).
        scale (`float`):
            Length dividing all features, e.g. the mesh spacing.

    Returns:
        `np.ndarray` of shape `(..., 4)`: eastward, northward and upward components of the displacement, and its
        length. At the poles, where east is undefined, a fixed horizontal direction is used.
    """
    receivers, senders = np.broadcast_arrays(receivers, senders)
    up = receivers
    east = np.cross(np.array([0.0, 0.0, 1.0]), up)
    at_pole = np.linalg.norm(east, axis=-1) < 1e-8
    east[at_pole] = np.array([0.0, 1.0, 0.0])
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(up, east)
    displacement = senders - receivers
    features = np.stack(
        [
            (displacement * east).sum(-1),
            (displacement * north).sum(-1),
            (displacement * up).sum(-1),
            np.linalg.norm(displacement, axis=-1),
        ],
        axis=-1,
    )
    return features / scale


def mesh_spacing(nodes: np.ndarray) -> float:
    """Median chord distance between a node and its nearest other node."""
    distances, _ = cKDTree(nodes).query(nodes, k=2)
    return float(np.median(distances[:, 1]))
