"""Reproducible verification meshes; not a general meshing backend."""

import numpy as np
from scipy.spatial import Delaunay

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import BoundaryTag, MeshData


def rectangle_mesh(nx: int, ny: int, *, width_mm: float = 1, height_mm: float = 1,
                   seed: int = 19, jitter: float = 0.2, reverse_alternating: bool = False) -> MeshData:
    if (isinstance(nx, bool) or isinstance(ny, bool) or not isinstance(nx, int) or
            not isinstance(ny, int) or nx < 1 or ny < 1 or
            not np.isfinite([width_mm, height_mm, jitter]).all() or
            width_mm <= 0 or height_mm <= 0 or not 0 <= jitter < 0.4):
        raise ContractError("verification rectangle needs positive lengths/subdivisions and 0<=jitter<0.4")
    x, y = np.meshgrid(np.linspace(0, width_mm, nx+1), np.linspace(0, height_mm, ny+1))
    nodes = np.column_stack((x.ravel(), y.ravel()))
    interior = ((nodes[:, 0] > 0) & (nodes[:, 0] < width_mm) &
                (nodes[:, 1] > 0) & (nodes[:, 1] < height_mm))
    rng = np.random.default_rng(seed)
    nodes[interior] += rng.uniform(-jitter, jitter, (interior.sum(), 2))*[width_mm/nx, height_mm/ny]
    cells = np.array(sorted(Delaunay(nodes).simplices.tolist(), key=lambda c: tuple(sorted(c))))
    if reverse_alternating:
        cells[::2] = cells[::2, ::-1]
    edge_counts = {}
    for cell in cells:
        for a, b in zip(cell, np.roll(cell, -1)):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0)+1
    boundaries = {}
    for name, axis, coordinate in (("left", 0, 0), ("right", 0, width_mm),
                                    ("bottom", 1, 0), ("top", 1, height_mm)):
        ids = np.flatnonzero(np.isclose(nodes[:, axis], coordinate, rtol=0, atol=1e-13*max(width_mm, height_mm)))
        idset = set(ids.tolist())
        facets = [edge for edge, count in edge_counts.items() if count == 1 and set(edge) <= idset]
        boundaries[name] = BoundaryTag(ids, np.array(facets, dtype=int).reshape((-1, 2)))
    identity = f"verification-rectangle-{nx}-{ny}-{width_mm:g}-{height_mm:g}-{seed}-{jitter:g}-{reverse_alternating}"
    return MeshData(nodes, cells, ["tri3"]*len(cells), np.zeros(len(cells), dtype=int), "mm", "specimen",
                    boundaries, {"generator": "seeded_jittered_Delaunay", "seed": seed}, identity)
