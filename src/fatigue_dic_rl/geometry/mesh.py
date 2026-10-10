"""Immutable 2D mesh topology with ragged connectivity for mixed low-order cells."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fatigue_dic_rl.core.fields import FieldLocation, FieldReference
from fatigue_dic_rl.core.validation import ContractError, array, label, metadata


class ElementType(str, Enum):
    TRI3 = "tri3"
    QUAD4 = "quad4"

    @property
    def node_count(self) -> int:
        return {self.TRI3: 3, self.QUAD4: 4}[self]


@dataclass(frozen=True, eq=False)
class BoundaryTag:
    node_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    facets: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    def __post_init__(self) -> None:
        nodes = array(self.node_ids, kind="int", name="boundary node ids")
        facets = array(self.facets, kind="int", name="boundary facets")
        if nodes.ndim != 1 or facets.ndim != 2 or facets.shape[1] != 2:
            raise ContractError("boundary node ids need [B]; facets need [F, 2]")
        if len(set(nodes.tolist())) != len(nodes) or np.any(facets[:, 0] == facets[:, 1]):
            raise ContractError("boundary ids must be unique and facet endpoints distinct")
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "facets", facets)


@dataclass(frozen=True, eq=False)
class MeshData:
    nodes: np.ndarray
    connectivity: Sequence[np.ndarray]
    element_types: Sequence[ElementType]
    material_region_ids: np.ndarray
    coordinate_units: str
    coordinate_system: str
    boundary_tags: Mapping[str, BoundaryTag] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mesh_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        nodes = array(self.nodes, name="mesh nodes")
        if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3 or not np.all(np.isfinite(nodes)):
            raise ContractError("nodes must be finite [V, 2], V >= 3")
        if self.coordinate_units not in {"m", "mm"}:
            raise ContractError("mesh coordinate units must be m or mm")
        label(self.coordinate_system, "coordinate_system")
        label(self.mesh_id, "mesh_id")
        types = tuple(ElementType(t) for t in self.element_types)
        cells = tuple(array(c, kind="int", name="connectivity") for c in self.connectivity)
        if not cells or len(cells) != len(types):
            raise ContractError("connectivity and element_types require the same nonzero length")
        regions = array(self.material_region_ids, kind="int", name="material regions")
        if regions.shape != (len(cells),) or np.any(regions < 0):
            raise ContractError("material_region_ids must be nonnegative [E]")
        seen = set()
        edge_counts: dict[tuple[int, int], int] = {}
        for cell, cell_type in zip(cells, types):
            if cell.shape != (cell_type.node_count,) or np.any(cell < 0) or np.any(cell >= len(nodes)):
                raise ContractError("invalid cell arity or node index")
            if len(set(cell.tolist())) != len(cell):
                raise ContractError("cell repeats a node")
            key = tuple(sorted(cell.tolist()))
            if key in seen:
                raise ContractError("duplicate element")
            seen.add(key)
            points = nodes[cell]
            with np.errstate(over="ignore", invalid="ignore"):
                edges = np.roll(points, -1, axis=0) - points
                scale = np.max(np.abs(edges))
            if not np.isfinite(scale) or scale == 0:
                raise ContractError("invalid element length scale")
            edges = edges / scale
            cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
            tolerance = 64 * np.finfo(float).eps
            if not (np.all(cross > tolerance) or np.all(cross < -tolerance)):
                raise ContractError("degenerate triangle or non-convex/crossed quadrilateral")
            for i, j in zip(cell, np.roll(cell, -1)):
                edge = tuple(sorted((int(i), int(j))))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if any(n > 2 for n in edge_counts.values()):
            raise ContractError("non-manifold edge shared by more than two cells")
        boundary_edges = {e for e, n in edge_counts.items() if n == 1}
        boundary_nodes = {v for e in boundary_edges for v in e}
        for name, tag in self.boundary_tags.items():
            label(name, "boundary tag")
            if not isinstance(tag, BoundaryTag):
                raise ContractError("boundary tag values must be BoundaryTag")
            if any(int(n) not in boundary_nodes for n in tag.node_ids):
                raise ContractError("boundary tag contains a node outside the mesh boundary")
            if any(tuple(sorted(e.tolist())) not in boundary_edges for e in tag.facets):
                raise ContractError("tagged facet is not an exterior mesh edge")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "connectivity", cells)
        object.__setattr__(self, "element_types", types)
        object.__setattr__(self, "material_region_ids", regions)
        object.__setattr__(self, "boundary_tags", MappingProxyType(dict(self.boundary_tags)))
        object.__setattr__(self, "metadata", metadata(self.metadata))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.connectivity)

    def field_reference(self, location: FieldLocation) -> FieldReference:
        location = FieldLocation(location)
        if location not in {FieldLocation.NODE, FieldLocation.ELEMENT}:
            raise ContractError("quadrature/observation spaces need their own explicit layout reference")
        count = self.n_nodes if location == FieldLocation.NODE else self.n_elements
        return FieldReference(f"{self.mesh_id}:{location.value}", location, count, self.mesh_id)
