"""Solver-independent linear triangle shape-function observation mapping."""

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from fatigue_dic_rl.core import FieldData, FieldLocation
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.geometry import ElementType, MeshData


class UnsupportedElementError(NotImplementedError):
    """The mesh is valid, but this operator has no shape functions for its cells."""


@dataclass(frozen=True, eq=False)
class ObservationMapping:
    mesh: MeshData
    coordinates: np.ndarray
    node_ids: np.ndarray
    weights: np.ndarray
    element_ids: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        for name, kind in (("coordinates", "float"), ("node_ids", "int"), ("weights", "float"),
                           ("element_ids", "int"), ("valid_mask", "bool")):
            object.__setattr__(self, name, array(getattr(self, name), kind=kind, name=name))
        count = len(self.coordinates)
        if (self.coordinates.shape != (count, 2) or self.node_ids.shape != (count, 3) or
                self.weights.shape != (count, 3) or self.element_ids.shape != (count,) or
                self.valid_mask.shape != (count,)):
            raise ContractError("invalid observation mapping shapes")
        valid = self.valid_mask
        if not np.all(np.isfinite(self.coordinates)) or not np.all(np.isfinite(self.weights)):
            raise ContractError("mapping coordinates/weights must be finite")
        if (np.any(self.node_ids[valid] < 0) or np.any(self.node_ids[valid] >= self.mesh.n_nodes) or
                np.any(self.element_ids[valid] < 0) or np.any(self.element_ids[valid] >= self.mesh.n_elements)):
            raise ContractError("mapping indices outside mesh")
        for element_id in np.unique(self.element_ids[valid]):
            selected = valid & (self.element_ids == element_id)
            if not np.all(self.node_ids[selected] == self.mesh.connectivity[element_id]):
                raise ContractError("mapping node indices disagree with the selected element")
        if (np.any(self.weights[valid] < 0) or
                not np.allclose(self.weights[valid].sum(axis=1), 1, rtol=0, atol=8*np.finfo(float).eps)):
            raise ContractError("invalid shape-function weights")
        if (np.any(self.node_ids[~valid] != -1) or np.any(self.element_ids[~valid] != -1) or
                np.any(self.weights[~valid] != 0)):
            raise ContractError("invalid points need sentinel indices and zero weights")

    def apply(self, displacement: FieldData) -> FieldData:
        displacement.require(location=FieldLocation.NODE, components=(2,))
        if displacement.units not in {"m", "mm"}:
            raise ContractError("displacement must explicitly use m or mm")
        if displacement.reference != self.mesh.field_reference(FieldLocation.NODE):
            raise ContractError("displacement reference does not match mapping mesh nodes")
        if displacement.coordinates is not None:
            if (displacement.coordinate_units != self.mesh.coordinate_units or
                    displacement.coordinate_system != self.mesh.coordinate_system or
                    not np.array_equal(displacement.coordinates, self.mesh.nodes)):
                raise ContractError("nodal displacement coordinates disagree with mesh")
        mask = self.valid_mask.copy()
        candidates = np.flatnonzero(mask)
        mask[candidates] &= displacement.valid_mask[self.node_ids[candidates]].all(axis=1)
        values = np.full((len(mask), 2), np.nan)
        valid = np.flatnonzero(mask)
        values[valid] = np.einsum("qi,qij->qj", self.weights[valid], displacement.values[self.node_ids[valid]])
        return FieldData(values, FieldLocation.OBSERVATION, displacement.units,
                         coordinates=self.coordinates, coordinate_units=self.mesh.coordinate_units,
                         coordinate_system=self.mesh.coordinate_system, valid_mask=mask,
                         metadata={"operator": "tri3_shape_functions", "mesh_id": self.mesh.mesh_id})


class ObservationOperator:
    """Prepare O(E Q) mappings; cache a bounded number of immutable query sets.

    Tolerance is in dimensionless barycentric coordinates. Points within this
    declared roundoff tolerance are snapped to a boundary, never extrapolated.
    Shared-edge ties use the first element in connectivity order.
    """

    def __init__(self, mesh: MeshData, *, tolerance: float = 64*np.finfo(float).eps,
                 cache_size: int = 8) -> None:
        if any(t != ElementType.TRI3 for t in mesh.element_types):
            raise UnsupportedElementError("Phase 0 operator supports tri3 only; quad4/mixed meshes need a new adapter")
        self._mesh = mesh
        self._tolerance = scalar(tolerance, "barycentric tolerance", minimum=0)
        if self.tolerance > 1e-10:
            raise ContractError("barycentric tolerance must stay within the roundoff range <= 1e-10")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ContractError("cache_size must be a nonnegative integer")
        self._cache_size = cache_size
        self._cache: OrderedDict[str, ObservationMapping] = OrderedDict()

    @property
    def mesh(self) -> MeshData:
        return self._mesh

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def cache_size(self) -> int:
        return self._cache_size

    def prepare(self, coordinates: np.ndarray, *, coordinate_units: str,
                coordinate_system: str) -> ObservationMapping:
        points = array(coordinates, name="query coordinates")
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ContractError("query coordinates must be finite [Q,2]")
        if coordinate_units != self.mesh.coordinate_units or coordinate_system != self.mesh.coordinate_system:
            raise ContractError("FE and DIC coordinates need explicit unit/frame conversion before mapping")
        key = sha256(str(points.shape).encode() + points.tobytes()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        count = len(points)
        ids = np.full((count, 3), -1, dtype=int)
        elements = np.full(count, -1, dtype=int)
        weights = np.zeros((count, 3))
        valid = np.zeros(count, dtype=bool)
        for element_id, cell in enumerate(self.mesh.connectivity):
            remaining = np.flatnonzero(~valid)
            if not remaining.size:
                break
            triangle = self.mesh.nodes[cell]
            edges = (triangle[1:] - triangle[0]).T
            scale = np.max(np.abs(edges))
            with np.errstate(over="ignore", invalid="ignore"):
                relative = (points[remaining] - triangle[0]) / scale
            # Far-away queries may overflow subtraction: they are outside, not extrapolated.
            finite = np.all(np.isfinite(relative), axis=1)
            remaining = remaining[finite]
            uv = np.linalg.solve(edges / scale, relative[finite].T).T
            bary = np.column_stack((1 - uv.sum(axis=1), uv))
            inside = np.all(bary >= -self.tolerance, axis=1) & np.all(bary <= 1+self.tolerance, axis=1)
            chosen = remaining[inside]
            chosen_weights = np.clip(bary[inside], 0, 1)
            chosen_weights /= chosen_weights.sum(axis=1, keepdims=True)
            ids[chosen] = cell
            weights[chosen] = chosen_weights
            elements[chosen] = element_id
            valid[chosen] = True
        mapping = ObservationMapping(self.mesh, points, ids, weights, elements, valid)
        if self.cache_size:
            self._cache[key] = mapping
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return mapping

    def clear_cache(self) -> None:
        self._cache.clear()
