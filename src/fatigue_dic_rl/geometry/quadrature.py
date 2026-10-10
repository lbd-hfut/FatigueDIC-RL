"""Explicit material-state quadrature layouts; G is independent of V and E."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from fatigue_dic_rl.core import FieldData, FieldLocation, FieldReference
from fatigue_dic_rl.core.validation import ContractError, array, label
from .mesh import ElementType, MeshData
from .triangles import Tri3Geometry, tri3_shape_functions


@dataclass(frozen=True, eq=False)
class QuadratureLayout:
    mesh: MeshData
    rule_name: str
    reference_points: np.ndarray
    reference_weights: np.ndarray
    coordinates: np.ndarray
    jacobians: np.ndarray
    signed_jacobian_determinants: np.ndarray
    quadrature_to_element: np.ndarray
    element_to_quadrature: tuple[np.ndarray, ...]
    material_state_indices: np.ndarray

    def __post_init__(self) -> None:
        label(self.rule_name, "quadrature rule name")
        if any(t != ElementType.TRI3 for t in self.mesh.element_types):
            raise NotImplementedError("this quadrature contract currently supports affine tri3 cells")
        for name in ("reference_points", "reference_weights", "coordinates", "jacobians", "signed_jacobian_determinants"):
            object.__setattr__(self, name, array(getattr(self, name), name=name))
        for name in ("quadrature_to_element", "material_state_indices"):
            object.__setattr__(self, name, array(getattr(self, name), kind="int", name=name))
        groups = tuple(array(g, kind="int") for g in self.element_to_quadrature)
        object.__setattr__(self, "element_to_quadrature", groups)
        count = len(self.reference_weights)
        if (self.reference_points.shape != (count, 2) or self.coordinates.shape != (count, 2) or
                self.jacobians.shape != (count, 2, 2) or self.signed_jacobian_determinants.shape != (count,) or
                self.quadrature_to_element.shape != (count,) or self.material_state_indices.shape != (count,) or
                len(groups) != self.mesh.n_elements):
            raise ContractError("invalid quadrature layout shapes")
        if count == 0 or any(not np.isfinite(getattr(self, n)).all() for n in
                             ("reference_points", "reference_weights", "coordinates", "jacobians", "signed_jacobian_determinants")):
            raise ContractError("quadrature layout must be nonempty and finite")
        if np.any(self.reference_weights <= 0) or np.any(tri3_shape_functions(self.reference_points) < -1e-14):
            raise ContractError("quadrature weights must be positive and points inside reference triangles")
        if sorted(self.material_state_indices.tolist()) != list(range(count)):
            raise ContractError("material_state_indices must be a permutation of [0,G)")
        all_ids = []
        for e, ids in enumerate(groups):
            if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= count):
                raise ContractError("each element needs nonempty valid quadrature ids")
            if not np.all(self.quadrature_to_element[ids] == e):
                raise ContractError("element/quadrature ownership mappings disagree")
            if not np.isclose(self.reference_weights[ids].sum(), 0.5, rtol=0, atol=2e-14):
                raise ContractError("per-element reference weights must sum to 1/2")
            vertices = self.mesh.nodes[self.mesh.connectivity[e]]
            geometry = Tri3Geometry.from_vertices(vertices)
            expected = tri3_shape_functions(self.reference_points[ids]) @ vertices
            if (not np.allclose(self.coordinates[ids], expected, rtol=1e-13, atol=1e-14) or
                    not np.allclose(self.jacobians[ids], geometry.jacobian, rtol=1e-13, atol=0) or
                    not np.allclose(self.signed_jacobian_determinants[ids], geometry.signed_determinant, rtol=1e-13, atol=0)):
                raise ContractError("quadrature geometry disagrees with mesh")
            all_ids.extend(ids.tolist())
        if sorted(all_ids) != list(range(count)):
            raise ContractError("quadrature ids must partition [0,G)")

    @classmethod
    def tri3_centroid(cls, mesh: MeshData) -> "QuadratureLayout":
        if any(t != ElementType.TRI3 for t in mesh.element_types):
            raise NotImplementedError("centroid quadrature requires tri3")
        geometry = [Tri3Geometry.from_vertices(mesh.nodes[c]) for c in mesh.connectivity]
        count = mesh.n_elements
        return cls(mesh, "tri3-centroid-v1", np.full((count, 2), 1/3), np.full(count, 0.5),
                   np.array([mesh.nodes[c].mean(axis=0) for c in mesh.connectivity]),
                   np.array([g.jacobian for g in geometry]), np.array([g.signed_determinant for g in geometry]),
                   np.arange(count), tuple(np.array([i]) for i in range(count)), np.arange(count))

    @property
    def count(self) -> int:
        return len(self.reference_weights)

    @property
    def physical_weights(self) -> np.ndarray:
        return array(self.reference_weights*np.abs(self.signed_jacobian_determinants))

    @property
    def reference(self) -> FieldReference:
        payload = b"".join(getattr(self, n).tobytes() for n in
                           ("reference_points", "reference_weights", "quadrature_to_element", "material_state_indices"))
        identity = sha256(payload).hexdigest()[:16]
        return FieldReference(f"{self.mesh.mesh_id}:{self.rule_name}:{identity}", FieldLocation.QUADRATURE,
                              self.count, self.mesh.mesh_id)

    def field(self, values: np.ndarray, units: str, *, metadata: dict | None = None) -> FieldData:
        return FieldData(values, FieldLocation.QUADRATURE, units, reference=self.reference,
                         metadata={} if metadata is None else metadata)

    def assert_field(self, value: FieldData) -> None:
        value.require(location=FieldLocation.QUADRATURE, fully_valid=True)
        if value.reference != self.reference:
            raise ContractError("field does not belong to this quadrature layout")
        if value.coordinates is not None and (not np.array_equal(value.coordinates, self.coordinates) or
                value.coordinate_units != self.mesh.coordinate_units or value.coordinate_system != self.mesh.coordinate_system):
            raise ContractError("quadrature field coordinates/frame disagree with the layout")

    def project_element_field(self, value: FieldData) -> FieldData:
        """Explicit piecewise-constant lift; never interprets a node field as an element field."""
        value.require(location=FieldLocation.ELEMENT, fully_valid=True)
        if value.reference != self.mesh.field_reference(FieldLocation.ELEMENT):
            raise ContractError("element field belongs to a different mesh")
        return self.field(value.values[self.quadrature_to_element], value.units,
                          metadata=dict(value.metadata, projection="element_constant_to_quadrature"))

    def element_average(self, value: FieldData) -> FieldData:
        self.assert_field(value)
        weights = self.physical_weights
        average = np.array([np.tensordot(weights[ids]/weights[ids].sum(), value.values[ids], axes=1)
                            for ids in self.element_to_quadrature])
        return FieldData(average, FieldLocation.ELEMENT, value.units,
                         reference=self.mesh.field_reference(FieldLocation.ELEMENT),
                         metadata=dict(value.metadata, projection="quadrature_volume_average"))
