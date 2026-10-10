"""Additive Phase 1A contracts; legacy SolverRequest and Pa inputs remain valid."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fatigue_dic_rl.core import FieldData, LoadingState
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.geometry import MeshData, QuadratureLayout
from .solver import MaterialAssignment, SolverRequest


@dataclass(frozen=True, eq=False)
class BodyForceDensity:
    values: np.ndarray | Callable[[np.ndarray], np.ndarray]
    units: str = "N/mm^3"

    def __post_init__(self) -> None:
        if self.units != "N/mm^3":
            raise ContractError("body-force density must explicitly use N/mm^3")
        if not callable(self.values):
            values = array(self.values)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ContractError("constant body-force density must be finite [2]")
            object.__setattr__(self, "values", values)

    def evaluate(self, coordinates: np.ndarray) -> np.ndarray:
        points = array(coordinates)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ContractError("body-force evaluation requires finite [Q,2] coordinates")
        values = array(self.values(points) if callable(self.values) else np.tile(self.values, (len(points), 1)))
        if values.shape != (len(points), 2) or not np.isfinite(values).all():
            raise ContractError("body-force function must return finite [Q,2] values")
        return values


@dataclass(frozen=True, eq=False)
class Tri3SolverRequest(SolverRequest):
    damage: FieldData | None
    quadrature_layout: QuadratureLayout | None = None
    body_force: BodyForceDensity | None = None

    def __post_init__(self) -> None:
        layout = self.quadrature_layout
        if layout is None:
            layout = QuadratureLayout.tri3_centroid(self.mesh)
        if (layout.mesh.mesh_id != self.mesh.mesh_id or
                layout.mesh.coordinate_units != self.mesh.coordinate_units or
                layout.mesh.coordinate_system != self.mesh.coordinate_system or
                not np.array_equal(layout.mesh.nodes, self.mesh.nodes) or
                len(layout.mesh.connectivity) != len(self.mesh.connectivity) or
                any(not np.array_equal(a, b) for a, b in zip(layout.mesh.connectivity, self.mesh.connectivity))):
            raise ContractError("quadrature layout and request mesh disagree")
        object.__setattr__(self, "quadrature_layout", layout)
        if self.damage is None:
            object.__setattr__(self, "damage", layout.field(np.zeros(layout.count), "1"))
        if self.body_force is not None and not isinstance(self.body_force, BodyForceDensity):
            raise ContractError("body_force must be BodyForceDensity")
        super().__post_init__()

    @property
    def thickness_mm(self) -> float:
        """Explicit conversion of the retained Phase 0 thickness_m field."""
        return self.thickness_m*1000

    @classmethod
    def from_mm(cls, mesh: MeshData, materials: dict[int, MaterialAssignment],
                loading_state: LoadingState, *, thickness_mm: float,
                cycle_count: float = 0, damage: FieldData | None = None, **kwargs) -> "Tri3SolverRequest":
        thickness = scalar(thickness_mm, "thickness_mm", minimum=0)
        if thickness == 0 or "thickness_m" in kwargs:
            raise ContractError("use one positive thickness_mm value, without a competing thickness_m")
        return cls(mesh, damage, materials, loading_state, cycle_count, thickness_m=thickness/1000, **kwargs)


@dataclass(frozen=True)
class Tri3SolverConfig:
    damage_max: float = 0.999
    residual_atol_n: float = 1e-9
    residual_rtol: float = 1e-10
    symmetry_rtol: float = 1e-12
    rigid_rank_tolerance: float = 1e-12
    minimum_pivot_ratio: float = 1e-14
    load_quadrature_order: int = 5

    def __post_init__(self) -> None:
        maximum = scalar(self.damage_max, "damage_max", minimum=0)
        if not maximum < 1:
            raise ContractError("fixed-damage mechanics needs 0 <= damage_max < 1")
        object.__setattr__(self, "damage_max", maximum)
        for name in ("residual_atol_n", "residual_rtol", "symmetry_rtol", "rigid_rank_tolerance", "minimum_pivot_ratio"):
            value = scalar(getattr(self, name), name, minimum=0)
            if value == 0:
                raise ContractError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.symmetry_rtol > 1e-8 or self.rigid_rank_tolerance >= 1 or self.minimum_pivot_ratio >= 1:
            raise ContractError("invalid symmetry/rank/pivot tolerance")
        from fatigue_dic_rl.geometry.triangles import triangle_gauss_rule
        triangle_gauss_rule(self.load_quadrature_order)
        if self.load_quadrature_order < 2:
            raise ContractError("consistent Tri3 body loads require Duffy Gaussian order >= 2")
