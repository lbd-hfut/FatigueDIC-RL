"""Backend-independent solver contracts. No FE backend is implemented in Phase 0."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import FieldData, FieldLocation, LoadingState, MechanicsState, StateRole
from fatigue_dic_rl.core.validation import ContractError, array, label, scalar
from fatigue_dic_rl.geometry import MeshData
from fatigue_dic_rl.materials import ConstitutiveModel


@dataclass(frozen=True, eq=False)
class DisplacementBC:
    node_ids: np.ndarray
    components: np.ndarray
    values: np.ndarray
    units: str

    def __post_init__(self) -> None:
        nodes = array(self.node_ids, kind="int")
        components = array(self.components, kind="int")
        values = array(self.values)
        if nodes.ndim != 1 or components.shape != nodes.shape or values.shape != nodes.shape:
            raise ContractError("displacement BC node/component/value arrays need equal [B] shape")
        if np.any(nodes < 0) or np.any(components < 0) or np.any(components > 1) or not np.all(np.isfinite(values)):
            raise ContractError("invalid displacement BC index/component/value")
        if self.units not in {"m", "mm"}:
            raise ContractError("displacement BC units must be m or mm")
        for name, value in (("node_ids", nodes), ("components", components), ("values", values)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class NodalForceBC:
    node_ids: np.ndarray
    values: np.ndarray
    units: str = "N"

    def __post_init__(self) -> None:
        nodes = array(self.node_ids, kind="int")
        values = array(self.values)
        if nodes.ndim != 1 or values.shape != (len(nodes), 2) or np.any(nodes < 0) or not np.all(np.isfinite(values)):
            raise ContractError("nodal force BC needs nonnegative [B] ids and finite [B,2] forces")
        if self.units != "N":
            raise ContractError("nodal forces must explicitly use N")
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, eq=False)
class TractionBC:
    boundary_tag: str
    values: np.ndarray
    units: str = "Pa"

    def __post_init__(self) -> None:
        label(self.boundary_tag, "traction boundary_tag")
        values = array(self.values)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or self.units not in {"Pa", "N/mm^2"}:
            raise ContractError("constant boundary traction needs finite [2] values in Pa or N/mm^2")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class MaterialAssignment:
    constitutive_model: ConstitutiveModel
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.constitutive_model, ConstitutiveModel):
            raise ContractError("material assignment needs a ConstitutiveModel")
        parameters = {label(k, "material parameter"): scalar(v, k) for k, v in self.parameters.items()}
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True, eq=False)
class SolverRequest:
    mesh: MeshData
    damage: FieldData
    materials: Mapping[int, MaterialAssignment]
    loading_state: LoadingState
    cycle_count: float
    displacement_bcs: tuple[DisplacementBC, ...] = ()
    nodal_force_bcs: tuple[NodalForceBC, ...] = ()
    traction_bcs: tuple[TractionBC, ...] = ()
    thickness_m: float = 1.0
    internal_variables: Mapping[str, FieldData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.damage.require(units="1", components=(), fully_valid=True)
        if (self.damage.location == FieldLocation.OBSERVATION or self.damage.reference is None or
                self.damage.reference.mesh_id != self.mesh.mesh_id):
            raise ContractError("solver damage needs a mechanics space on the request mesh")
        if np.any(self.damage.values < 0) or np.any(self.damage.values > 1):
            raise ContractError("solver damage outside [0,1]")
        if self.damage.location in {FieldLocation.NODE, FieldLocation.ELEMENT}:
            if self.damage.reference != self.mesh.field_reference(self.damage.location):
                raise ContractError("solver damage reference/count mismatch")
        if not isinstance(self.loading_state, LoadingState):
            raise ContractError("solver loading_state needs LoadingState")
        if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0 for k in self.materials):
            raise ContractError("material region keys must be nonnegative integers")
        if set(np.unique(self.mesh.material_region_ids)) - set(self.materials):
            raise ContractError("material assignment missing a mesh region")
        if not all(isinstance(v, MaterialAssignment) for v in self.materials.values()):
            raise ContractError("materials need MaterialAssignment values")
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "cycle_count", minimum=0))
        thickness = scalar(self.thickness_m, "thickness_m", minimum=0)
        if thickness == 0:
            raise ContractError("thickness_m must be positive")
        object.__setattr__(self, "thickness_m", thickness)
        prescribed: dict[tuple[int, int], float] = {}
        for bc in self.displacement_bcs:
            if not isinstance(bc, DisplacementBC) or np.any(bc.node_ids >= self.mesh.n_nodes):
                raise ContractError("invalid displacement BC node reference")
            if bc.units != self.mesh.coordinate_units:
                raise ContractError("displacement BC units must match mesh length units")
            for node, component, value in zip(bc.node_ids, bc.components, bc.values):
                key = (int(node), int(component))
                if key in prescribed and prescribed[key] != value:
                    raise ContractError("conflicting prescribed displacement")
                prescribed[key] = float(value)
        for bc in self.nodal_force_bcs:
            if not isinstance(bc, NodalForceBC) or np.any(bc.node_ids >= self.mesh.n_nodes):
                raise ContractError("invalid force BC node reference")
        for bc in self.traction_bcs:
            if not isinstance(bc, TractionBC) or bc.boundary_tag not in self.mesh.boundary_tags:
                raise ContractError("unknown traction boundary tag")
            if not len(self.mesh.boundary_tags[bc.boundary_tag].facets):
                raise ContractError("traction boundary tag must contain exterior facets")
        for name, value in self.internal_variables.items():
            label(name, "internal variable name")
            if (not isinstance(value, FieldData) or value.reference is None or
                    value.reference.mesh_id != self.mesh.mesh_id or value.location == FieldLocation.OBSERVATION):
                raise ContractError("solver internal variables need mechanics fields on this mesh")
        object.__setattr__(self, "materials", MappingProxyType(dict(self.materials)))
        object.__setattr__(self, "internal_variables", MappingProxyType(dict(self.internal_variables)))
        for name in ("displacement_bcs", "nodal_force_bcs", "traction_bcs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class SolverDiagnostics:
    converged: bool
    backend: str
    iterations: int
    residual_norm: float
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.converged, bool):
            raise ContractError("converged must be bool")
        label(self.backend, "backend")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations < 0:
            raise ContractError("iterations must be a nonnegative integer")
        object.__setattr__(self, "residual_norm", scalar(self.residual_norm, "residual_norm", minimum=0))


@dataclass(frozen=True)
class SolverResult:
    state: MechanicsState
    diagnostics: SolverDiagnostics


class MechanicsConvergenceError(RuntimeError):
    def __init__(self, diagnostics: SolverDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(f"{diagnostics.backend} did not converge: {diagnostics.message}")


class MechanicsBackendError(RuntimeError):
    """Backend numerical error; the original exception is available as __cause__."""


class MechanicsSolver(ABC):
    """Template interface: subclasses implement only _solve, with no NN dependency.

    Backend adapters must explicitly reject unsupported materials, elements,
    quadrature layouts or BCs. Numerical convergence does not establish
    scientific validation. Residual norm meaning and tolerance are backend
    documentation responsibilities.
    """

    def solve(self, request: SolverRequest) -> SolverResult:
        if not isinstance(request, SolverRequest):
            raise ContractError("solver needs SolverRequest")
        try:
            result = self._solve(request)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise MechanicsBackendError(str(exc)) from exc
        if not isinstance(result, SolverResult):
            raise ContractError("backend must return SolverResult")
        if not result.diagnostics.converged:
            raise MechanicsConvergenceError(result.diagnostics)
        state = result.state
        if (state.role != StateRole.PREDICTION or state.mesh_id != request.mesh.mesh_id or
                state.cycle_count != request.cycle_count or state.loading_state != request.loading_state):
            raise ContractError("backend returned the wrong response role/mesh/cycle/load")
        request.damage.assert_compatible(state.damage)
        if not np.array_equal(request.damage.values, state.damage.values):
            raise ContractError("equilibrium solver may not evolve damage")
        if state.displacement is None or state.strain is None or state.stress is None:
            raise ContractError("successful mechanics solve needs displacement, strain and stress")
        state.strain.assert_same_support(state.stress)
        if state.strain.values.shape != state.stress.values.shape:
            raise ContractError("solver strain/stress component shapes differ")
        for value in (state.strain, state.stress):
            if value.location in {FieldLocation.NODE, FieldLocation.ELEMENT}:
                if value.reference != request.mesh.field_reference(value.location):
                    raise ContractError("solver strain/stress reference/count mismatch")
        if (state.displacement.reference != request.mesh.field_reference(FieldLocation.NODE) or
                state.displacement.units != request.mesh.coordinate_units):
            raise ContractError("solver nodal displacement reference/units mismatch")
        if state.displacement.coordinates is not None:
            if (state.displacement.coordinate_units != request.mesh.coordinate_units or
                    state.displacement.coordinate_system != request.mesh.coordinate_system or
                    not np.array_equal(state.displacement.coordinates, request.mesh.nodes)):
                raise ContractError("solver displacement coordinates disagree with mesh")
        return result

    @abstractmethod
    def _solve(self, request: SolverRequest) -> SolverResult:
        """Compute equilibrium and report numerical diagnostics; no implementation here."""
