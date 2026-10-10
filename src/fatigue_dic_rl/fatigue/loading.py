"""A scalar proportional cycle applied to an explicitly unit-bearing BC pattern."""

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.core.validation import ContractError, label, scalar
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverRequest


class LoadingMode(str, Enum):
    LOAD = "load_controlled"
    DISPLACEMENT = "displacement_controlled"


@dataclass(frozen=True)
class FatigueLoadCycle:
    N_start: float
    delta_N: float
    load_max: float
    load_min: float
    nominal_stress_ratio: float
    boundary_condition_refs: tuple[str, ...]
    loading_mode: LoadingMode
    load_id: str = "proportional-fatigue"
    loading_path: str = "proportional"
    amplitude_units: str = "1"
    ratio_definition: str = "proportional_elastic_reference"
    metadata: Mapping[str, str | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("N_start", "delta_N", "load_max", "load_min", "nominal_stress_ratio"):
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if (self.N_start < 0 or self.delta_N <= 0 or not np.isfinite(self.N_start+self.delta_N) or
                self.N_start+self.delta_N <= self.N_start):
            raise ContractError("N_start >= 0 and delta_N > 0 must give a finite representable endpoint")
        if self.loading_path != "proportional":
            raise NotImplementedError("non-proportional multiaxial paths are unsupported")
        if (self.amplitude_units != "1" or self.ratio_definition != "proportional_elastic_reference" or
                self.load_max <= 0 or self.load_min > self.load_max):
            raise ContractError("use dimensionless factors of one positive maximum BC pattern")
        ratio = self.load_min/self.load_max
        if not np.isfinite(ratio) or not np.isclose(self.nominal_stress_ratio, ratio, rtol=1e-13, atol=1e-14):
            raise ContractError("nominal R must equal load_min/load_max for the proportional reference pattern")
        object.__setattr__(self, "loading_mode", LoadingMode(self.loading_mode))
        refs = tuple(self.boundary_condition_refs)
        if not refs or len(set(refs)) != len(refs):
            raise ContractError("cycle needs unique boundary-condition references")
        for name in (*refs, self.load_id):
            label(name, "cycle reference")
        object.__setattr__(self, "boundary_condition_refs", refs)
        if not isinstance(self.metadata, Mapping):
            raise ContractError("cycle metadata must be a mapping")
        values = {}
        for k, v in self.metadata.items():
            label(k, "metadata key")
            values[k] = v if isinstance(v, str) else scalar(v, k)
        object.__setattr__(self, "metadata", MappingProxyType(values))

    @property
    def N_end(self) -> float:
        return self.N_start+self.delta_N

    def subdivide(self) -> tuple["FatigueLoadCycle", "FatigueLoadCycle"]:
        midpoint = self.N_start+self.delta_N/2
        if midpoint <= self.N_start or midpoint >= self.N_end:
            raise ContractError("cycle midpoint is not representable")
        return (replace(self, delta_N=midpoint-self.N_start),
                replace(self, N_start=midpoint, delta_N=self.N_end-midpoint))

    def request(self, pattern: Tri3SolverRequest, damage, N: float, phase: str) -> Tri3SolverRequest:
        """Homogeneous constraints with scaled forces, or scaled imposed displacement."""
        if not isinstance(pattern, Tri3SolverRequest):
            raise ContractError("cycle requires a Tri3SolverRequest BC pattern")
        if set(self.boundary_condition_refs)-set(pattern.mesh.boundary_tags):
            raise ContractError("cycle boundary references are absent from the pattern mesh")
        if pattern.body_force is not None or pattern.internal_variables:
            raise NotImplementedError("Phase 1B pattern excludes body forces and constitutive history")
        if phase not in {"min", "max"}:
            raise ContractError("cycle phase must be min or max")
        if N < self.N_start or N > self.N_end:
            raise ContractError("response cycle count outside requested interval")
        factor = self.load_max if phase == "max" else self.load_min
        if self.loading_mode == LoadingMode.LOAD:
            if any(np.any(bc.values != 0) for bc in pattern.displacement_bcs):
                raise NotImplementedError("load control requires homogeneous imposed displacements")
            if not any(np.any(bc.values != 0) for bc in (*pattern.traction_bcs, *pattern.nodal_force_bcs)):
                raise ContractError("load control requires a nonzero force pattern")
            bcs = pattern.displacement_bcs
            tractions = tuple(replace(bc, values=bc.values*factor) for bc in pattern.traction_bcs)
            forces = tuple(replace(bc, values=bc.values*factor) for bc in pattern.nodal_force_bcs)
        else:
            if pattern.traction_bcs or pattern.nodal_force_bcs:
                raise NotImplementedError("displacement control excludes independent force patterns")
            if not any(np.any(bc.values != 0) for bc in pattern.displacement_bcs):
                raise ContractError("displacement control requires a nonzero displacement pattern")
            bcs = tuple(replace(bc, values=bc.values*factor) for bc in pattern.displacement_bcs)
            tractions, forces = (), ()
        return replace(pattern, damage=damage, cycle_count=N,
                       loading_state=LoadingState(self.load_id, phase),
                       displacement_bcs=bcs, traction_bcs=tractions, nodal_force_bcs=forces)

