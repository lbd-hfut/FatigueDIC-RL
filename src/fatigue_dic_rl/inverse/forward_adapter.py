"""Observation-only adapter to frozen-damage static FE diagnostic load cases.

Each pattern must contain zero damage: it supplies public geometry/material/BC
information, never a hidden true state. Unknown E0 is substituted from the
candidate vector; patterns then carry only a nominal reference E0.
"""
from dataclasses import dataclass, replace
from collections import OrderedDict
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.mechanics import MaterialAssignment
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.observations.registration import RegistrationError
from .parameterizations import ParameterSpace


@dataclass(frozen=True, eq=False)
class ReactionObservation:
    node_ids: np.ndarray
    component: int
    value_n: float
    standard_deviation_n: float

    def __post_init__(self):
        ids = array(self.node_ids, kind="int", name="reaction nodes")
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or len(np.unique(ids)) != len(ids) or self.component not in (0, 1):
            raise ContractError("reaction needs unique node ids and component 0 or 1")
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "value_n", scalar(self.value_n, "reaction N"))
        s = scalar(self.standard_deviation_n, "reaction standard deviation N", minimum=0)
        if s == 0:
            raise ContractError("reaction likelihood needs positive uncertainty N")
        object.__setattr__(self, "standard_deviation_n", s)


@dataclass(frozen=True)
class StaticLoadCase:
    pattern: object
    measurement: DICMeasurement
    reaction: ReactionObservation | None = None
    variance_floor_mm2: float = 0.
    known_healthy: bool = False


class StaticForwardAdapter:
    nuisance_names = {"E0", "load_factor", "tx", "ty", "theta"}
    nuisance_units = {"E0": "N/mm^2", "load_factor": "1", "tx": "mm", "ty": "mm", "theta": "rad"}

    def __init__(self, damage_model, parameter_space: ParameterSpace, cases, *, mechanics=None):
        self.damage_model, self.parameter_space = damage_model, parameter_space
        self.cases = tuple(cases)
        supported = {p.name for p in damage_model.parameters} | self.nuisance_names
        if (set(parameter_space.names) | set(parameter_space.fixed))-supported:
            raise ContractError("candidate contains parameters unsupported by this forward adapter")
        expected_units = {p.name: p.units for p in damage_model.parameters} | self.nuisance_units
        damage_specs = {p.name: p for p in damage_model.parameters}
        for spec in parameter_space.parameters:
            if spec.units != expected_units[spec.name]:
                raise ContractError("inverse parameter physical units disagree with forward model")
            if spec.name in damage_specs:
                legal = damage_specs[spec.name]
                if spec.lower < legal.lower or spec.upper > legal.upper:
                    raise ContractError("inverse search box exceeds damage model's legal parameter box")
            if spec.name == "load_factor" and spec.lower <= 0:
                raise ContractError("load-factor search box must remain positive")
            if spec.name == "theta" and max(abs(spec.lower), abs(spec.upper)) > .05:
                raise ContractError("registration rotation box exceeds small-angle model")
        self.mechanics = mechanics or Tri3MechanicsSolver()
        if not self.cases or any(not isinstance(c, StaticLoadCase) for c in self.cases):
            raise ContractError("inverse adapter requires StaticLoadCase entries")
        if "E0" in parameter_space.names:
            # Remove any actual/known E0 value from the retained public model.
            # This nominal value is derived solely from the declared search box
            # and is overwritten on every solve. The optimizer never sees E truth.
            spec = next(s for s in parameter_space.parameters if s.name == "E0")
            nominal = (spec.lower+spec.upper)/2
            if spec.lower <= 0:
                raise ContractError("unknown E0 bounds must be strictly positive N/mm^2")
            cases_without_E_truth = []
            for c in self.cases:
                if any("young_modulus_n_per_mm2" not in a.parameters for a in c.pattern.materials.values()):
                    raise ContractError("unknown E0 adapter requires explicit mm plane-stress materials")
                materials = {r: MaterialAssignment(a.constitutive_model, dict(a.parameters, young_modulus_n_per_mm2=nominal))
                             for r, a in c.pattern.materials.items()}
                cases_without_E_truth.append(replace(c, pattern=replace(c.pattern, materials=materials)))
            self.cases = tuple(cases_without_E_truth)
        self.mappings, self.factors = [], []
        for c in self.cases:
            p, o = c.pattern, c.measurement.observation
            if np.any(p.damage.values != 0) or p.internal_variables:
                raise ContractError("public forward patterns must not contain hidden material states")
            if not np.array_equal(p.quadrature_layout.material_state_indices, np.arange(p.quadrature_layout.count)):
                raise ContractError("inverse Tri3 adapter requires identity material-state mapping")
            if o.loading_state_ref != p.loading_state.load_id or o.coordinate_system != p.mesh.coordinate_system:
                raise ContractError("observation and diagnostic load/frame references disagree")
            m = ObservationOperator(p.mesh).prepare(o.coordinates, coordinate_units="mm", coordinate_system=o.coordinate_system)
            if np.any(c.measurement.active_mask & ~m.valid_mask):
                raise ContractError("active DIC point lies outside FEM domain")
            if c.reaction is not None and np.any(c.reaction.node_ids >= p.mesh.n_nodes):
                raise ContractError("reaction node outside mesh")
            self.mappings.append(m)
            self.factors.append(c.measurement.covariance.factor(c.measurement.active_mask,
                variance_floor_mm2=c.variance_floor_mm2))
        self._cache = OrderedDict()
        self.evaluation_count = 0
        self.solver_diagnostics = []

    def predict(self, normalized):
        z = np.asarray(normalized, dtype=float)
        physical = self.parameter_space.decode(z)
        key = z.tobytes()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        predicted = []
        for c, mapping in zip(self.cases, self.mappings):
            p = c.pattern
            layout = p.quadrature_layout
            d = np.zeros(layout.count) if c.known_healthy else self.damage_model.evaluate(layout.coordinates, physical)
            materials = p.materials
            if "E0" in physical:
                if physical["E0"] <= 0:
                    raise ContractError("E0 must be positive N/mm^2")
                materials = {r: MaterialAssignment(a.constitutive_model, dict(a.parameters, young_modulus_n_per_mm2=physical["E0"]))
                             for r, a in p.materials.items()}
            f = physical.get("load_factor", 1.)
            if f <= 0:
                raise ContractError("load factor must be positive")
            candidate = replace(p, damage=layout.field(d, "1"), materials=materials,
                traction_bcs=tuple(replace(b, values=b.values*f) for b in p.traction_bcs),
                nodal_force_bcs=tuple(replace(b, values=b.values*f) for b in p.nodal_force_bcs))
            result = self.mechanics.solve(candidate)
            self.solver_diagnostics.append({"residual_norm": float(np.linalg.norm(result.equilibrium_residual.values)),
                                            "load_id": p.loading_state.load_id})
            u = mapping.apply(result.state.displacement).values.copy()
            if any(k in physical for k in ("tx", "ty", "theta")):
                u += RegistrationError([physical.get("tx", 0.), physical.get("ty", 0.)], physical.get("theta", 0.)).field(mapping.coordinates)
            reaction = None if c.reaction is None else float(result.reaction_forces.values[c.reaction.node_ids, c.reaction.component].sum())
            predicted.append((array(u), reaction))
        self.evaluation_count += 1
        self._cache[key] = tuple(predicted)
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return self._cache[key]

    def whitened_prediction(self, normalized):
        parts = []
        for (u, r), c, factor in zip(self.predict(normalized), self.cases, self.factors):
            parts.append(factor.whiten(u[c.measurement.active_mask].ravel()))
            if r is not None:
                parts.append(np.array([r/c.reaction.standard_deviation_n]))
        return np.concatenate(parts)

    def residual(self, normalized):
        parts = []
        for (u, r), c, factor in zip(self.predict(normalized), self.cases, self.factors):
            e = (u-c.measurement.observation.displacement)[c.measurement.active_mask].ravel()
            parts.append(factor.whiten(e))
            if r is not None:
                parts.append(np.array([(r-c.reaction.value_n)/c.reaction.standard_deviation_n]))
        return np.concatenate(parts)

    def displacement_prediction(self, normalized):
        return np.concatenate([u[c.measurement.active_mask].ravel() for (u, _), c in zip(self.predict(normalized), self.cases)])
