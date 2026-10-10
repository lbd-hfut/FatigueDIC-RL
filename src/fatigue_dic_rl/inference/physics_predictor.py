"""Fresh candidate-specific Phase 1B histories and new-damage FEM observations.

Integration anchors depend only on public load segments and cycle_step, not on
which observations are requested. Off-grid observations use a temporary branch
from the preceding anchor; they do not change the canonical later trajectory.
No candidate ever reuses another candidate's material history.
"""
from collections import OrderedDict
from dataclasses import replace
from hashlib import sha256
import json
from weakref import WeakKeyDictionary
import numpy as np

from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.core.validation import ContractError, scalar
from fatigue_dic_rl.mechanics import MaterialAssignment
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.observations.registration import RegistrationError
from fatigue_dic_rl.fatigue import (MaterialHistoryState, FatigueEvolutionSolver, FatigueStepRequest,
    FatigueLoadCycle, FatigueLawParameters, SyntheticFatigueDamageLaw, FatigueSolverConfig)
from .belief_state import ObservationBatch, ObservationDesign, PredictedObservation, EstimateRole


class PhysicsForwardAdapter(StaticForwardAdapter):
    """Retain the Phase 1C likelihood, mapping, masking and optimizer interface."""
    nuisance_names = StaticForwardAdapter.nuisance_names | {"A"}
    nuisance_units = dict(StaticForwardAdapter.nuisance_units, A="cycle^-1")

    def __init__(self, predictor, batch):
        if not isinstance(batch, ObservationBatch):
            raise ContractError("physics likelihood requires ObservationBatch")
        predictor.validate_noise_contract(batch.noise_contract)
        self.predictor = predictor
        super().__init__(predictor.damage_model, predictor.parameter_space,
                         [f.case for f in batch.frames], mechanics=predictor.mechanics)
        self.designs = tuple(replace(f.design, pattern=c.pattern) for f, c in zip(batch.frames, self.cases))
        for design in self.designs:
            predictor.validate_design(design)

    def predict(self, normalized):
        z = np.asarray(normalized, dtype=float)
        self.parameter_space.decode(z)
        key = z.tobytes()
        if key not in self._cache:
            states = self.predictor.damage_states(z, [d.cycle_count for d in self.designs])
            result = []
            for d, mapping in zip(self.designs, self.mappings):
                predicted = self.predictor.observe(z, d, states[d.cycle_count], mapping=mapping)
                result.append((predicted.displacement_mm, predicted.reaction_n))
            self.evaluation_count += 1
            self._cache[key] = tuple(result)
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
        return self._cache[key]


class PhysicsPredictor:
    def __init__(self, damage_model, parameter_space, pattern, *, load_history=(), cycle_step=2.,
                 reference_energy_n_mm2=.025, law_exponent=1., law_threshold=0., law_beta_R=0.,
                 mechanics=None, solver_config=None):
        self.damage_model, self.parameter_space = damage_model, parameter_space
        self.mechanics = mechanics or Tri3MechanicsSolver()
        self.solver_config = solver_config or FatigueSolverConfig(max_delta_damage_per_step=.03)
        self.pattern = self.public_pattern(pattern)
        self.cycle_step = scalar(cycle_step, "cycle_step", minimum=0)
        self.reference_energy_n_mm2 = scalar(reference_energy_n_mm2, "reference energy", minimum=0)
        self.law_exponent = scalar(law_exponent, "fatigue m", minimum=0)
        self.law_threshold = scalar(law_threshold, "fatigue eta", minimum=0)
        self.law_beta_R = scalar(law_beta_R, "fatigue beta_R")
        if min(self.cycle_step, self.reference_energy_n_mm2, self.law_exponent) <= 0:
            raise ContractError("time step, energy reference and fatigue exponent must be positive")
        self.load_history = tuple(load_history)
        if any(not isinstance(c, FatigueLoadCycle) for c in self.load_history):
            raise ContractError("load history requires FatigueLoadCycle segments")
        if self.load_history and (self.load_history[0].N_start != 0 or
                any(a.N_end != b.N_start for a, b in zip(self.load_history, self.load_history[1:]))):
            raise ContractError("fatigue load segments must be contiguous starting at N=0")
        supported = set(parameter_space.names) | set(parameter_space.fixed)
        if self.load_history and "A" not in supported:
            raise ContractError("fatigue prediction must declare known or unknown A")
        if not self.load_history and "A" in parameter_space.names:
            raise ContractError("unknown A requires a fatigue loading history")
        if any(p.name == "A" and p.lower < 0 for p in parameter_space.parameters) or parameter_space.fixed.get("A", 0.) < 0:
            raise ContractError("A must be nonnegative cycle^-1")
        if set(p.name for p in damage_model.parameters)-supported:
            raise ContractError("initial damage parameters must be declared known or unknown")
        # Reuse Phase 1C legal parameter boxes/units and public-pattern checks.
        x = self.pattern.mesh.nodes[:1]
        from fatigue_dic_rl.observations.covariance import DisplacementCovariance
        d = ObservationDesign("public-validation", self.pattern, 0., x, np.ones(1, dtype=bool),
                              DisplacementCovariance.diagonal(np.ones((1, 2))))
        StaticForwardAdapter.__init__(object.__new__(PhysicsForwardAdapter), damage_model,
                                      parameter_space, [d.empty_case()], mechanics=self.mechanics)
        self._history_cache = OrderedDict()
        self._history_candidate_keys = WeakKeyDictionary()
        self.rollout_count = 0
        self.mechanics_observation_count = 0
        self.failure_diagnostics = []
        self.model_id = sha256(json.dumps(self.provenance(), sort_keys=True).encode()).hexdigest()

    def public_pattern(self, pattern):
        if np.any(pattern.damage.values != 0) or pattern.internal_variables or pattern.cycle_count != 0:
            raise ContractError("public model must contain zero damage, no hidden history and N=0")
        if "E0" in self.parameter_space.names:
            spec = next(p for p in self.parameter_space.parameters if p.name == "E0")
            materials = {r: MaterialAssignment(a.constitutive_model,
                dict(a.parameters, young_modulus_n_per_mm2=(spec.lower+spec.upper)/2))
                for r, a in pattern.materials.items()}
            pattern = replace(pattern, materials=materials)
        return pattern

    def provenance(self):
        p = self.pattern
        return {"damage_model": type(self.damage_model).__name__, "damage_max": self.damage_model.damage_max,
            "parameter_box": [{"name": s.name, "lower": s.lower, "upper": s.upper, "units": s.units}
                              for s in self.parameter_space.parameters], "fixed": dict(self.parameter_space.fixed),
            "mesh_id": p.mesh.mesh_id, "mesh_nodes_sha256": sha256(p.mesh.nodes.tobytes()).hexdigest(),
            "connectivity_sha256": sha256(np.asarray(p.mesh.connectivity).tobytes()).hexdigest(),
            "material_regions_sha256": sha256(p.mesh.material_region_ids.tobytes()).hexdigest(),
            "quadrature_coordinates_sha256": sha256(p.quadrature_layout.coordinates.tobytes()).hexdigest(),
            "materials": {str(r): {"model": type(a.constitutive_model).__name__, "parameters": dict(a.parameters)}
                          for r, a in p.materials.items()},
            "thickness_mm": p.thickness_mm,
            "boundary_conditions": {n: [{"nodes": getattr(b, "node_ids", np.empty(0)).tolist(),
                "components": getattr(b, "components", np.empty(0)).tolist(),
                "boundary_tag": getattr(b, "boundary_tag", None), "values": b.values.tolist()}
                for b in getattr(p, n)] for n in ("displacement_bcs", "nodal_force_bcs", "traction_bcs")},
            "cycle_step": self.cycle_step, "reference_energy_n_mm2": self.reference_energy_n_mm2,
            "law_exponent": self.law_exponent, "law_threshold": self.law_threshold, "law_beta_R": self.law_beta_R,
            "load_history": [{n: getattr(c, n) for n in ("N_start", "delta_N", "load_max", "load_min",
                              "nominal_stress_ratio", "load_id", "loading_mode")} for c in self.load_history],
            "fatigue_config": dict(vars(self.solver_config)),
            "mechanics_config": dict(vars(self.mechanics.config)),
            "integration": "canonical anchors with read-only off-grid branches"}

    def validate_noise_contract(self, contract):
        if contract == "independent_given_shared_registration":
            names = set(self.parameter_space.names) | set(self.parameter_space.fixed)
            if not {"tx", "ty", "theta"} <= names:
                raise ContractError("shared registration contract requires shared tx, ty, theta parameters")

    def validate_design(self, design):
        if not isinstance(design, ObservationDesign):
            raise ContractError("prediction accepts observation designs, never measured frames")
        p = self.public_pattern(design.pattern)
        if p.mesh is not self.pattern.mesh:
            raise ContractError("predictor supports one fixed mesh; history transfer is unsupported")
        a, b = p.quadrature_layout, self.pattern.quadrature_layout
        if not np.array_equal(a.coordinates, b.coordinates) or not np.array_equal(a.material_state_indices, b.material_state_indices):
            raise ContractError("diagnostic quadrature support differs from material history")
        if self.load_history and design.cycle_count > self.load_history[-1].N_end:
            raise ContractError("requested cycle exceeds public load history")
        if design.reaction_node_ids is not None and np.any(design.reaction_node_ids >= p.mesh.n_nodes):
            raise ContractError("reaction observation node lies outside the FE mesh")

    def _candidate_pattern(self, pattern, physical):
        materials = pattern.materials
        if "E0" in physical:
            materials = {r: MaterialAssignment(a.constitutive_model,
                dict(a.parameters, young_modulus_n_per_mm2=physical["E0"])) for r, a in materials.items()}
        factor = physical.get("load_factor", 1.)
        if factor <= 0:
            raise ContractError("loading amplitude must be positive")
        return replace(pattern, materials=materials,
            traction_bcs=tuple(replace(b, values=b.values*factor) for b in pattern.traction_bcs),
            nodal_force_bcs=tuple(replace(b, values=b.values*factor) for b in pattern.nodal_force_bcs))

    def damage_states(self, normalized, cycles):
        z = np.asarray(normalized, dtype=float)
        physical = self.parameter_space.decode(z)
        times = tuple(sorted(set(scalar(c, "target cycle", minimum=0) for c in cycles)))
        if not times:
            raise ContractError("physics rollout requires at least one target")
        if self.load_history and times[-1] > self.load_history[-1].N_end:
            raise ContractError("public loading history does not cover target cycle")
        key = (z.tobytes(), times)
        if key in self._history_cache:
            self._history_cache.move_to_end(key)
            return self._history_cache[key].copy()
        layout = self.pattern.quadrature_layout
        initial_damage = layout.field(self.damage_model.evaluate(layout.coordinates, physical), "1")
        initial = MaterialHistoryState.initialize(layout, initial_damage, damage_max=self.damage_model.damage_max)
        if not self.load_history:
            result = {t: MaterialHistoryState.initialize(layout, initial_damage, damage_max=self.damage_model.damage_max,
                                                        cycle_count=t) for t in times}
        else:
            pattern = self._candidate_pattern(self.pattern, physical)
            law = SyntheticFatigueDamageLaw({int(r): FatigueLawParameters(physical["A"], self.reference_energy_n_mm2,
                self.law_threshold, self.law_exponent, self.law_beta_R) for r in np.unique(initial.material_ids)})
            solver = FatigueEvolutionSolver(pattern, initial, law, mechanics=self.mechanics, config=self.solver_config)
            result = {}
            try:
                for target in times:
                    while solver.committed_state.cycle_count < target:
                        n = solver.committed_state.cycle_count
                        segment = next(c for c in self.load_history if c.N_start <= n < c.N_end)
                        step_index = int(np.floor((n-segment.N_start)/self.cycle_step+1e-12))+1
                        endpoint = min(segment.N_end, segment.N_start+step_index*self.cycle_step)
                        if endpoint <= n:
                            raise ContractError("canonical time step cannot advance representably")
                        if endpoint <= target:
                            solver.advance(FatigueStepRequest(replace(segment, N_start=n, delta_N=endpoint-n)))
                        else:
                            # Off-grid queries cannot commit into the canonical path.
                            branch = FatigueEvolutionSolver(pattern, solver.committed_state, law,
                                                           mechanics=self.mechanics, config=self.solver_config)
                            branch.advance(FatigueStepRequest(replace(segment, N_start=n, delta_N=target-n)))
                            result[target] = branch.committed_state
                            break
                    if target not in result:
                        result[target] = solver.committed_state
            except Exception as exc:
                self.failure_diagnostics.append({"parameters": z.tolist(), "type": type(exc).__name__,
                    "message": str(exc), "committed_cycle_before_failure": solver.committed_state.cycle_count})
                raise
        self.rollout_count += 1
        for state in result.values():
            self._history_candidate_keys[state] = z.tobytes()
        self._history_cache[key] = result
        if len(self._history_cache) > 64:
            self._history_cache.popitem(last=False)
        return result.copy()

    def observe(self, normalized, design, state=None, *, mapping=None, role=EstimateRole.PREDICTION,
                conditioned_observation_ids=()):
        self.validate_design(design)
        p = self.parameter_space.decode(normalized)
        state = self.damage_states(normalized, [design.cycle_count])[design.cycle_count] if state is None else state
        if state.cycle_count != design.cycle_count or state.layout is not self.pattern.quadrature_layout:
            raise ContractError("observation state has wrong cycle or history layout")
        if self._history_candidate_keys.get(state) != np.asarray(normalized, dtype=float).tobytes():
            raise ContractError("material history does not belong to this parameter candidate")
        pattern = self._candidate_pattern(self.public_pattern(design.pattern), p)
        layout = pattern.quadrature_layout
        values = np.zeros(layout.count) if design.known_healthy else state.damage.values
        # Always solve using the newly propagated damage at this target cycle.
        request = replace(pattern, damage=layout.field(values, "1"), cycle_count=design.cycle_count,
                          loading_state=LoadingState(pattern.loading_state.load_id, design.loading_phase))
        response = self.mechanics.solve(request)
        self.mechanics_observation_count += 1
        if mapping is None:
            from fatigue_dic_rl.observations.operator import ObservationOperator
            mapping = ObservationOperator(pattern.mesh).prepare(design.coordinates_mm, coordinate_units="mm",
                                                               coordinate_system=pattern.mesh.coordinate_system)
        if np.any(design.active_mask & ~mapping.valid_mask):
            raise ContractError("active observation lies outside FEM support")
        u = mapping.apply(response.state.displacement).values.copy()
        if any(name in p for name in ("tx", "ty", "theta")):
            u += RegistrationError([p.get("tx", 0.), p.get("ty", 0.)], p.get("theta", 0.)).field(design.coordinates_mm)
        r = None if design.reaction_node_ids is None else float(
            response.reaction_forces.values[design.reaction_node_ids, design.reaction_component].sum())
        return PredictedObservation(design.observation_id, design.cycle_count, design.coordinates_mm, u,
            design.active_mask, r, role, tuple(conditioned_observation_ids))
