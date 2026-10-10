"""Synthetic truth and displacement observations have separate persistence APIs."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.mechanics.tri3 import Tri3SolverResult
from fatigue_dic_rl.observations import DICObservation
from .history import HistoryStatus, MaterialHistoryState
from .loading import FatigueLoadCycle
from .models import FatigueDrivingForceResult, SyntheticFatigueDamageLaw
from .solver import FatigueEvolutionSolver, FatigueStepRequest, FatigueStepResult


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def cycle_metadata(cycle: FatigueLoadCycle) -> dict:
    return {name: _plain(getattr(cycle, name)) for name in cycle.__dataclass_fields__}


def save_observations(observations: tuple[DICObservation, ...], folder: Path) -> None:
    """This interface accepts only DIC observations, never material history."""
    if not observations or not all(isinstance(o, DICObservation) for o in observations):
        raise ContractError("save_observations requires nonempty DIC observations")
    first = observations[0]
    if any(not np.array_equal(o.coordinates, first.coordinates) or
           o.coordinate_units != first.coordinate_units or o.displacement_units != first.displacement_units or
           o.coordinate_system != first.coordinate_system for o in observations):
        raise ContractError("series requires one fixed observation coordinate layout and units")
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/"displacement.npz", N=np.array([o.cycle_count for o in observations]),
                        coordinates_mm=first.coordinates,
                        displacement_mm=np.array([o.displacement for o in observations]),
                        valid_mask=np.array([o.valid_mask for o in observations]),
                        confidence=np.array([o.confidence for o in observations]))
    (folder/"metadata.json").write_text(json.dumps({"schema": "dic-displacement-series-v1",
        "source": "synthetic_fe_displacement", "image_correlation_performed": False,
        "coordinate_units": first.coordinate_units, "displacement_units": first.displacement_units,
        "coordinate_system": first.coordinate_system,
        "loading_phase": [o.loading_phase for o in observations],
        "loading_state_ref": [o.loading_state_ref for o in observations]}, indent=2)+'\n')


def load_observations(folder: Path) -> tuple[DICObservation, ...]:
    """Load only the observation directory; no truth path is accepted or read."""
    meta = json.loads((folder/"metadata.json").read_text())
    with np.load(folder/"displacement.npz", allow_pickle=False) as data:
        return tuple(DICObservation(data["coordinates_mm"], u, mask, confidence, N,
                     meta["loading_phase"][i], meta["loading_state_ref"][i], meta["coordinate_system"],
                     meta["coordinate_units"], meta["displacement_units"],
                     metadata={"source": meta["source"]}) for i, (N, u, mask, confidence) in enumerate(
                     zip(data["N"], data["displacement_mm"], data["valid_mask"], data["confidence"], strict=True)))


@dataclass(frozen=True, eq=False)
class FatigueTrajectory:
    initial_state: MaterialHistoryState
    initial_minimum: Tri3SolverResult
    initial_maximum: Tri3SolverResult
    initial_driving: FatigueDrivingForceResult
    initial_hazard: FieldData
    initial_observation: DICObservation | None
    steps: tuple[FatigueStepResult, ...]
    # Store provenance as JSON bytes: nested metadata cannot mutate a stored trajectory.
    provenance_json: str

    def __post_init__(self) -> None:
        if self.initial_state.status != HistoryStatus.COMMITTED:
            raise ContractError("trajectory initial history must be committed")
        self.initial_state.layout.assert_field(self.initial_hazard)
        self.initial_hazard.require(units="cycle^-1", components=(), fully_valid=True)
        if np.any(self.initial_hazard.values < 0):
            raise ContractError("trajectory initial hazard must be nonnegative")
        previous = self.initial_state
        steps = tuple(self.steps)
        if not steps:
            raise ContractError("trajectory requires committed steps")
        for step in steps:
            if (step.request.cycle.N_start != previous.cycle_count or not step.substeps or
                    step.committed_state.status != HistoryStatus.COMMITTED):
                raise ContractError("trajectory steps must be complete, committed and contiguous")
            for row in step.substeps:
                if (row.state_after.status != HistoryStatus.COMMITTED or
                        row.state_before.cycle_count != previous.cycle_count or
                        row.cycle.N_start != previous.cycle_count or row.state_after.cycle_count != row.cycle.N_end):
                    raise ContractError("trajectory substeps are inconsistent")
                previous.damage.assert_compatible(row.state_before.damage)
                if not np.array_equal(previous.damage.values, row.state_before.damage.values):
                    raise ContractError("trajectory damage history is discontinuous")
                previous = row.state_after
            if (step.committed_state.cycle_count != step.request.cycle.N_end or
                    not np.array_equal(previous.damage.values, step.committed_state.damage.values)):
                raise ContractError("outer-step commit disagrees with accepted substeps")
        for response in (self.initial_minimum, self.initial_maximum):
            if (not isinstance(response, Tri3SolverResult) or
                    response.state.cycle_count != self.initial_state.cycle_count or
                    not np.array_equal(response.state.damage.values, self.initial_state.damage.values)):
                raise ContractError("trajectory initial response must use the initial history")
        provenance = json.loads(self.provenance_json)
        if not isinstance(provenance, dict) or "random_seed" not in provenance or not provenance.get("code_version"):
            raise ContractError("trajectory requires seed and code provenance")
        object.__setattr__(self, "steps", steps)

    @property
    def substeps(self):
        return tuple(row for step in self.steps for row in step.substeps)

    @property
    def observations(self) -> tuple[DICObservation, ...]:
        rows = (self.initial_observation, *(s.observation for s in self.substeps))
        return tuple(o for o in rows if o is not None)

    def save(self, folder: Path) -> None:
        """Publish a complete trajectory by one rename; refuse existing outputs."""
        folder = Path(folder)
        if folder.exists():
            raise FileExistsError(folder)
        folder.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{folder.name}-trial-", dir=folder.parent))
        try:
            self._save_truth(temp/"truth")
            if self.observations:
                save_observations(self.observations, temp/"observations")
            (temp/"provenance.json").write_text(self.provenance_json+'\n')
            temp.rename(folder)
        except Exception:
            shutil.rmtree(temp)
            raise

    def _save_truth(self, folder: Path) -> None:
        folder.mkdir(exist_ok=False)
        rows = self.substeps
        states = (self.initial_state, *(r.state_after for r in rows))
        maxima = (self.initial_maximum, *(r.maximum for r in rows))
        minima = (self.initial_minimum, *(r.minimum for r in rows))
        drivers = (self.initial_driving, *(r.driving_end for r in rows))
        hazards = (self.initial_hazard, *(r.hazard_end for r in rows))
        layout, mesh = self.initial_state.layout, self.initial_state.layout.mesh
        payload = {"N": np.array([s.cycle_count for s in states]),
                   "damage": np.array([s.damage.values for s in states]),
                   "accumulated_damage_increment": np.array([s.accumulated_damage_increment.values for s in states]),
                   "accumulated_cycles": np.array([s.accumulated_cycles.values for s in states]),
                   "history": np.array([s.history.values for s in states]),
                   "nodes_mm": mesh.nodes, "connectivity": np.array(mesh.connectivity),
                   "material_region_ids": mesh.material_region_ids, "material_ids": self.initial_state.material_ids,
                   "quadrature_ids": self.initial_state.quadrature_ids,
                   "material_state_indices": layout.material_state_indices,
                   "quadrature_to_element": layout.quadrature_to_element,
                   "quadrature_coordinates_mm": layout.coordinates, "reference_points": layout.reference_points,
                   "reference_weights": layout.reference_weights, "physical_weights_mm2": layout.physical_weights,
                   "jacobians": layout.jacobians, "signed_jacobian_determinants": layout.signed_jacobian_determinants,
                   "strain_amplitude": np.array([d.strain_amplitude.values for d in drivers]),
                   "strain_mean": np.array([d.strain_mean.values for d in drivers]),
                   "Ya_endpoint_n_mm2": np.array([d.energy_amplitude.values for d in drivers]),
                   "Y_max_endpoint_n_mm2": np.array([d.Y_max.values for d in drivers]),
                   "Y_min_endpoint_n_mm2": np.array([d.Y_min.values for d in drivers]),
                   "hazard_endpoint_per_cycle": np.array([h.values for h in hazards]),
                   "N_start_used": np.array([r.cycle.N_start for r in rows]),
                   "delta_N_used": np.array([r.cycle.delta_N for r in rows]),
                   "Ya_frozen_used_n_mm2": np.array([r.driving_used.energy_amplitude.values for r in rows]),
                   "hazard_frozen_used_per_cycle": np.array([r.hazard_used.values for r in rows]),
                   "Y_max_frozen_used_n_mm2": np.array([r.driving_used.Y_max.values for r in rows]),
                   "Y_min_frozen_used_n_mm2": np.array([r.driving_used.Y_min.values for r in rows]),
                   "damage_increment": np.array([r.damage_increment.values for r in rows])}
        for phase, responses in (("max", maxima), ("min", minima)):
            for name, getter in (("displacement_mm", lambda r: r.state.displacement.values),
                                 ("strain_engineering", lambda r: r.state.strain.values),
                                 ("stress_n_mm2", lambda r: r.state.stress.values),
                                 ("reaction_n", lambda r: r.reaction_forces.values),
                                 ("applied_force_n", lambda r: r.applied_forces.values),
                                 ("equilibrium_residual_n", lambda r: r.equilibrium_residual.values),
                                 ("elastic_energy_n_mm", lambda r: r.elastic_strain_energy),
                                 ("constrained_dofs", lambda r: r.constrained_dofs)):
                payload[f"{phase}_{name}"] = np.array([getter(r) for r in responses])
        np.savez_compressed(folder/"trajectory.npz", **payload)
        diagnostics = {"steps": [dict(r.diagnostics) for r in rows],
                       "max_solver": [asdict(r.diagnostics) for r in maxima],
                       "min_solver": [asdict(r.diagnostics) for r in minima],
                       "rejected_attempts": [asdict(a) for s in self.steps for a in s.rejected_attempts]}
        (folder/"diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+'\n')
        (folder/"metadata.json").write_text(json.dumps({"schema": "synthetic-fatigue-truth-v1",
            "mesh_id": mesh.mesh_id, "quadrature_reference": layout.reference.space_id,
            "quadrature_rule": layout.rule_name, "history_names": list(self.initial_state.history_names),
            "damage_max": self.initial_state.damage_max,
            "units": {"N": "cycle", "damage": "1", "history": "1", "length": "mm", "strain": "1",
                      "strain_shear": "engineering gamma_xy", "stress": "N/mm^2", "Ya": "N/mm^2",
                      "hazard": "cycle^-1", "reaction": "N", "energy": "N*mm"},
            "energy_semantics": "Ymax(start)*deltaD integrated at maximum phase; not total fatigue loss",
            "source": "physics_only_synthetic"}, indent=2)+'\n')


def run_trajectory(solver: FatigueEvolutionSolver, cycles: tuple[FatigueLoadCycle, ...], *,
                   random_seed: int, code_version: str) -> FatigueTrajectory:
    if not cycles:
        raise ContractError("trajectory requires at least one cycle increment")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or not isinstance(code_version, str) or not code_version:
        raise ContractError("trajectory needs an explicit integer seed and code version")
    # Validate the whole requested load history before advancing any interval.
    expected = solver.committed_state.cycle_count
    for cycle in cycles:
        if not isinstance(cycle, FatigueLoadCycle) or cycle.N_start != expected:
            raise ContractError("load history must be contiguous and begin at committed N")
        cycle.request(solver.pattern, solver.committed_state.damage, cycle.N_start, "max")
        expected = cycle.N_end
    initial = solver.committed_state
    lo, hi, driving = solver.evaluate_committed(cycles[0])
    hazard = solver.damage_law.hazard(driving, cycles[0], initial)
    observation = solver._observation(hi)
    steps = tuple(solver.advance(FatigueStepRequest(cycle)) for cycle in cycles)
    pattern = solver.pattern
    bcs = {name: [{k: _plain(getattr(bc, k)) for k in bc.__dataclass_fields__}
                  for bc in getattr(pattern, name)]
           for name in ("displacement_bcs", "nodal_force_bcs", "traction_bcs")}
    law = solver.damage_law
    law_parameters = {str(k): asdict(v) for k, v in law.parameters.items()} if isinstance(law, SyntheticFatigueDamageLaw) else {
        "hazard_per_cycle": getattr(law, "hazard_per_cycle", None)}
    provenance = {"random_seed": random_seed, "code_version": code_version,
        "solver_configuration": asdict(solver.config), "mechanics_configuration": asdict(solver.mechanics.config),
        "damage_law": type(law).__name__, "damage_law_parameters": law_parameters,
        "driving_force": type(solver.driving_force).__name__, "thickness_mm": pattern.thickness_mm,
        "constitutive_parameters": {str(k): {"model": type(v.constitutive_model).__name__,
                                    "parameters": dict(v.parameters)} for k, v in pattern.materials.items()},
        "mesh_id": pattern.mesh.mesh_id, "quadrature_reference": initial.layout.reference.space_id,
        "coordinate_units": pattern.mesh.coordinate_units, "coordinate_system": pattern.mesh.coordinate_system,
        "boundary_tags": {k: {"node_ids": v.node_ids.tolist(), "facets": v.facets.tolist()}
                          for k, v in pattern.mesh.boundary_tags.items()},
        "boundary_conditions": bcs,
        "load_history": [cycle_metadata(c) for c in cycles], "initial_N_cycle": initial.cycle_count,
        "observations": "displacement-only synthetic; image_correlation_performed=False",
        "method": "first-order frozen-hazard staggering; not exact nonlinear coupled evolution"}
    return FatigueTrajectory(initial, lo, hi, driving, hazard, observation, steps,
                             json.dumps(_plain(provenance), indent=2, allow_nan=False))
