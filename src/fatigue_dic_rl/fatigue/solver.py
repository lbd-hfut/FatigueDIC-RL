"""First-order mechanics/damage staggering with atomic outer-step rollback."""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError, scalar
from fatigue_dic_rl.mechanics import MechanicsConvergenceError, SolverDiagnostics
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver, Tri3SolverResult
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverRequest
from fatigue_dic_rl.observations import DICObservation, ObservationMapping
from fatigue_dic_rl.physics import DamageTransition
from .history import HistoryStatus, MaterialHistoryState
from .loading import FatigueLoadCycle
from .models import EnergyAmplitudeDriver, FatigueDamageLaw, FatigueDrivingForce, FatigueDrivingForceResult


@dataclass(frozen=True)
class FatigueSolverConfig:
    max_delta_damage_per_step: float = 0.03
    max_step_subdivisions: int = 8
    minimum_cycle_increment: float = 1e-8
    adaptive_subdivision: bool = True
    energy_identity_rtol: float = 2e-10

    def __post_init__(self) -> None:
        for name in ("max_delta_damage_per_step", "minimum_cycle_increment", "energy_identity_rtol"):
            value = scalar(getattr(self, name), name, minimum=0)
            if value <= 0:
                raise ContractError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.max_delta_damage_per_step >= 1 or self.energy_identity_rtol > 1e-6:
            raise ContractError("inadmissible damage limit or energy tolerance")
        if (isinstance(self.max_step_subdivisions, bool) or not isinstance(self.max_step_subdivisions, int) or
                not 0 <= self.max_step_subdivisions <= 16 or not isinstance(self.adaptive_subdivision, bool)):
            raise ContractError("subdivision depth must be an integer in [0,16] and adaptive flag a bool")


@dataclass(frozen=True)
class FatigueStepRequest:
    cycle: FatigueLoadCycle

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, FatigueLoadCycle):
            raise ContractError("step requires a FatigueLoadCycle")


@dataclass(frozen=True)
class FatigueFailureDiagnostic:
    N_start: float
    delta_N: float
    subdivision_depth: int
    stage: str
    exception_type: str
    message: str
    mechanics_diagnostics: SolverDiagnostics | None = None


class FatigueStepError(RuntimeError):
    def __init__(self, failures: tuple[FatigueFailureDiagnostic, ...]) -> None:
        self.failures = failures
        last = failures[-1]
        super().__init__(f"fatigue step rolled back at {last.stage}: {last.message}")


class _DamageStepTooLarge(ContractError):
    pass


@dataclass(frozen=True, eq=False)
class FatigueSubstepResult:
    cycle: FatigueLoadCycle
    state_before: MaterialHistoryState
    state_after: MaterialHistoryState
    minimum: Tri3SolverResult
    maximum: Tri3SolverResult
    driving_used: FatigueDrivingForceResult
    driving_end: FatigueDrivingForceResult
    hazard_used: FieldData
    hazard_end: FieldData
    damage_increment: FieldData
    diagnostics: Mapping[str, float]
    observation: DICObservation | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True, eq=False)
class FatigueStepResult:
    request: FatigueStepRequest
    committed_state: MaterialHistoryState
    substeps: tuple[FatigueSubstepResult, ...]
    rejected_attempts: tuple[FatigueFailureDiagnostic, ...] = ()


def equilibrium_diagnostics(result: Tri3SolverResult, rtol: float) -> dict[str, float]:
    """Algebraic work identity, not accumulated work over a loading cycle."""
    u = result.state.displacement.values.ravel()
    f, r = result.applied_forces.values, result.reaction_forces.values
    U = result.elastic_strain_energy
    matrix = float(0.5*u @ (result.assembly.stiffness @ u))
    force_work, reaction_work = float(u @ f.ravel()), float(u @ r.ravel())
    scale = max(abs(matrix), abs(U), abs(force_work)+abs(reaction_work), 1e-12)
    gap = abs(2*U-force_work-reaction_work)
    if abs(matrix-U) > rtol*scale or gap > rtol*scale:
        raise ContractError("elastic energy / reaction-inclusive work identity failed")
    total = f+r
    nodes = result.assembly.quadrature_layout.mesh.nodes
    return {"elastic_energy_n_mm": U, "matrix_energy_n_mm": matrix,
            "force_displacement_product_n_mm": force_work,
            "reaction_displacement_product_n_mm": reaction_work,
            "work_identity_gap_n_mm": gap,
            "force_balance_x_n": float(total[:, 0].sum()),
            "force_balance_y_n": float(total[:, 1].sum()),
            "moment_balance_n_mm": float(np.sum(nodes[:, 0]*total[:, 1]-nodes[:, 1]*total[:, 0])),
            "solver_residual_n": result.diagnostics.residual_norm}


class FatigueEvolutionSolver:
    """Only advance() changes the committed history, after all substeps succeed.

    Subdivision limits damage change; it is not an embedded time-error estimator.
    Mechanics failures and illegal/nonfinite states reject the whole requested
    interval. Passive optional dimensionless history is retained, never passed
    to the stateless elastic constitutive model or interpolated across meshes.
    """

    def __init__(self, pattern: Tri3SolverRequest, initial_state: MaterialHistoryState,
                 damage_law: FatigueDamageLaw, *, mechanics: Tri3MechanicsSolver | None = None,
                 driving_force: FatigueDrivingForce | None = None,
                 config: FatigueSolverConfig | None = None,
                 observation_mapping: ObservationMapping | None = None) -> None:
        if not isinstance(pattern, Tri3SolverRequest) or not isinstance(initial_state, MaterialHistoryState):
            raise ContractError("fatigue solver requires an explicit Tri3 pattern and material history")
        if initial_state.status != HistoryStatus.COMMITTED:
            raise ContractError("solver must start from committed material history")
        # Exact object binding prevents same mesh_id/array-length aliases from replacing history support.
        if initial_state.layout is not pattern.quadrature_layout:
            raise ContractError("history and mechanics must share the exact QuadratureLayout object")
        if initial_state.layout.mesh is not pattern.mesh:
            raise ContractError("quadrature material ownership must use the exact pattern mesh object")
        if not isinstance(damage_law, FatigueDamageLaw):
            raise ContractError("fatigue solver needs a FatigueDamageLaw")
        self.pattern = pattern
        self.damage_law = damage_law
        self.mechanics = Tri3MechanicsSolver() if mechanics is None else mechanics
        self.driving_force = EnergyAmplitudeDriver() if driving_force is None else driving_force
        self.config = FatigueSolverConfig() if config is None else config
        if (not isinstance(self.mechanics, Tri3MechanicsSolver) or
                not isinstance(self.driving_force, FatigueDrivingForce) or
                not isinstance(self.config, FatigueSolverConfig)):
            raise ContractError("invalid mechanics, driver or fatigue solver configuration")
        if initial_state.damage_max > self.mechanics.config.damage_max:
            raise ContractError("history damage cap exceeds mechanics admissibility cap")
        if observation_mapping is not None and observation_mapping.mesh is not pattern.mesh:
            raise ContractError("observation mapping must use the pattern mesh object")
        self.observation_mapping = observation_mapping
        self.transition = DamageTransition(initial_state.damage_max)
        self._committed_state = initial_state
        self._failure_log: list[FatigueFailureDiagnostic] = []

    @property
    def committed_state(self) -> MaterialHistoryState:
        return self._committed_state

    @property
    def failure_log(self) -> tuple[FatigueFailureDiagnostic, ...]:
        return tuple(self._failure_log)

    def _endpoints(self, state, cycle, N):
        if state.layout is not self.pattern.quadrature_layout or state.layout.mesh is not self.pattern.mesh:
            raise ContractError("history/layout/mesh binding changed; state transfer is unsupported")
        minimum = self.mechanics.solve(cycle.request(self.pattern, state.damage, N, "min"))
        maximum = self.mechanics.solve(cycle.request(self.pattern, state.damage, N, "max"))
        return minimum, maximum, self.driving_force.evaluate(minimum, maximum)

    def evaluate_committed(self, cycle: FatigueLoadCycle):
        if cycle.N_start != self.committed_state.cycle_count:
            raise ContractError("cycle N_start must equal committed N")
        return self._endpoints(self.committed_state, cycle, cycle.N_start)

    def _observation(self, result):
        if self.observation_mapping is None:
            return None
        mapped = self.observation_mapping.apply(result.state.displacement)
        return DICObservation(self.observation_mapping.coordinates, mapped.values, mapped.valid_mask,
                              mapped.valid_mask.astype(float), result.state.cycle_count, "max",
                              result.state.loading_state.load_id, self.pattern.mesh.coordinate_system,
                              "mm", "mm", metadata={"source": "synthetic_fe_displacement"})

    def _attempt(self, state, cycle, depth, attempts):
        stage = "validate_increment"
        try:
            if cycle.delta_N < self.config.minimum_cycle_increment:
                raise ContractError("delta_N below minimum_cycle_increment")
            if cycle.N_start != state.cycle_count:
                raise ContractError("cycle N_start does not match the current history")
            stage = "start_mechanics"
            lo, hi, driving = self._endpoints(state, cycle, cycle.N_start)
            stage = "hazard"
            hazard = self.damage_law.hazard(driving, cycle, state)
            stage = "damage_transition"
            candidate = self.transition.apply(state.damage, hazard, cycle.delta_N)
            increment = candidate.values-state.damage.values
            maximum_increment = float(increment.max())
            if maximum_increment > self.config.max_delta_damage_per_step:
                raise _DamageStepTooLarge(f"max delta D={maximum_increment:.6g} exceeds "
                                          f"{self.config.max_delta_damage_per_step:.6g}")
            stage = "trial_history"
            trial = state.trial(candidate, cycle.delta_N)
            stage = "end_mechanics"
            lo_new, hi_new, driving_new = self._endpoints(trial, cycle, cycle.N_end)
            stage = "endpoint_hazard"
            hazard_new = self.damage_law.hazard(driving_new, cycle, trial)
            stage = "energy_diagnostics"
            diagnostics = {"delta_N_cycle": cycle.delta_N, "max_delta_damage": maximum_increment,
                           "subdivision_depth": float(depth)}
            weights = state.layout.physical_weights*self.pattern.thickness_mm
            for phase, before, after, Y in (("min", lo, lo_new, driving.Y_min),
                                           ("max", hi, hi_new, driving.Y_max)):
                for endpoint, response in (("start", before), ("end", after)):
                    diagnostics.update({f"{phase}_{endpoint}_{k}": v for k, v in
                                        equilibrium_diagnostics(response, self.config.energy_identity_rtol).items()})
                dissipation = float(np.dot(weights*Y.values, increment))
                u_delta = after.state.displacement.values-before.state.displacement.values
                load_average = 0.5*(before.applied_forces.values+before.reaction_forces.values+
                                    after.applied_forces.values+after.reaction_forces.values)
                work = float(np.sum(load_average*u_delta))
                delta_U = after.elastic_strain_energy-before.elastic_strain_energy
                diagnostics.update({f"{phase}_damage_dissipation_estimate_n_mm": dissipation,
                                    f"{phase}_fixed_phase_work_estimate_n_mm": work,
                                    f"{phase}_elastic_energy_change_n_mm": delta_U,
                                    f"{phase}_fixed_phase_balance_gap_n_mm": work-delta_U-dissipation})
                if dissipation < 0:
                    raise ContractError("negative damage dissipation")
            if not np.isfinite(list(diagnostics.values())).all():
                raise ContractError("nonfinite fatigue diagnostics")
            stage = "observation"
            observation = self._observation(hi_new)
            return trial, (FatigueSubstepResult(cycle, state, trial, lo_new, hi_new, driving, driving_new,
                                               hazard, hazard_new, state.layout.field(increment, "1"),
                                               diagnostics, observation),)
        except Exception as exc:
            diagnostic = FatigueFailureDiagnostic(cycle.N_start, cycle.delta_N, depth, stage,
                type(exc).__name__, str(exc), exc.diagnostics if isinstance(exc, MechanicsConvergenceError) else None)
            attempts.append(diagnostic)
            if isinstance(exc, _DamageStepTooLarge) and self.config.adaptive_subdivision:
                if depth < self.config.max_step_subdivisions and cycle.delta_N/2 >= self.config.minimum_cycle_increment:
                    first, second = cycle.subdivide()
                    middle, rows_a = self._attempt(state, first, depth+1, attempts)
                    final, rows_b = self._attempt(middle, second, depth+1, attempts)
                    return final, rows_a+rows_b
            raise

    def advance(self, request: FatigueStepRequest) -> FatigueStepResult:
        if not isinstance(request, FatigueStepRequest):
            raise ContractError("advance requires FatigueStepRequest")
        attempts: list[FatigueFailureDiagnostic] = []
        try:
            trial, rows = self._attempt(self.committed_state, request.cycle, 0, attempts)
            committed = trial.commit()
            # Publish only complete intervals; intermediate accepted trials become committed records together.
            rows = tuple(replace(row,
                state_before=replace(row.state_before, status=HistoryStatus.COMMITTED),
                state_after=replace(row.state_after, status=HistoryStatus.COMMITTED)) for row in rows)
            result = FatigueStepResult(request, committed, rows, tuple(attempts))
        except Exception as exc:
            if not attempts:
                attempts.append(FatigueFailureDiagnostic(request.cycle.N_start, request.cycle.delta_N, 0,
                                "publish", type(exc).__name__, str(exc)))
            self._failure_log.extend(attempts)
            raise FatigueStepError(tuple(attempts)) from exc
        self._failure_log.extend(attempts)
        self._committed_state = committed
        return result
