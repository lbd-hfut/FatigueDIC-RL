"""Irregular observation and parameter-belief contracts for replaceable estimators.

A MAP parameter vector decoded into damage is a plug-in state estimate, not
the mean of the nonlinear damage posterior. Forecast designs carry no observed
displacement, reaction value, or hidden-state channel.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol
import numpy as np

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError, array, label, scalar
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase, ReactionObservation
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.observations.dic import DICObservation
from fatigue_dic_rl.observations.noise import DICMeasurement


class EstimateRole(str, Enum):
    PRIOR = "prior"
    PREDICTION = "prediction_before_observation"
    FILTERED = "filtered_after_observation"
    FORECAST = "future_open_loop_forecast"


@dataclass(frozen=True, eq=False)
class GaussianPrior:
    """One fixed diagonal Gaussian prior in Phase 1C dimensionless coordinates."""
    normalized_mean: np.ndarray
    normalized_standard_deviation: np.ndarray
    description: str = "declared before observing data"

    def __post_init__(self):
        mean = array(self.normalized_mean)
        std = array(self.normalized_standard_deviation)
        if (mean.ndim != 1 or mean.shape != std.shape or not len(mean)
                or not np.isfinite(mean).all() or np.any(np.abs(mean) > 1)
                or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ContractError("prior needs bounded dimensionless mean and positive [K] scales")
        object.__setattr__(self, "normalized_mean", mean)
        object.__setattr__(self, "normalized_standard_deviation", std)
        label(self.description, "prior description")

    @property
    def precision(self):
        return np.diag(1/self.normalized_standard_deviation**2)


@dataclass(frozen=True, eq=False)
class ObservationDesign:
    observation_id: str
    pattern: object
    cycle_count: float
    coordinates_mm: np.ndarray
    active_mask: np.ndarray
    covariance: DisplacementCovariance
    variance_floor_mm2: float = 0.
    loading_phase: str = "static"
    known_healthy: bool = False
    reaction_node_ids: np.ndarray | None = None
    reaction_component: int = 0
    reaction_standard_deviation_n: float | None = None

    def __post_init__(self):
        label(self.observation_id, "observation ID")
        x, mask = array(self.coordinates_mm), array(self.active_mask, kind="bool")
        if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
                or mask.shape != (len(x),) or not mask.any()
                or not isinstance(self.covariance, DisplacementCovariance)
                or len(self.covariance.matrix) != 2*len(x)):
            raise ContractError("design requires irregular [Q,2] mm, active [Q] and matching covariance")
        object.__setattr__(self, "coordinates_mm", x)
        object.__setattr__(self, "active_mask", mask)
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "cycle", minimum=0))
        object.__setattr__(self, "variance_floor_mm2", scalar(self.variance_floor_mm2, "variance floor", minimum=0))
        label(self.loading_phase, "loading phase")
        if not isinstance(self.known_healthy, bool):
            raise ContractError("known_healthy must be a declared bool")
        if (self.reaction_node_ids is None) != (self.reaction_standard_deviation_n is None):
            raise ContractError("reaction design needs nodes and uncertainty together")
        if self.reaction_node_ids is not None:
            r = ReactionObservation(self.reaction_node_ids, self.reaction_component, 0., self.reaction_standard_deviation_n)
            object.__setattr__(self, "reaction_node_ids", r.node_ids)

    def empty_case(self):
        """Public zero placeholder used only to reuse Phase 1C support validation."""
        o = DICObservation(self.coordinates_mm, np.zeros_like(self.coordinates_mm), self.active_mask,
            np.ones(len(self.active_mask)), self.cycle_count, self.loading_phase,
            self.pattern.loading_state.load_id, self.pattern.mesh.coordinate_system, "mm", "mm")
        r = None if self.reaction_node_ids is None else ReactionObservation(
            self.reaction_node_ids, self.reaction_component, 0., self.reaction_standard_deviation_n)
        return StaticLoadCase(self.pattern, DICMeasurement(o, self.covariance, self.active_mask),
                              r, self.variance_floor_mm2, self.known_healthy)


@dataclass(frozen=True)
class ObservationFrame:
    observation_id: str
    case: StaticLoadCase

    def __post_init__(self):
        label(self.observation_id, "observation ID")
        if not isinstance(self.case, StaticLoadCase):
            raise ContractError("frame requires an observation-only Phase 1C StaticLoadCase")

    @property
    def cycle_count(self):
        return self.case.measurement.observation.cycle_count

    @property
    def design(self):
        c, o = self.case, self.case.measurement.observation
        r = c.reaction
        return ObservationDesign(self.observation_id, c.pattern, o.cycle_count, o.coordinates,
            c.measurement.active_mask, c.measurement.covariance, c.variance_floor_mm2,
            o.loading_phase, c.known_healthy, None if r is None else r.node_ids,
            0 if r is None else r.component, None if r is None else r.standard_deviation_n)


@dataclass(frozen=True)
class ObservationBatch:
    frames: tuple[ObservationFrame, ...]
    noise_contract: str = "conditionally_independent"

    def __post_init__(self):
        fs = tuple(self.frames)
        if not fs or any(not isinstance(f, ObservationFrame) for f in fs):
            raise ContractError("observation batch needs nonempty ObservationFrame entries")
        if len({f.observation_id for f in fs}) != len(fs):
            raise ContractError("duplicate observation IDs would count the same evidence twice")
        if any(a.cycle_count > b.cycle_count for a, b in zip(fs, fs[1:])):
            raise ContractError("observation batch must be chronological")
        if self.noise_contract not in {"conditionally_independent", "independent_given_shared_registration"}:
            raise ContractError("unmodelled cross-frame noise correlation is unsupported")
        object.__setattr__(self, "frames", fs)

    @property
    def ids(self):
        return tuple(f.observation_id for f in self.frames)


@dataclass(frozen=True, eq=False)
class PredictedObservation:
    observation_id: str
    cycle_count: float
    coordinates_mm: np.ndarray
    displacement_mm: np.ndarray
    active_mask: np.ndarray
    reaction_n: float | None
    role: EstimateRole
    conditioned_observation_ids: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "role", EstimateRole(self.role))
        for name in ("coordinates_mm", "displacement_mm"):
            object.__setattr__(self, name, array(getattr(self, name)))
        object.__setattr__(self, "active_mask", array(self.active_mask, kind="bool"))
        x, u, mask = self.coordinates_mm, self.displacement_mm, self.active_mask
        if (x.ndim != 2 or x.shape[1] != 2 or u.shape != x.shape or mask.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(u[mask]).all()):
            raise ContractError("predicted irregular DIC support is invalid")
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "cycle", minimum=0))
        if self.reaction_n is not None:
            object.__setattr__(self, "reaction_n", scalar(self.reaction_n, "predicted reaction"))


@dataclass(frozen=True, eq=False)
class StatePrediction:
    cycle_count: float
    damage: FieldData
    observations: tuple[PredictedObservation, ...]
    role: EstimateRole
    conditioned_through_cycle: float
    conditioned_observation_ids: tuple[str, ...]
    damage_standard_deviation: np.ndarray | None = None
    displacement_standard_deviation_mm: tuple[np.ndarray, ...] = ()
    uncertainty_status: str = "not_propagated"

    def __post_init__(self):
        role = EstimateRole(self.role)
        if role not in {EstimateRole.PREDICTION, EstimateRole.FORECAST}:
            raise ContractError("StatePrediction must label pre-update prediction or open-loop forecast")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "prediction cycle", minimum=0))
        object.__setattr__(self, "conditioned_through_cycle", scalar(self.conditioned_through_cycle, "conditioning cycle", minimum=0))
        self.damage.require(units="1", components=(), fully_valid=True)
        if any(o.role != role or o.cycle_count != self.cycle_count or
               o.conditioned_observation_ids != self.conditioned_observation_ids for o in self.observations):
            raise ContractError("predicted observation conditioning/role disagrees with state")
        if self.cycle_count < self.conditioned_through_cycle:
            raise ContractError("prediction precedes conditioning cutoff")
        if self.damage_standard_deviation is not None:
            s = array(self.damage_standard_deviation)
            if s.shape != self.damage.values.shape or not np.isfinite(s).all() or np.any(s < 0):
                raise ContractError("damage uncertainty support is invalid")
            object.__setattr__(self, "damage_standard_deviation", s)
        us = tuple(array(s) for s in self.displacement_standard_deviation_mm)
        if us and (len(us) != len(self.observations) or any(s.shape != o.displacement_mm.shape
                or not np.isfinite(s[o.active_mask]).all() or np.any(s[o.active_mask] < 0)
                for s, o in zip(us, self.observations))):
            raise ContractError("displacement uncertainty support is invalid")
        object.__setattr__(self, "displacement_standard_deviation_mm", us)


@dataclass(frozen=True, eq=False)
class MechanicsBelief:
    parameter_names: tuple[str, ...]
    normalized_mean: np.ndarray
    parameter_mean: np.ndarray
    uncertainty: object
    current_damage: FieldData
    cycle_count: float
    role: EstimateRole
    observation_ids: tuple[str, ...]
    prior: GaussianPrior | None
    provenance_json: str
    innovation: object | None = None
    damage_summary: str = "physics_at_parameter_MAP_not_damage_posterior_mean"

    def __post_init__(self):
        z, p = array(self.normalized_mean), array(self.parameter_mean)
        if (not self.parameter_names or len(set(self.parameter_names)) != len(self.parameter_names)
                or z.shape != (len(self.parameter_names),) or p.shape != z.shape
                or not np.isfinite(z).all() or not np.isfinite(p).all() or np.any(np.abs(z) > 1)):
            raise ContractError("belief parameter support is invalid")
        object.__setattr__(self, "normalized_mean", z)
        object.__setattr__(self, "parameter_mean", p)
        role = EstimateRole(self.role)
        if role not in {EstimateRole.PRIOR, EstimateRole.FILTERED}:
            raise ContractError("parameter belief must be prior or filtered")
        object.__setattr__(self, "role", role)
        self.current_damage.require(units="1", components=(), fully_valid=True)
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "cycle", minimum=0))

    @property
    def parameter_covariance(self):
        return self.uncertainty.covariance_physical


BeliefState = MechanicsBelief


class BeliefEstimator(Protocol):
    """Phase 2B may implement this protocol without changing the evaluator."""
    def estimate_initial(self, observations=None): ...
    def predict_to_cycle(self, cycle_count, designs=(), *, propagate_uncertainty=False): ...
    def assimilate_observation(self, observation): ...
    def forecast_future(self, designs, *, propagate_uncertainty=True): ...
