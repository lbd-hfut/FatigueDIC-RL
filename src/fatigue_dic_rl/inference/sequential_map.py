"""Auditable prefix MAP with a fixed prior, Phase 1C TRF and atomic publication.

Every update reoptimizes the complete observed prefix. The previous posterior
is a warm start, never a second prior: each frame and the original prior enter
the objective exactly once. Local minima and failed starts remain visible.
"""
from dataclasses import dataclass, replace
import json
import time
import numpy as np

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from .belief_state import (MechanicsBelief, ObservationFrame, ObservationBatch, EstimateRole)
from .physics_predictor import PhysicsForwardAdapter
from .uncertainty import posterior_uncertainty, checked_local_jacobian
from .innovation import innovation
from .forecast import predictions_from_belief, forecast_future


@dataclass(frozen=True)
class SequentialMAPConfig:
    warm_start: bool = True
    max_nfev: int = 120
    tolerance: float = 1e-9
    assimilation_cutoff: float | None = None
    rank_rtol: float = 1e-6

    def __post_init__(self):
        if not isinstance(self.warm_start, bool) or isinstance(self.max_nfev, bool) or not isinstance(self.max_nfev, int) or self.max_nfev < 1:
            raise ContractError("MAP needs bool warm_start and positive integer evaluation budget")
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1 or not np.isfinite(self.rank_rtol) or self.rank_rtol <= 0:
            raise ContractError("MAP tolerances must be finite and positive")
        if self.assimilation_cutoff is not None and (not np.isfinite(self.assimilation_cutoff) or self.assimilation_cutoff < 0):
            raise ContractError("assimilation cutoff must be nonnegative")


@dataclass(frozen=True)
class AssimilationResult:
    status: str
    prefix_ids: tuple[str, ...]
    optimization: object | None
    selected_start_index: int | None
    filtered_belief: MechanicsBelief | None
    predictions_before_update: tuple
    innovations_before_update: tuple
    elapsed_seconds: float
    forward_evaluations: int
    diagnostics: tuple[str, ...] = ()

    @property
    def selected(self):
        return None if self.selected_start_index is None else self.optimization.starts[self.selected_start_index]


class PhysicsBasedBeliefEstimator:
    def __init__(self, predictor, *, prior=None, initializations=None, config=None,
                 noise_contract="conditionally_independent"):
        self.predictor = predictor
        self.parameter_space = predictor.parameter_space
        self.prior = prior
        self.config = config or SequentialMAPConfig()
        self.noise_contract = noise_contract
        predictor.validate_noise_contract(noise_contract)
        if noise_contract not in {"conditionally_independent", "independent_given_shared_registration"}:
            raise ContractError("cross-frame dependence must be represented explicitly")
        k = len(self.parameter_space.names)
        starts = initializations if initializations is not None else (np.zeros(k), np.full(k, -.35), np.full(k, .35))
        self.initializations = tuple(np.array(s, dtype=float) for s in starts)
        if not self.initializations:
            raise ContractError("declare at least one truth-independent initialization")
        for z in self.initializations:
            self.parameter_space.decode(z)
        if prior is not None:
            self.parameter_space.decode(prior.normalized_mean)
        self._frames = ()
        self._attempts = []
        self._belief = None
        self.estimate_initial()

    @property
    def belief(self):
        return self._belief

    @property
    def attempts(self):
        return tuple(self._attempts)

    @property
    def observed_ids(self):
        return tuple(f.observation_id for f in self._frames)

    def _provenance(self, ids, role):
        return json.dumps({"algorithm": "prefix_MAP_TRF", "role": role.value,
            "physics_model_id": self.predictor.model_id, "physics": self.predictor.provenance(),
            "conditioning_ids": ids, "noise_contract": self.noise_contract,
            "likelihood_terms": len(ids), "fixed_prior_terms": int(self.prior is not None),
            "previous_posterior_as_prior": False, "config": dict(vars(self.config)),
            "initializations": [z.tolist() for z in self.initializations],
            "posterior": "local_Gauss_Newton_Laplace_no_global_unimodality_claim"}, sort_keys=True)

    def estimate_initial(self, observations=None):
        if self._belief is None:
            z = self.prior.normalized_mean if self.prior is not None else self.initializations[0]
            state = self.predictor.damage_states(z, [0.])[0.]
            p = self.parameter_space.decode(z)
            u = posterior_uncertainty(None, self.parameter_space.scales, self.prior, normalized=z)
            self._belief = MechanicsBelief(self.parameter_space.names, z, [p[n] for n in self.parameter_space.names],
                u, state.damage, 0., EstimateRole.PRIOR, (), self.prior, self._provenance((), EstimateRole.PRIOR))
        if observations is None:
            return self._belief
        if self._frames:
            raise ContractError("estimate_initial(observations) requires an unassimilated estimator")
        return self.assimilate_observation(observations)

    def predict_to_cycle(self, cycle_count, designs=(), *, propagate_uncertainty=False):
        return predictions_from_belief(self.predictor, self.belief, [cycle_count], designs,
                                      propagate_uncertainty=propagate_uncertainty)[0]

    def forecast_future(self, designs, *, propagate_uncertainty=True):
        return forecast_future(self.predictor, self.belief, designs, propagate_uncertainty=propagate_uncertainty)

    def _prediction_innovation(self, frame):
        design = frame.design
        prediction = self.predict_to_cycle(frame.cycle_count, [design])
        j = None
        u = self.belief.uncertainty
        if u.reliable_for_propagation:
            empty = ObservationFrame(frame.observation_id, design.empty_case())
            ad = PhysicsForwardAdapter(self.predictor, ObservationBatch((empty,), self.noise_contract))
            s, _ = checked_local_jacobian(ad.whitened_prediction, self.belief.normalized_mean)
            if s.converged:
                j = s.jacobian
        e = innovation(frame, prediction.observations[0], covariance_normalized=u.covariance_normalized,
                       jacobian_whitened=j, posterior_reliable=u.reliable_for_propagation)
        return prediction, e

    def assimilate_observation(self, observation):
        start = time.perf_counter()
        new = ObservationBatch((observation,), self.noise_contract) if isinstance(observation, ObservationFrame) else observation
        if not isinstance(new, ObservationBatch) or new.noise_contract != self.noise_contract:
            raise ContractError("update requires a batch with the estimator's noise contract")
        # Reject time leakage before constructing a forward objective or evaluating residuals.
        if self.config.assimilation_cutoff is not None and any(f.cycle_count > self.config.assimilation_cutoff for f in new.frames):
            raise ContractError("future observation cannot enter the assimilation prefix")
        frames = self._frames+tuple(replace(f, case=replace(f.case,
            pattern=self.predictor.public_pattern(f.case.pattern))) for f in new.frames)
        batch = ObservationBatch(frames, self.noise_contract)
        # Parameter consistency/public support validated before publishing anything.
        adapter = PhysicsForwardAdapter(self.predictor, batch)
        objective = InverseObjective(adapter, prior_mean=None if self.prior is None else self.prior.normalized_mean,
            prior_standard_deviation=None if self.prior is None else self.prior.normalized_standard_deviation)
        predictions, innovations, diagnostics = [], [], []
        for frame in new.frames:
            try:
                p, e = self._prediction_innovation(frame)
                predictions.append(p); innovations.append(e)
            except Exception as exc:
                diagnostics.append("pre_prediction_failure: "+type(exc).__name__+": "+str(exc))
        starts = self.initializations
        if self.config.warm_start and self._frames:
            # Warm starts use the previous MAP only. The original prior is unchanged.
            starts = (self.belief.normalized_mean,)
        fit = invert(objective, starts, max_nfev=self.config.max_nfev, tolerance=self.config.tolerance)
        finite = [(i, r) for i, r in enumerate(fit.starts) if np.isfinite(r.cost)]
        converged = [(i, r) for i, r in finite if r.converged]
        selected = min(converged or finite, key=lambda item: item[1].cost) if finite else None
        filtered, index = None, None
        status = "FAILED_ROLLED_BACK"
        if selected is not None:
            index, candidate = selected
            z = candidate.normalized
            try:
                s, method = checked_local_jacobian(adapter.whitened_prediction, z)
                u = posterior_uncertainty(s, self.parameter_space.scales, self.prior, normalized=z,
                    converged=candidate.converged, derivative_method=method, rank_rtol=self.config.rank_rtol)
                separated = any(np.linalg.norm(other.normalized-z) > .1 and other.cost-candidate.cost < 2.
                                for _, other in converged)
                if separated:
                    diagnostics.append("separated_local_optima_at_similar_objective")
                    u = replace(u, status="unreliable_local_approximation",
                                warnings=u.warnings+("separated_local_optima_Laplace_unreliable",))
                latest = frames[-1]
                state = self.predictor.damage_states(z, [latest.cycle_count])[latest.cycle_count]
                fitted = self.predictor.observe(z, latest.design, state, role=EstimateRole.FILTERED,
                                              conditioned_observation_ids=batch.ids)
                e = innovation(latest, fitted)
                p = self.parameter_space.decode(z)
                filtered = MechanicsBelief(self.parameter_space.names, z, [p[n] for n in self.parameter_space.names],
                    u, state.damage, latest.cycle_count, EstimateRole.FILTERED, batch.ids, self.prior,
                    self._provenance(batch.ids, EstimateRole.FILTERED), e)
                status = "CONVERGED" if candidate.converged else "UNCONVERGED_TENTATIVE"
            except Exception as exc:
                diagnostics.append("posterior_publication_failure: "+type(exc).__name__+": "+str(exc))
        result = AssimilationResult(status, batch.ids, fit, index, filtered, tuple(predictions), tuple(innovations),
                                    time.perf_counter()-start, adapter.evaluation_count, tuple(diagnostics))
        self._attempts.append(result)
        if filtered is not None:
            self._belief = filtered
            self._frames = frames
        return result
