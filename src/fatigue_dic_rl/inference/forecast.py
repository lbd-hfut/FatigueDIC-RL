"""Open-loop forecasts conditioned on a frozen parameter belief and public loads."""
from dataclasses import dataclass
import json
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array
from .belief_state import EstimateRole, StatePrediction
from .uncertainty import checked_local_jacobian, sample_gaussian_parameters


@dataclass(frozen=True)
class ForecastResult:
    origin_cycle_count: float
    conditioned_observation_ids: tuple[str, ...]
    predictions: tuple[StatePrediction, ...]
    method: str = "physics_only_rollout_of_frozen_parameter_belief"

    def __post_init__(self):
        if not self.predictions or any(p.role != EstimateRole.FORECAST or
                p.cycle_count <= self.origin_cycle_count or
                p.conditioned_observation_ids != self.conditioned_observation_ids for p in self.predictions):
            raise ContractError("forecast must be strictly future and use one frozen conditioning prefix")


def predictions_from_belief(predictor, belief, cycles, designs=(), *, role=EstimateRole.PREDICTION,
                            propagate_uncertainty=False):
    cycles = tuple(sorted(set(float(t) for t in cycles)))
    designs = tuple(designs)
    if (belief.parameter_names != predictor.parameter_space.names or
            json.loads(belief.provenance_json).get("physics_model_id") != predictor.model_id):
        raise ContractError("prediction belief belongs to another parameter space or physics model")
    if not cycles or min(cycles) < belief.cycle_count:
        raise ContractError("prediction targets must follow the current belief")
    if any(d.cycle_count not in cycles for d in designs):
        raise ContractError("observation design cycles disagree with requested state cycles")
    if len({d.observation_id for d in designs}) != len(designs):
        raise ContractError("prediction designs require unique IDs")
    for d in designs:
        predictor.validate_design(d)
    states = predictor.damage_states(belief.normalized_mean, cycles)
    observations = {t: tuple(predictor.observe(belief.normalized_mean, d, states[t], role=role,
                   conditioned_observation_ids=belief.observation_ids) for d in designs if d.cycle_count == t)
                   for t in cycles}
    damage_sd, displacement_sd = {}, {}
    status = "not_requested"
    if propagate_uncertainty:
        status = "unavailable_"+belief.uncertainty.status
        if belief.uncertainty.reliable_for_propagation:
            def output(z):
                ss = predictor.damage_states(z, cycles)
                parts = []
                for t in cycles:
                    parts.append(ss[t].damage.values)
                    for d in designs:
                        if d.cycle_count == t:
                            p = predictor.observe(z, d, ss[t])
                            parts.append(p.displacement_mm[d.active_mask].ravel())
                return np.concatenate(parts)

            sensitivity, method = checked_local_jacobian(output, belief.normalized_mean)
            if sensitivity.converged:
                j, c = sensitivity.jacobian, belief.uncertainty.covariance_normalized
                variance = np.einsum("ij,jk,ik->i", j, c, j)
                tol = 1e-12*max(float(np.max(np.abs(variance))), 1e-30)
                if np.min(variance) < -tol:
                    raise ContractError("linearized output variance is not positive semidefinite")
                sd = np.sqrt(np.maximum(variance, 0.))
                start = 0
                for t in cycles:
                    size = len(states[t].damage.values)
                    damage_sd[t] = sd[start:start+size]; start += size
                    us = []
                    for o in observations[t]:
                        size = 2*int(np.sum(o.active_mask))
                        u_sd = np.full_like(o.displacement_mm, np.nan)
                        u_sd[o.active_mask] = sd[start:start+size].reshape(-1, 2); start += size
                        us.append(u_sd)
                    displacement_sd[t] = tuple(us)
                status = "linearized_parameter_pushforward_"+method
            else:
                status = "unavailable_output_derivative_check_failed"
    return tuple(StatePrediction(t, states[t].damage, observations[t], role, belief.cycle_count,
        belief.observation_ids, damage_sd.get(t), displacement_sd.get(t, ()), status) for t in cycles)


def forecast_future(predictor, belief, designs, *, propagate_uncertainty=True):
    designs = tuple(designs)
    if not designs or any(d.cycle_count <= belief.cycle_count for d in designs):
        raise ContractError("forecast designs must be strictly after the assimilation cutoff")
    ps = predictions_from_belief(predictor, belief, [d.cycle_count for d in designs], designs,
                                role=EstimateRole.FORECAST, propagate_uncertainty=propagate_uncertainty)
    return ForecastResult(belief.cycle_count, belief.observation_ids, ps)


def propagate_parameter_samples(predictor, belief, designs, *, count=64, seed=0):
    """Nonlinear parameter pushforward; retain every failed physical sample.

    Damage sample mean differs from the plug-in estimate. Rejected physical
    samples make this a conditional diagnostic rather than a complete posterior.
    """
    designs = tuple(designs)
    if not designs:
        raise ContractError("sample propagation needs public observation designs")
    samples = sample_gaussian_parameters(belief, predictor.parameter_space, count=count, seed=seed)
    cycles = tuple(sorted({d.cycle_count for d in designs}))
    damage, displacement, failures = [], [], []
    for index, z in enumerate(samples.normalized):
        try:
            states = predictor.damage_states(z, cycles)
            damage.append(np.array([states[t].damage.values for t in cycles]))
            displacement.append(tuple(predictor.observe(z, d, states[d.cycle_count]).displacement_mm for d in designs))
        except Exception as exc:
            failures.append({"sample_index": index, "normalized": z.tolist(), "type": type(exc).__name__, "message": str(exc)})
    if not damage:
        raise ContractError("all posterior parameter samples failed physics rollout")
    ds = np.array(damage)
    return {"cycles": cycles, "damage_samples": array(ds), "damage_sample_mean": array(ds.mean(axis=0)),
        "damage_quantiles": array(np.quantile(ds, [.025, .975], axis=0)),
        "displacement_samples_mm": tuple(tuple(array(u) for u in row) for row in displacement),
        "attempted_parameter_draws": samples.attempted, "bounds_rejections": samples.rejected_outside_bounds,
        "physical_failures": tuple(failures), "accepted_physical_samples": len(damage),
        "sample_status": samples.status, "interpretation": "parameter_pushforward_not_arbitrary_field_posterior"}
