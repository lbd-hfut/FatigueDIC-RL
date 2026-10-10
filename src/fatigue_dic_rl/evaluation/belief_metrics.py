"""Truth-aware metrics and temporal file access live outside inference."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.inference.belief_state import ObservationFrame, ObservationBatch


def rmse(estimate, reference):
    a, b = np.asarray(estimate), np.asarray(reference)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ContractError("RMSE needs finite matching nonempty supports")
    return float(np.sqrt(np.mean((a-b)**2)))


@dataclass(frozen=True)
class TemporalSplit:
    assimilation_cycles: tuple[float, ...] = (0., 10., 20.)
    forecast_cycles: tuple[float, ...] = (30., 40.)

    def __post_init__(self):
        a, f = tuple(self.assimilation_cycles), tuple(self.forecast_cycles)
        if (not a or not f or a != tuple(sorted(set(a))) or f != tuple(sorted(set(f)))
                or not np.isfinite(a+f).all() or min(a) < 0 or max(a) >= min(f)):
            raise ContractError("temporal split requires disjoint ordered assimilation and strictly later forecasts")
        object.__setattr__(self, "assimilation_cycles", a)
        object.__setattr__(self, "forecast_cycles", f)

    @property
    def cutoff(self):
        return self.assimilation_cycles[-1]


@dataclass(frozen=True)
class ObservationFileRef:
    observation_id: str
    cycle_count: float
    folder: Path
    pattern: object
    variance_floor_mm2: float = 0.


def load_observation_prefix(entries, split, *, noise_contract="conditionally_independent"):
    """Filter a PUBLIC schedule before loading displacement files.

    No truth path is accepted. Future measurements are not opened, including
    their NPZs. Acquisition times are checked against the public schedule.
    """
    frames = []
    for entry in entries:
        if entry.cycle_count not in split.assimilation_cycles:
            continue
        m = DICMeasurement.load(entry.folder)
        if m.observation.cycle_count != entry.cycle_count:
            raise ContractError("loaded observation time disagrees with public schedule")
        frames.append(ObservationFrame(entry.observation_id, StaticLoadCase(
            entry.pattern, m, variance_floor_mm2=entry.variance_floor_mm2)))
    if tuple(f.cycle_count for f in frames) != split.assimilation_cycles:
        raise ContractError("observation prefix does not match declared temporal split")
    return ObservationBatch(tuple(frames), noise_contract)


def belief_metrics(belief, parameter_space, true_parameters, true_damage):
    truth = np.array([true_parameters[n] for n in belief.parameter_names])
    error = belief.parameter_mean-truth
    u = belief.uncertainty
    return {"role": belief.role.value, "cycle_count": belief.cycle_count,
        "parameters": dict(zip(belief.parameter_names, belief.parameter_mean.tolist())),
        "physical_parameter_error": dict(zip(belief.parameter_names, error.tolist())),
        "normalized_parameter_rmse": float(np.sqrt(np.mean((error/parameter_space.scales)**2))),
        "current_damage_rmse": rmse(belief.current_damage.values, true_damage),
        "current_damage_mean_plugin": float(belief.current_damage.values.mean()),
        "data_rank": u.data_rank, "posterior_rank": u.posterior_rank,
        "parameter_standard_deviation": dict(zip(belief.parameter_names, u.marginal_standard_deviation_physical.tolist())),
        "posterior_status": u.status, "posterior_warnings": u.warnings,
        "post_fit_dic_rmse_mm": None if belief.innovation is None else belief.innovation.displacement_rmse_mm}


def coverage_summary(errors, standard_deviations, *, nominal=.95):
    """Pointwise per-seed coverage and Wilson precision, not independent FE points."""
    errors, sd = np.asarray(errors), np.asarray(standard_deviations)
    if errors.shape != sd.shape:
        raise ContractError("coverage errors and uncertainty support disagree")
    valid = np.isfinite(errors) & np.isfinite(sd) & (sd > 0)
    n = int(valid.sum())
    if not n:
        return {"nominal": nominal, "valid_count": 0, "coverage": None, "wilson_interval_95": None}
    if nominal != .95:
        raise ContractError("this protocol declares two-sided 95 percent intervals")
    covered = int(np.sum(np.abs(errors[valid]) <= 1.959963984540054*sd[valid]))
    p, z = covered/n, 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"nominal": nominal, "valid_count": n, "covered_count": covered, "coverage": p,
            "wilson_interval_95": [center-half, center+half], "interval": "local_linearized_Gaussian"}
