"""Keep pre-assimilation innovations distinct from post-fit residuals."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular
from fatigue_dic_rl.core.validation import ContractError, array
from .belief_state import EstimateRole


@dataclass(frozen=True, eq=False)
class Innovation:
    observation_id: str
    cycle_count: float
    kind: str
    displacement_residual_mm: np.ndarray
    measurement_whitened: np.ndarray
    predictive_whitened: np.ndarray | None
    normalized_innovation_squared: float | None
    uncertainty_status: str

    def __post_init__(self):
        if self.kind not in {"pre_assimilation_innovation", "post_fit_residual", "heldout_forecast_innovation"}:
            raise ContractError("innovation kind must distinguish prediction and fitting")
        for name in ("displacement_residual_mm", "measurement_whitened", "predictive_whitened"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, array(getattr(self, name)))

    @property
    def displacement_rmse_mm(self):
        return float(np.sqrt(np.mean(self.displacement_residual_mm**2)))

    @property
    def measurement_whitened_rmse(self):
        return float(np.sqrt(np.mean(self.measurement_whitened**2)))

    @property
    def normalized_nis(self):
        if self.normalized_innovation_squared is None:
            return None
        return self.normalized_innovation_squared/len(self.measurement_whitened)


def innovation(frame, predicted, *, covariance_normalized=None, jacobian_whitened=None,
               posterior_reliable=False):
    if predicted.observation_id != frame.observation_id or predicted.cycle_count != frame.cycle_count:
        raise ContractError("innovation frame and prediction disagree")
    if predicted.role == EstimateRole.FILTERED:
        kind = "post_fit_residual"
    elif predicted.role == EstimateRole.FORECAST:
        kind = "heldout_forecast_innovation"
    elif predicted.role == EstimateRole.PREDICTION:
        kind = "pre_assimilation_innovation"
    else:
        raise ContractError("innovation needs predicted or fitted DIC")
    c = frame.case
    mask = c.measurement.active_mask
    if not np.array_equal(predicted.active_mask, mask):
        raise ContractError("innovation must use the likelihood active mask")
    e = (c.measurement.observation.displacement-predicted.displacement_mm)[mask].ravel()
    factor = c.measurement.covariance.factor(mask, variance_floor_mm2=c.variance_floor_mm2)
    w = factor.whiten(e)
    if c.reaction is not None:
        w = np.r_[w, (c.reaction.value_n-predicted.reaction_n)/c.reaction.standard_deviation_n]
    normalized, nis = None, None
    status = "measurement_covariance_only_predictive_uncertainty_unavailable"
    if kind == "post_fit_residual":
        status = "post_fit_residual_is_not_independent_predictive_innovation"
    elif posterior_reliable and covariance_normalized is not None and jacobian_whitened is not None:
        j = np.asarray(jacobian_whitened)
        if j.shape[0] != len(w):
            raise ContractError("predictive innovation Jacobian support disagrees")
        S = np.eye(len(w))+j @ covariance_normalized @ j.T
        normalized = solve_triangular(np.linalg.cholesky((S+S.T)/2), w, lower=True)
        nis = float(normalized @ normalized)
        status = "measurement_plus_linearized_parameter_covariance"
    return Innovation(frame.observation_id, frame.cycle_count, kind, e, w, normalized, nis, status)
