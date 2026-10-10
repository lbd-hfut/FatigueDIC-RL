"""Rank-aware Gauss-Newton/Laplace diagnostics in normalized parameter space.

The identifiable-subspace inverse is never represented as a full posterior
covariance. Box bounds are not Gaussian information. A prior can close a data
null space without increasing data identifiability.
"""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian, SensitivityResult
from fatigue_dic_rl.inverse.identifiability import fisher_diagnostics


def checked_local_jacobian(function, normalized):
    """Reuse centered Phase 1C differences; explicitly label boundary derivatives."""
    z = np.asarray(normalized, dtype=float)
    if np.all(np.abs(z) < 1-1e-7):
        return centered_jacobian(function, z), "centered_two_scales"
    y = np.asarray(function(z), dtype=float)
    h = 1e-3

    def difference(step):
        columns = []
        for i in range(len(z)):
            e = np.zeros_like(z)
            if abs(z[i]) < 1-2*step:
                e[i] = step
                column = (np.asarray(function(z+e))-np.asarray(function(z-e)))/(2*step)
            else:
                sign = -1 if z[i] >= 0 else 1
                e[i] = sign*step
                column = sign*(-3*y+4*np.asarray(function(z+e))-np.asarray(function(z+2*e)))/(2*step)
            columns.append(column)
        return np.column_stack(columns)

    coarse, fine = difference(h), difference(h/2)
    errors = np.linalg.norm(fine-coarse, axis=0)
    norms = np.linalg.norm(fine, axis=0)
    roundoff = float(64*np.finfo(float).eps*np.linalg.norm(y)*np.sqrt(len(z))/(h/2))
    result = SensitivityResult(fine, coarse, np.full(len(z), h/2), errors/np.maximum(norms, 1e-7),
        max(float(np.linalg.norm(fine-coarse, 2)), roundoff),
        bool(np.all(errors <= 1e-7+.01*norms)), roundoff)
    return result, "one_sided_second_order_at_boundary_two_scales"


@dataclass(frozen=True, eq=False)
class PosteriorUncertainty:
    jacobian_whitened: np.ndarray
    data_fisher: np.ndarray
    prior_fisher: np.ndarray
    posterior_fisher: np.ndarray
    data_rank: int
    posterior_rank: int
    data_null_directions: np.ndarray
    posterior_null_directions: np.ndarray
    weak_directions: np.ndarray
    singular_values_data: np.ndarray
    singular_values_posterior: np.ndarray
    identifiable_covariance_normalized: np.ndarray
    parameter_scales: np.ndarray
    status: str
    warnings: tuple[str, ...]
    derivative_method: str
    rank_threshold_data: float
    rank_threshold_posterior: float
    finite_difference_error: float

    def __post_init__(self):
        for name in ("jacobian_whitened", "data_fisher", "prior_fisher", "posterior_fisher",
                     "data_null_directions", "posterior_null_directions", "weak_directions",
                     "singular_values_data", "singular_values_posterior", "identifiable_covariance_normalized",
                     "parameter_scales"):
            object.__setattr__(self, name, array(getattr(self, name)))

    @property
    def covariance_normalized(self):
        if self.posterior_rank != len(self.parameter_scales) or "finite_difference_check_failed" in self.warnings:
            return None
        return self.identifiable_covariance_normalized

    @property
    def covariance_physical(self):
        c = self.covariance_normalized
        return None if c is None else array(c*self.parameter_scales[:, None]*self.parameter_scales[None, :])

    @property
    def marginal_standard_deviation_physical(self):
        """Infinity denotes unresolved Gaussian directions, not box-marginal variance."""
        variance = np.diag(self.identifiable_covariance_normalized).copy()
        if "finite_difference_check_failed" in self.warnings:
            variance[:] = np.inf
        if len(self.posterior_null_directions):
            variance[np.linalg.norm(self.posterior_null_directions, axis=0) > 1e-8] = np.inf
        return array(np.sqrt(variance)*self.parameter_scales)

    @property
    def reliable_for_propagation(self):
        return self.status == "local_gaussian" and self.covariance_normalized is not None


def posterior_uncertainty(sensitivity, scales, prior=None, *, normalized=None, converged=True,
                          derivative_method="centered_two_scales", rank_rtol=1e-6):
    scales = np.asarray(scales, dtype=float)
    k = len(scales)
    if sensitivity is None:
        sensitivity = SensitivityResult(np.empty((0, k)), np.empty((0, k)), np.ones(k),
                                         np.zeros(k), 0., True)
    if sensitivity.converged:
        data = fisher_diagnostics(sensitivity, rank_rtol=rank_rtol)
    else:
        # Failed derivative checks cannot establish identifiability.
        unresolved = SensitivityResult(np.zeros_like(sensitivity.jacobian), np.zeros_like(sensitivity.jacobian),
                                        sensitivity.steps, np.zeros(k), 0., True)
        data = fisher_diagnostics(unresolved, rank_rtol=rank_rtol)
    j = sensitivity.jacobian
    prior_root = np.empty((0, k)) if prior is None else np.diag(1/prior.normalized_standard_deviation)
    if prior is not None and len(prior.normalized_mean) != k:
        raise ContractError("prior/parameter dimensions disagree")
    # SVD of the square-root information avoids squaring the condition number.
    stacked = np.vstack((j if sensitivity.converged else np.zeros_like(j), prior_root))
    _, singular, vh = np.linalg.svd(stacked, full_matrices=stacked.shape[0] < k)
    singular = np.pad(singular, (0, k-len(singular)))
    threshold = max(rank_rtol*singular[0], 5*sensitivity.absolute_error_norm, 1e-9)
    rank = int(np.sum(singular > threshold))
    v = vh.T
    covariance = (v[:, :rank]/singular[:rank]**2) @ v[:, :rank].T if rank else np.zeros((k, k))
    warnings = []
    if data.effective_rank < k:
        warnings.append("data_rank_deficient_prior_is_not_DIC_information")
    if rank < k:
        warnings.append("improper_Gaussian_in_null_directions_no_full_covariance")
    if not sensitivity.converged:
        warnings.append("finite_difference_check_failed")
    if normalized is not None and np.any(np.abs(normalized) >= 1-1e-6):
        warnings.append("active_parameter_bound_Laplace_unreliable")
    if not converged:
        warnings.append("optimization_not_converged")
    if normalized is not None and rank == k:
        sd = np.sqrt(np.diag(covariance))
        if np.any(1-np.abs(normalized) < 3*sd):
            warnings.append("Gaussian_mass_near_parameter_box_boundary")
    status = "local_gaussian"
    if rank < k:
        status = "rank_deficient"
    if any(w in warnings for w in ("active_parameter_bound_Laplace_unreliable",
           "Gaussian_mass_near_parameter_box_boundary", "optimization_not_converged", "finite_difference_check_failed")):
        status = "unreliable_local_approximation"
    # Unit normalized displacement changes below one sigma are practically weak.
    weak = data.right_directions[(data.singular_values <= max(1., data.rank_threshold))]
    return PosteriorUncertainty(j, j.T @ j, prior_root.T @ prior_root,
        j.T @ j+prior_root.T @ prior_root, data.effective_rank, rank, data.null_directions,
        vh[rank:], weak, data.singular_values, singular, covariance, scales, status,
        tuple(warnings), derivative_method, data.rank_threshold, threshold, sensitivity.absolute_error_norm)


@dataclass(frozen=True, eq=False)
class ParameterSamples:
    normalized: np.ndarray
    attempted: int
    rejected_outside_bounds: int
    status: str

    def __post_init__(self):
        object.__setattr__(self, "normalized", array(self.normalized))


def sample_gaussian_parameters(belief, parameter_space, *, count, seed):
    """Reject inadmissible samples; never clip or invent variance in null space.

    When rejection occurs these are box-conditioned Gaussian samples, not a
    claim to have computed the bounded nonlinear posterior.
    """
    if not belief.uncertainty.reliable_for_propagation:
        raise ContractError("Gaussian sampling requires a reliable proper local covariance")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("sampling requires positive count and nonnegative seed")
    if tuple(parameter_space.names) != belief.parameter_names:
        raise ContractError("sample parameter space disagrees with belief")
    rng = np.random.default_rng(seed)
    c = belief.uncertainty.covariance_normalized
    root = np.linalg.cholesky((c+c.T)/2)
    accepted, attempted, rejected = [], 0, 0
    while len(accepted) < count and attempted < 100*count:
        z = belief.normalized_mean+root @ rng.standard_normal(len(c))
        attempted += 1
        if np.any(np.abs(z) > 1):
            rejected += 1
            continue
        parameter_space.decode(z)
        accepted.append(z)
    if len(accepted) != count:
        raise ContractError("Gaussian sampling exhausted admissible-sample budget")
    return ParameterSamples(np.array(accepted), attempted, rejected,
        "box_conditioned_Gaussian" if rejected else "local_Gaussian")
