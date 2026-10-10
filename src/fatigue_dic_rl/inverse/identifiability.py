"""Local Fisher diagnostics. Rank deficiency retains explicit null directions."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from .sensitivity import SensitivityResult


@dataclass(frozen=True, eq=False)
class FisherDiagnostics:
    singular_values: np.ndarray
    fisher: np.ndarray
    effective_rank: int
    condition_number: float
    right_directions: np.ndarray
    null_directions: np.ndarray
    identifiable_covariance: np.ndarray
    parameter_correlation: np.ndarray
    rank_threshold: float

    def __post_init__(self):
        for name in ("singular_values", "fisher", "right_directions", "null_directions",
                     "identifiable_covariance", "parameter_correlation"):
            object.__setattr__(self, name, array(getattr(self, name), name=name))


def fisher_diagnostics(sensitivity, *, rank_rtol=1e-6, numerical_error_multiplier=5.):
    if not isinstance(sensitivity, SensitivityResult) or not sensitivity.converged:
        raise ContractError("Fisher analysis requires a checked, converged sensitivity")
    j = sensitivity.jacobian
    if j.ndim != 2 or not np.isfinite(j).all():
        raise ContractError("whitened Jacobian must be finite [M,K]")
    _, s, vh = np.linalg.svd(j, full_matrices=j.shape[0] < j.shape[1])
    s = np.pad(s, (0, j.shape[1]-len(s)))
    threshold = max(scalar(rank_rtol, "rank rtol", minimum=0)*s[0],
                    numerical_error_multiplier*sensitivity.absolute_error_norm, 1e-9)
    rank = int(np.sum(s > threshold))
    v = vh.T
    covariance = (v[:, :rank]/s[:rank]**2) @ v[:, :rank].T if rank else np.zeros((j.shape[1], j.shape[1]))
    column_norm = np.linalg.norm(j, axis=0)
    denominator = column_norm[:, None]*column_norm[None, :]
    correlation = np.divide(j.T @ j, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    condition = float(s[0]/s[-1]) if rank == len(s) else float("inf")
    return FisherDiagnostics(s, j.T @ j, rank, condition, vh, vh[rank:], covariance, correlation, threshold)
