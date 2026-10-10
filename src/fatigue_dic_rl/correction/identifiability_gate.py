"""Practical observability on data-only information, separate from numerical rank."""
import numpy as np
from fatigue_dic_rl.core.validation import ContractError


def observability(uncertainty, normalized, *, minimum_singular_value=3.):
    z = np.asarray(normalized, float)
    if not np.isfinite(z).all() or not np.isfinite(minimum_singular_value) or minimum_singular_value <= 0:
        raise ContractError('invalid observability request')
    s = uncertainty.singular_values_data
    reliable_fd = 'finite_difference_check_failed' not in uncertainty.warnings
    rank = int(np.sum(s >= max(minimum_singular_value, uncertainty.rank_threshold_data))) if reliable_fd else 0
    boundary = bool(np.any(np.abs(z) >= .995))
    condition = float(s[0]/s[-1]) if len(s) and s[-1] > 0 else None
    return dict(numerical_rank=uncertainty.data_rank, practical_rank=rank, dimension=len(z),
                minimum_singular_value=float(s[-1]) if len(s) else 0., condition_number=condition,
                derivative_checked=reliable_fd, boundary=boundary,
                eligible=bool(reliable_fd and rank == len(z) and not boundary and 'optimization_not_converged' not in uncertainty.warnings),
                information='data only; dimensionless parameter-box coordinates; no ridge information')


def conditional_singular_values(jacobian, target_indices):
    """Project out d0/nuisance using square-root information (all cross terms)."""
    j = np.asarray(jacobian, float); indices = tuple(target_indices)
    if j.ndim != 2 or not np.isfinite(j).all() or len(set(indices)) != len(indices) or any(i < 0 or i >= j.shape[1] for i in indices):
        raise ContractError('invalid conditional sensitivity')
    other = [i for i in range(j.shape[1]) if i not in indices]
    target = j[:, indices]
    if other:
        nuisance = j[:, other]
        # Least-squares projection retains correlations, including shared registration.
        target = target-nuisance@np.linalg.lstsq(nuisance,target,rcond=1e-6)[0]
    return np.linalg.svd(target,compute_uv=False)
