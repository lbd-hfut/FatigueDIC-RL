"""Checked centered finite differences on dimensionless parameters.

No clipping or one-sided fallback at boundaries. Difference errors provide an
absolute numerical resolution scale for interpreting tiny singular values.
"""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar


@dataclass(frozen=True, eq=False)
class SensitivityResult:
    jacobian: np.ndarray
    coarse_jacobian: np.ndarray
    steps: np.ndarray
    column_relative_changes: np.ndarray
    absolute_error_norm: float
    converged: bool
    roundoff_resolution: float = 0.

    def __post_init__(self):
        for name in ("jacobian", "coarse_jacobian", "steps", "column_relative_changes"):
            object.__setattr__(self, name, array(getattr(self, name), name=name))


def centered_jacobian(function, normalized, *, step=1e-3, relative_tolerance=.01,
                      absolute_tolerance=1e-7, max_refinements=3):
    z = np.asarray(normalized, dtype=float)
    h = scalar(step, "dimensionless FD step", minimum=0)
    relative_tolerance = scalar(relative_tolerance, "FD relative tolerance", minimum=0)
    absolute_tolerance = scalar(absolute_tolerance, "FD absolute tolerance", minimum=0)
    if isinstance(max_refinements, bool) or not isinstance(max_refinements, int) or max_refinements < 0:
        raise ContractError("max_refinements must be a nonnegative integer")
    if z.ndim != 1 or not np.isfinite(z).all() or np.any(np.abs(z) >= 1) or h == 0:
        raise ContractError("centered sensitivity needs interior parameters and positive step")
    hvec = np.minimum(h, (1-np.abs(z))/3)
    y = np.asarray(function(z), dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ContractError("prediction must be a finite vector")

    def difference(steps):
        columns = []
        for i, hi in enumerate(steps):
            e = np.zeros_like(z); e[i] = hi
            a, b = np.asarray(function(z+e)), np.asarray(function(z-e))
            if a.shape != y.shape or b.shape != y.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ContractError("perturbed forward response invalid")
            columns.append((a-b)/(2*hi))
        return np.column_stack(columns)

    coarse = difference(hvec)
    for _ in range(max_refinements+1):
        fine = difference(hvec/2)
        errors = np.linalg.norm(fine-coarse, axis=0)
        norms = np.linalg.norm(fine, axis=0)
        converged = bool(np.all(errors <= absolute_tolerance+relative_tolerance*norms))
        if converged:
            break
        hvec /= 2
        coarse = fine
    relative = errors/np.maximum(norms, absolute_tolerance)
    # Conservative floating-point resolution estimate, not a rigorous FE error
    # bound. Subtracting nearly identical FE responses cannot establish a
    # nonzero sensitivity below this scale, even if two steps agree by chance.
    roundoff = float(64*np.finfo(float).eps*np.linalg.norm(y)*np.sqrt(len(z))/np.min(hvec/2))
    error = max(float(np.linalg.norm(fine-coarse, 2)), roundoff)
    return SensitivityResult(fine, coarse, hvec/2, relative, error, converged, roundoff)
