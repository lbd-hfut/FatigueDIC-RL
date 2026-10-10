"""Bounded TRF multi-start baseline; failed starts retain diagnostics.

SciPy's numerical optimizer uses centered three-point differences. The final
solution is independently checked with two centered difference scales. Only
observation residuals select the best start, never hidden parameter error.
"""
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np
from scipy.optimize import least_squares
from fatigue_dic_rl.core.validation import ContractError, array
from .sensitivity import centered_jacobian


@dataclass(frozen=True, eq=False)
class InverseStartResult:
    initial: np.ndarray
    normalized: np.ndarray
    cost: float
    converged: bool
    message: str
    nfev: int
    optimality: float
    history: tuple
    sensitivity_checked: bool

    def __post_init__(self):
        object.__setattr__(self, "initial", array(self.initial))
        object.__setattr__(self, "normalized", array(self.normalized))


@dataclass(frozen=True)
class MultiStartResult:
    starts: tuple[InverseStartResult, ...]

    @property
    def best(self):
        candidates = [r for r in self.starts if np.isfinite(r.cost) and r.converged]
        if not candidates:
            raise ContractError("all inverse starts failed or did not converge")
        return min(candidates, key=lambda r: r.cost)


def invert(objective, initializations, *, max_nfev=120, loss="linear", f_scale=1., tolerance=1e-9):
    if loss not in {"linear", "soft_l1", "huber"}:
        raise ContractError("unsupported robust residual loss")
    results = []
    for initial in initializations:
        x = np.array(initial, dtype=float)
        objective.adapter.parameter_space.decode(x)
        history = []

        def residual(z):
            try:
                r = objective.residual(z)
            except Exception as exc:
                record = {"parameters": z.tolist(), "failure": type(exc).__name__+": "+str(exc)}
                if hasattr(exc, "diagnostics") and is_dataclass(exc.diagnostics):
                    record["mechanics_diagnostics"] = asdict(exc.diagnostics)
                if hasattr(exc, "failures"):
                    record["fatigue_failures"] = [asdict(f) for f in exc.failures]
                history.append(record)
                raise
            history.append({"parameters": z.tolist(), "half_weighted_square": float(.5*r @ r)})
            return r

        try:
            r = least_squares(residual, x, bounds=(-np.ones(len(x)), np.ones(len(x))), method="trf",
                jac="3-point", diff_step=1e-4, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                ftol=tolerance, xtol=tolerance, gtol=tolerance)
            # Bounds can be active: report check as not applicable there rather
            # than treating one-sided / clipped differences as centered ones.
            checked = False
            if np.all(np.abs(r.x) < 1-1e-8):
                checked = centered_jacobian(objective.adapter.whitened_prediction, r.x).converged
            results.append(InverseStartResult(x, r.x, float(r.cost), bool(r.success), r.message,
                r.nfev, float(r.optimality), tuple(history), checked))
        except Exception as exc:
            results.append(InverseStartResult(x, x, float("inf"), False, type(exc).__name__+": "+str(exc),
                0, float("inf"), tuple(history), False))
    if not results:
        raise ContractError("multi-start inversion requires at least one start")
    return MultiStartResult(tuple(results))
