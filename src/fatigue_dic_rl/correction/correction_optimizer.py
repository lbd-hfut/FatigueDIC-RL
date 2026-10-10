"""Full-prefix, multi-trajectory TRF in normalized low-dimensional coordinates."""
from dataclasses import dataclass
import time
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inference.belief_state import ObservationBatch
from fatigue_dic_rl.inference.physics_predictor import PhysicsForwardAdapter
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian, posterior_uncertainty


class MultiTrajectoryAdapter:
    def __init__(self, predictors, batches, *, cutoff=20.):
        if len(predictors) != len(batches) or not predictors:
            raise ContractError('matched predictor/acquisition tuples required')
        ids = [f.observation_id for b in batches for f in b.frames]
        if len(set(ids)) != len(ids) or any(f.cycle_count > cutoff for b in batches for f in b.frames):
            raise ContractError('duplicate evidence or future measurement in correction prefix')
        self.parameter_space = predictors[0].parameter_space
        if any(p.parameter_space.names != self.parameter_space.names or p.parameter_space.parameters != self.parameter_space.parameters or dict(p.parameter_space.fixed) != dict(self.parameter_space.fixed) for p in predictors):
            raise ContractError('joint trajectories must share one declared material parameter space')
        self.adapters = tuple(PhysicsForwardAdapter(p, b) for p, b in zip(predictors, batches))
        self.evidence_ids = tuple(ids)

    def residual(self, z):
        return np.concatenate([a.residual(z) for a in self.adapters])

    def whitened_prediction(self, z):
        return np.concatenate([a.whitened_prediction(z) for a in self.adapters])

    def displacement_prediction(self, z):
        return np.concatenate([a.displacement_prediction(z) for a in self.adapters])

    @property
    def evaluation_count(self):
        return sum(a.evaluation_count for a in self.adapters)


class CorrectionObjective(InverseObjective):
    def __init__(self, adapter, *, coefficient_regularization=1e-3):
        super().__init__(adapter)
        if not np.isfinite(coefficient_regularization) or coefficient_regularization < 0:
            raise ContractError('regularization must be nonnegative and finite')
        self.regularization = coefficient_regularization
        self.coefficient_names = tuple(n for n in adapter.parameter_space.names if n.startswith('c') and n[1:].isdigit())

    def residual(self, z):
        data = self.adapter.residual(z)
        p = self.adapter.parameter_space.decode(z)
        return np.r_[data, np.sqrt(self.regularization)*np.array([p[n] for n in self.coefficient_names])]


@dataclass(frozen=True)
class CorrectionFit:
    normalized: np.ndarray
    physical: dict
    status: str
    optimization: object
    uncertainty: object
    wall_seconds: float
    evidence_ids: tuple
    data_cost: float
    regularization_cost: float


class CorrectionOptimizationError(ContractError):
    def __init__(self, result):
        super().__init__('all correction starts failed; retained optimization available on exception.optimization')
        self.optimization = result


def fit_correction(adapter, *, initializations=None, max_nfev=80, coefficient_regularization=1e-3):
    starts = initializations or [np.zeros(len(adapter.parameter_space.names))]
    objective = CorrectionObjective(adapter, coefficient_regularization=coefficient_regularization)
    t = time.perf_counter()
    result = invert(objective, starts, max_nfev=max_nfev)
    valid = [r for r in result.starts if np.isfinite(r.cost)]
    if not valid:
        raise CorrectionOptimizationError(result)
    converged = [r for r in valid if r.converged]
    selected = min(converged or valid, key=lambda r:r.cost)
    z = selected.normalized
    s, method = checked_local_jacobian(adapter.whitened_prediction, z)
    u = posterior_uncertainty(s, adapter.parameter_space.scales, normalized=z,
                              converged=selected.converged, derivative_method=method)
    residual = adapter.residual(z)
    return CorrectionFit(z.copy(), adapter.parameter_space.decode(z), 'CONVERGED' if selected.converged else 'UNCONVERGED_TENTATIVE', result, u, time.perf_counter()-t, adapter.evidence_ids, float(.5*residual@residual), float(.5*objective.residual(z)@objective.residual(z)-.5*residual@residual))
