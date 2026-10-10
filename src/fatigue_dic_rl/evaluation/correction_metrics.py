"""Quadrature-weighted damage metrics; no uncertainty qualification inherited."""
import numpy as np
from fatigue_dic_rl.core.validation import ContractError


def damage_metrics(predicted, reference, weights):
    a,b,w=map(np.asarray,(predicted,reference,weights))
    if a.ndim < 1 or w.ndim != 1 or not len(w) or a.shape!=b.shape or a.shape[-1]!=len(w) or np.any(w<=0) or not np.isfinite(w).all() or not np.isfinite([a,b]).all():raise ContractError('incompatible finite quadrature damage metrics')
    error=a-b
    return {'rmse':float(np.sqrt(np.sum(error**2*w)/np.sum(np.broadcast_to(w,a.shape)))),
        'mae':float(np.sum(abs(error)*w)/np.sum(np.broadcast_to(w,a.shape))), 'max_error':float(np.max(abs(error)))}


def correction_factor_metrics(values):
    v=np.asarray(values);return {'minimum':float(v.min()),'maximum':float(v.max()),'mean':float(v.mean()),'standard_deviation':float(v.std())}
