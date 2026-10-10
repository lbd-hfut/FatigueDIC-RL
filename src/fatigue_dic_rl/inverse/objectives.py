"""Weighted observation residual plus explicit dimensionless Gaussian prior."""
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array


class InverseObjective:
    def __init__(self, adapter, *, prior_mean=None, prior_standard_deviation=None):
        self.adapter = adapter
        self.prior_mean = None if prior_mean is None else array(prior_mean)
        self.prior_standard_deviation = None if prior_standard_deviation is None else array(prior_standard_deviation)
        if (self.prior_mean is None) != (self.prior_standard_deviation is None):
            raise ContractError("prior mean and scale must be supplied together")
        if self.prior_mean is not None:
            adapter.parameter_space.decode(self.prior_mean)
            if self.prior_standard_deviation.shape != self.prior_mean.shape or not np.isfinite(self.prior_standard_deviation).all() or np.any(self.prior_standard_deviation <= 0):
                raise ContractError("prior scales must be positive dimensionless [K]")

    def residual(self, normalized):
        r = self.adapter.residual(normalized)
        if self.prior_mean is not None:
            r = np.concatenate((r, (np.asarray(normalized)-self.prior_mean)/self.prior_standard_deviation))
        return r
