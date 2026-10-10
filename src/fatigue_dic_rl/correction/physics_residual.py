"""Controlled synthetic reference discrepancy; no empirical-law claim."""
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, scalar, array
from fatigue_dic_rl.fatigue.models import FatigueDamageLaw


class StructuralReferenceLaw(FatigueDamageLaw):
    def __init__(self, prior, *, beta=3., spatial_strength=0., structural_power=1, origin_mm=(0., 0.), lengths_mm=(2., 1.)):
        if not isinstance(prior, FatigueDamageLaw):
            raise ContractError('reference discrepancy needs an explicit prior law')
        self.prior, self.beta = prior, scalar(beta, 'beta', minimum=0)
        self.spatial_strength = scalar(spatial_strength, 'spatial strength')
        if isinstance(structural_power, bool) or structural_power not in (1, 2):
            raise ContractError('reference structural power must be 1 or 2')
        self.structural_power = structural_power
        self.origin_mm, self.lengths_mm = np.asarray(origin_mm), np.asarray(lengths_mm)
        if self.origin_mm.shape != (2,) or self.lengths_mm.shape != (2,) or not np.isfinite([self.origin_mm, self.lengths_mm]).all() or np.min(self.lengths_mm) <= 0 or abs(self.spatial_strength) > 2:
            raise ContractError('reference spatial law needs declared physical lengths')

    def _hazard(self, driving, cycle, state):
        raw = self.prior.hazard(driving, cycle, state)
        x = driving.energy_amplitude.values/.025
        q = (state.layout.coordinates-self.origin_mm)/self.lengths_mm
        factor = (1+self.beta*(1-1/(1+x))**self.structural_power)*np.exp(self.spatial_strength*np.cos(np.pi*q[:, 0]))
        with np.errstate(over='raise', invalid='raise'):
            return state.layout.field(raw.values*factor, 'cycle^-1')


def positive_hazard_log_ratio(true_hazard, prior_hazard):
    """Mask exact zeros. Never add epsilon to invent privileged supervision."""
    t, p = np.asarray(true_hazard, dtype=float), np.asarray(prior_hazard, dtype=float)
    if t.shape != p.shape or not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(t < 0) or np.any(p < 0):
        raise ContractError('hazard labels must be matching finite nonnegative arrays')
    mask = (t > 0) & (p > 0)
    targets = np.full(t.shape, np.nan)
    targets[mask] = np.log(t[mask])-np.log(p[mask])
    if not np.isfinite(targets[mask]).all():
        raise ContractError('unstable positive hazard ratio')
    return array(targets), array(mask, kind='bool')
