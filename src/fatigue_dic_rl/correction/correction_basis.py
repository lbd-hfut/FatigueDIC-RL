"""Smooth dimensionless bases; coefficients are global, not independent G actions."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array


@dataclass(frozen=True)
class CorrectionBasis:
    dimension: int = 2
    kind: str = 'energy'
    origin_mm: tuple = (0., 0.)
    lengths_mm: tuple = (2., 1.)

    def __post_init__(self):
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension not in (2, 4, 8):
            raise ContractError('correction dimension must be 2, 4 or 8')
        if self.kind not in ('energy', 'spatial', 'global_diagnostic'):
            raise ContractError('unknown correction basis')
        if len(self.origin_mm) != 2 or len(self.lengths_mm) != 2 or not np.isfinite([self.origin_mm, self.lengths_mm]).all() or min(self.lengths_mm) <= 0:
            raise ContractError('basis needs physical mm origin and positive reference lengths')
        object.__setattr__(self, 'origin_mm', tuple(self.origin_mm))
        object.__setattr__(self, 'lengths_mm', tuple(self.lengths_mm))

    def evaluate(self, x, damage, coordinates_mm, stress_ratio=0.):
        x, d, coords = map(np.asarray, (x, damage, coordinates_mm))
        if x.ndim != 1 or d.shape != x.shape or coords.shape != (len(x), 2) or not np.isfinite(x).all() or not np.isfinite(d).all() or not np.isfinite(coords).all() or np.any(x < 0) or np.any(d < 0) or np.any(d >= 1) or not np.isfinite(stress_ratio):
            raise ContractError('basis requires finite dimensionless x/D/R and [G,2] mm coordinates')
        # Stable saturation avoids x+1 overflow for extreme finite x.
        t = 1.-1./(1.+x)
        if self.kind == 'energy':
            return array(np.column_stack([t**(j+1) for j in range(self.dimension)]))
        if self.kind == 'global_diagnostic':
            logarithm = np.zeros(len(x))
            np.log(x, out=logarithm, where=x > 0)
            fields = [np.ones(len(x)), logarithm]+[t**j for j in range(1, self.dimension-1)]
            return array(np.column_stack(fields))
        q = (coords-np.asarray(self.origin_mm))/np.asarray(self.lengths_mm)
        spatial = [np.ones(len(x)), np.cos(np.pi*q[:, 0]), np.cos(np.pi*q[:, 1]),
                   np.cos(np.pi*q[:, 0])*np.cos(np.pi*q[:, 1])]
        return array(np.column_stack([t**(1+j//4)*spatial[j % 4] for j in range(self.dimension)]))
