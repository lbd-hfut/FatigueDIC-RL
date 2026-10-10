"""Declared synthetic reference laws; no empirical material calibration claim."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.fatigue.models import FatigueDamageLaw, SyntheticFatigueDamageLaw, FatigueLawParameters

FAMILIES = ('matched', 'parameter', 'S1', 'S2', 'S3', 'S4', 'S5')


def reference_hazard(x, damage, coordinates_mm, R, *, family, A=8e-5, m=1., strength=3., lengths_mm=(2., 1.)):
    x, d, xy = np.asarray(x, float), np.asarray(damage, float), np.asarray(coordinates_mm, float)
    if family not in FAMILIES or x.shape != d.shape or xy.shape != (len(x), 2) or not np.isfinite([A,m,strength,R]).all() or A < 0 or m <= 0 or strength < 0 or not np.isfinite(x).all() or not np.isfinite(d).all() or not np.isfinite(xy).all() or np.any(x < 0) or np.any((d < 0) | (d >= .95)) or R > 1 or not np.isfinite(lengths_mm).all() or min(lengths_mm) <= 0:
        raise ContractError('invalid reference law or quadrature inputs')
    base = A*x**m
    if family == 'S1': base *= 1+strength*x/(1+x)
    elif family == 'S2': base = A*np.maximum(x-strength, 0.)**m
    elif family == 'S3': base *= np.exp(strength*(R+1))
    elif family == 'S4': base *= np.exp(strength*d)
    elif family == 'S5': base *= np.exp(strength*np.cos(np.pi*xy[:, 0]/lengths_mm[0]))
    if not np.isfinite(base).all() or np.any(base < 0):
        raise ContractError('reference hazard is nonfinite or negative')
    return base


@dataclass(frozen=True)
class ReferenceLawSpecification:
    family: str = 'matched'
    A: float = 8e-5
    m: float = 1.
    strength: float = 3.
    lengths_mm: tuple = (2., 1.)
    dimension: int = 0

    def __post_init__(self):
        reference_hazard(np.array([0.,1.]),np.array([0.,.1]),np.zeros((2,2)),-1.,family=self.family,A=self.A,m=self.m,strength=self.strength,lengths_mm=self.lengths_mm)

    def build(self, physical, material_ids):
        return ReferenceFamilyLaw(self)


class ReferenceFamilyLaw(FatigueDamageLaw):
    def __init__(self, specification): self.specification = specification
    def _hazard(self, driving, cycle, state):
        s = self.specification
        values = reference_hazard(driving.energy_amplitude.values/.025, state.damage.values, state.layout.coordinates, cycle.nominal_stress_ratio, family=s.family,A=s.A,m=s.m,strength=s.strength,lengths_mm=s.lengths_mm)
        return state.layout.field(values, 'cycle^-1')
