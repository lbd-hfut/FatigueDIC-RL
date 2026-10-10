"""History-selected, bounded gate; original transition/solver remain unchanged."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from .hybrid_damage_law import LawSpecification


@dataclass(frozen=True)
class GatedLawSpecification:
    gate: float = 1.
    dimension: int = 2
    basis_kind: str = 'energy'
    alpha: float = 2.
    def __post_init__(self):
        if not np.isfinite(self.gate) or not 0 <= self.gate <= 1 or not np.isfinite(self.alpha) or not 0 <= self.alpha <= 10:
            raise ContractError('correction gate must be finite in [0,1]')
        LawSpecification(mode='hybrid',dimension=self.dimension,basis_kind=self.basis_kind,alpha=self.alpha)
    def build(self, physical, material_ids):
        return LawSpecification(mode='hybrid',dimension=self.dimension,basis_kind=self.basis_kind,alpha=self.alpha*self.gate).build(physical,material_ids)
