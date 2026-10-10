"""Non-RL low-dimensional constitutive evolution corrections."""
from .correction_basis import CorrectionBasis
from .hybrid_damage_law import HybridDamageLaw, LawSpecification

__all__ = ['CorrectionBasis', 'HybridDamageLaw', 'LawSpecification']
