"""Analytic evolution for a hazard held constant over one cycle interval."""

import numpy as np

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError, scalar

from .constraints import DamageAdmissibility


class DamageTransition:
    def __init__(self, dmax: float = 1.0) -> None:
        self.constraint = DamageAdmissibility(dmax)

    @property
    def dmax(self) -> float:
        return self.constraint.dmax

    def apply(self, damage: FieldData, hazard: FieldData, delta_cycles: float) -> FieldData:
        """D_new = D + (Dmax-D)*(-expm1(-delta_cycles*hazard)).

        Updates the damage field only. A mechanics solve is required to refresh
        displacement/strain/stress after changing damage; this method never
        pretends that a previous response is the updated equilibrium response.
        """
        self.constraint.check_field(damage)
        hazard.require(units="cycle^-1", components=(), fully_valid=True)
        damage.assert_same_support(hazard)
        dt = scalar(delta_cycles, "delta_cycles", minimum=0)
        if np.any(hazard.values < 0):
            raise ContractError("hazard must be nonnegative")
        # Finite huge inputs may yield +inf exposure: its exponential limit is 0.
        with np.errstate(over="ignore"):
            fraction = -np.expm1(-dt * hazard.values)
        values = damage.values + (self.dmax - damage.values) * fraction
        # Bound only floating-point roundoff after validating all physical inputs.
        values = np.minimum(self.dmax, np.maximum(damage.values, values))
        result = damage.with_values(values)
        self.constraint.validate(damage, result)
        return result
