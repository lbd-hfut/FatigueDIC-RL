"""Physical hazard models without an RL dependency."""

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import CycleLoading, FieldData, MechanicsState
from fatigue_dic_rl.core.validation import ContractError, label, scalar


class DamageLaw(ABC):
    """Produce a nonnegative scalar hazard [cycle^-1] on the damage support.

    Loading describes the proposed evolution interval; it need not equal the
    loading phase of the already computed response. Backend-specific history
    variables and parameter semantics belong to each concrete law.
    """

    def hazard(self, state: MechanicsState, material_parameters: Mapping[str, float],
               loading: CycleLoading) -> FieldData:
        if not isinstance(loading, CycleLoading):
            raise ContractError("damage law needs CycleLoading")
        result = self._hazard(state, material_parameters, loading)
        result.require(units="cycle^-1", components=(), fully_valid=True)
        state.damage.assert_same_support(result)
        if np.any(result.values < 0):
            raise ContractError("damage law returned a negative hazard")
        return result

    @abstractmethod
    def _hazard(self, state: MechanicsState, material_parameters: Mapping[str, float],
                loading: CycleLoading) -> FieldData:
        """Implement the physical driver, never the discrete damage update."""


class ConstantHazardLaw(DamageLaw):
    """Minimal interface example; not a calibrated fatigue material model.

    Parameters: hazard_per_cycle >= 0. Regional/history-dependent laws can
    replace this law without changing the transition or observation operator.
    """

    def _hazard(self, state: MechanicsState, material_parameters: Mapping[str, float],
                loading: CycleLoading) -> FieldData:
        if "hazard_per_cycle" not in material_parameters:
            raise ContractError("missing material parameter hazard_per_cycle")
        rate = scalar(material_parameters["hazard_per_cycle"], "hazard_per_cycle", minimum=0)
        return state.damage.with_values(np.full(state.damage.count, rate), units="cycle^-1")


class DamageLawRegistry:
    """Explicit per-instance registry; no import-time global plugin side effects."""

    def __init__(self) -> None:
        self._laws: dict[str, DamageLaw] = {}

    def register(self, name: str, law: DamageLaw) -> None:
        label(name, "damage law name")
        if not isinstance(law, DamageLaw) or name in self._laws:
            raise ContractError("registration needs a DamageLaw and an unused name")
        self._laws[name] = law

    def get(self, name: str) -> DamageLaw:
        if name not in self._laws:
            raise ContractError(f"unregistered damage law: {name}")
        return self._laws[name]
