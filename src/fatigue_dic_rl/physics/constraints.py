"""Hard admissibility checks; invalid physical inputs are never repaired."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fatigue_dic_rl.core import FieldData, FieldLocation
from fatigue_dic_rl.core.validation import ContractError, scalar


class PhysicsConstraint(ABC):
    @abstractmethod
    def validate(self, previous: FieldData, proposed: FieldData) -> None:
        """Raise ContractError for a hard physical constraint violation."""


@dataclass(frozen=True)
class DamageAdmissibility(PhysicsConstraint):
    dmax: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dmax", scalar(self.dmax, "dmax", minimum=0))
        if not 0 < self.dmax <= 1:
            raise ContractError("dmax must satisfy 0 < dmax <= 1")

    def check_field(self, damage: FieldData) -> None:
        damage.require(units="1", components=(), fully_valid=True)
        if damage.location == FieldLocation.OBSERVATION:
            raise ContractError("damage evolution requires a mechanics space, not DIC observation points")
        if np.any(damage.values < 0) or np.any(damage.values > self.dmax):
            raise ContractError("damage outside [0, dmax]")

    def validate(self, previous: FieldData, proposed: FieldData) -> None:
        previous.assert_compatible(proposed)
        self.check_field(previous)
        self.check_field(proposed)
        if np.any(proposed.values < previous.values):
            raise ContractError("irreversible damage decreased")
