"""Physical truth, estimated belief and predicted response have explicit roles."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .fields import FieldData, FieldLocation
from .validation import ContractError, label, scalar


class StateRole(str, Enum):
    GROUND_TRUTH = "ground_truth"
    BELIEF = "belief"
    PREDICTION = "prediction"


@dataclass(frozen=True)
class LoadingState:
    load_id: str
    phase: str

    def __post_init__(self) -> None:
        label(self.load_id, "load_id")
        label(self.phase, "phase")


@dataclass(frozen=True)
class CycleLoading:
    delta_cycles: float
    loading_state: LoadingState
    frequency_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_cycles", scalar(self.delta_cycles, "delta_cycles", minimum=0))
        if not isinstance(self.loading_state, LoadingState):
            raise ContractError("loading_state must be a LoadingState")
        if self.frequency_hz is not None:
            frequency = scalar(self.frequency_hz, "frequency_hz", minimum=0)
            if frequency == 0:
                raise ContractError("frequency_hz must be positive")
            object.__setattr__(self, "frequency_hz", frequency)


@dataclass(frozen=True, eq=False)
class MechanicsState:
    mesh_id: str
    role: StateRole
    damage: FieldData
    cycle_count: float
    loading_state: LoadingState
    displacement: FieldData | None = None
    strain: FieldData | None = None
    stress: FieldData | None = None
    internal_variables: Mapping[str, FieldData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label(self.mesh_id, "mesh_id")
        object.__setattr__(self, "role", StateRole(self.role))
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "cycle_count", minimum=0))
        if not isinstance(self.loading_state, LoadingState):
            raise ContractError("loading_state must be a LoadingState")
        self.damage.require(units="1", components=(), fully_valid=True)
        if self.damage.location == FieldLocation.OBSERVATION:
            raise ContractError("mechanics damage cannot be stored at DIC observation points")
        if np.any(self.damage.values < 0) or np.any(self.damage.values > 1):
            raise ContractError("damage must satisfy 0 <= D <= 1")
        fields = [self.damage]
        if self.displacement is not None:
            self.displacement.require(location=FieldLocation.NODE, components=(2,), fully_valid=True)
            if self.displacement.units not in {"m", "mm"}:
                raise ContractError("displacement units must be m or mm")
            fields.append(self.displacement)
        for name, value in (("strain", self.strain), ("stress", self.stress)):
            if value is not None:
                value.require(fully_valid=True)
                if value.location == FieldLocation.OBSERVATION or value.values.shape[1:] not in {(3,), (2, 2)}:
                    raise ContractError(f"{name} needs a mechanics location and 2D tensor components")
                if name == "strain" and value.units != "1":
                    raise ContractError("strain units must be 1")
                if name == "stress" and value.units not in {"Pa", "kPa", "MPa", "GPa", "N/mm^2"}:
                    raise ContractError("stress needs explicitly declared stress units")
                fields.append(value)
        if not isinstance(self.internal_variables, Mapping):
            raise ContractError("internal_variables must be a field mapping")
        for name, value in self.internal_variables.items():
            label(name, "internal variable name")
            if not isinstance(value, FieldData) or value.location == FieldLocation.OBSERVATION:
                raise ContractError("internal variables must be mechanics FieldData")
            fields.append(value)
        for value in fields:
            if value.reference is None or value.reference.mesh_id != self.mesh_id:
                raise ContractError("mechanics fields need an explicit reference to this mesh")
        object.__setattr__(self, "internal_variables", MappingProxyType(dict(self.internal_variables)))
