"""Immutable quadrature material history. No resampling or in-place updates."""

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.geometry import QuadratureLayout


class HistoryStatus(str, Enum):
    COMMITTED = "committed"
    TRIAL = "trial"


@dataclass(frozen=True, eq=False)
class MaterialHistoryState:
    layout: QuadratureLayout
    damage: FieldData
    accumulated_damage_increment: FieldData
    accumulated_cycles: FieldData
    history: FieldData
    material_ids: np.ndarray
    quadrature_ids: np.ndarray
    material_state_indices: np.ndarray
    damage_max: float
    history_names: tuple[str, ...] = ()
    status: HistoryStatus = HistoryStatus.COMMITTED

    def __post_init__(self) -> None:
        if not isinstance(self.layout, QuadratureLayout):
            raise ContractError("history requires a QuadratureLayout")
        cap = scalar(self.damage_max, "damage_max", minimum=0)
        if not 0 < cap < 1:
            raise ContractError("fatigue history requires 0 < damage_max < 1")
        object.__setattr__(self, "damage_max", cap)
        object.__setattr__(self, "status", HistoryStatus(self.status))
        for value, units in ((self.damage, "1"), (self.accumulated_damage_increment, "1"),
                             (self.accumulated_cycles, "cycle")):
            self.layout.assert_field(value)
            value.require(units=units, components=(), fully_valid=True)
        self.layout.assert_field(self.history)
        self.history.require(units="1", fully_valid=True)
        names = tuple(self.history_names)
        if (self.history.values.shape != (self.layout.count, len(names)) or
                len(set(names)) != len(names) or any(not isinstance(n, str) or not n.strip() for n in names)):
            raise ContractError("history requires [G,H] and unique history_names; first version is dimensionless")
        object.__setattr__(self, "history_names", names)
        for name in ("material_ids", "quadrature_ids", "material_state_indices"):
            value = array(getattr(self, name), kind="int", name=name)
            if value.shape != (self.layout.count,):
                raise ContractError("history IDs must have shape [G]")
            object.__setattr__(self, name, value)
        identity = np.arange(self.layout.count)
        if (not np.array_equal(self.layout.material_state_indices, identity) or
                not np.array_equal(self.material_state_indices, identity)):
            raise NotImplementedError("fatigue supports only identity material-state mapping")
        if not np.array_equal(self.quadrature_ids, identity):
            raise ContractError("quadrature IDs must agree with layout order")
        regions = self.layout.mesh.material_region_ids[self.layout.quadrature_to_element]
        if not np.array_equal(self.material_ids, regions):
            raise ContractError("material IDs disagree with quadrature ownership and mesh regions")
        d = self.damage.values
        acc = self.accumulated_damage_increment.values
        cycles = self.accumulated_cycles.values
        if (np.any(d < 0) or np.any(d >= cap) or np.any(acc < 0) or np.any(acc > d) or
                np.any(cycles < 0) or not np.all(cycles == cycles[0])):
            raise ContractError("inadmissible damage/history/cycles; require 0 <= D < Dmax and common N")

    @property
    def cycle_count(self) -> float:
        return float(self.accumulated_cycles.values[0])

    @classmethod
    def initialize(cls, layout: QuadratureLayout, damage: FieldData, *, damage_max: float,
                   cycle_count: float = 0, history: FieldData | None = None,
                   history_names: tuple[str, ...] = ()) -> "MaterialHistoryState":
        N = scalar(cycle_count, "cycle_count", minimum=0)
        return cls(layout, damage, layout.field(np.zeros(layout.count), "1"),
                   layout.field(np.full(layout.count, N), "cycle"),
                   layout.field(np.empty((layout.count, 0)), "1") if history is None else history,
                   layout.mesh.material_region_ids[layout.quadrature_to_element], np.arange(layout.count),
                   layout.material_state_indices, damage_max, history_names)

    def trial(self, new_damage: FieldData, delta_N: float) -> "MaterialHistoryState":
        self.layout.assert_field(new_damage)
        self.damage.assert_compatible(new_damage)
        dt = scalar(delta_N, "delta_N", minimum=0)
        with np.errstate(over="raise", invalid="raise"):
            N = self.cycle_count + dt
        if dt <= 0 or not np.isfinite(N) or N <= self.cycle_count:
            raise ContractError("cycle increment must be positive, finite and representable")
        increment = new_damage.values-self.damage.values
        if np.any(increment < 0):
            raise ContractError("damage irreversibility violated")
        return replace(self, damage=new_damage,
                       accumulated_damage_increment=self.accumulated_damage_increment.with_values(
                           self.accumulated_damage_increment.values+increment),
                       accumulated_cycles=self.accumulated_cycles.with_values(np.full(self.layout.count, N)),
                       status=HistoryStatus.TRIAL)

    def commit(self) -> "MaterialHistoryState":
        if self.status != HistoryStatus.TRIAL:
            raise ContractError("only a validated trial may be committed")
        return replace(self, status=HistoryStatus.COMMITTED)

