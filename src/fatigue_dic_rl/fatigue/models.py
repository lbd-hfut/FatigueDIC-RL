"""Amplitude driver and a calibratable synthetic hazard law, not a literature law."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError, scalar
from fatigue_dic_rl.mechanics.tri3 import Tri3SolverResult
from .history import MaterialHistoryState
from .loading import FatigueLoadCycle


@dataclass(frozen=True)
class FatigueDrivingForceResult:
    strain_amplitude: FieldData
    strain_mean: FieldData
    energy_amplitude: FieldData
    Y_max: FieldData
    Y_min: FieldData


class FatigueDrivingForce(ABC):
    def evaluate(self, minimum: Tri3SolverResult, maximum: Tri3SolverResult) -> FatigueDrivingForceResult:
        a, b = minimum.state, maximum.state
        a.damage.assert_compatible(b.damage)
        if (a.cycle_count != b.cycle_count or not np.array_equal(a.damage.values, b.damage.values) or
                a.loading_state.load_id != b.loading_state.load_id or
                a.loading_state.phase != "min" or b.loading_state.phase != "max"):
            raise ContractError("cycle endpoints must share damage, N, load ID and declared min/max phases")
        layout = maximum.assembly.quadrature_layout
        if minimum.assembly.quadrature_layout.reference != layout.reference:
            raise ContractError("endpoint quadrature layouts differ")
        for state in (a, b):
            layout.assert_field(state.strain)
            layout.assert_field(state.damage)
            state.strain.require(units="1", components=(3,), fully_valid=True)
            if state.strain.metadata.get("shear_convention") != "engineering":
                raise ContractError("fatigue driver requires explicit engineering shear convention")
        result = self._evaluate(minimum, maximum)
        for f in (result.strain_amplitude, result.strain_mean):
            layout.assert_field(f)
            f.require(units="1", components=(3,), fully_valid=True)
        for f in (result.energy_amplitude, result.Y_max, result.Y_min):
            layout.assert_field(f)
            f.require(units="N/mm^2", components=(), fully_valid=True)
            if np.any(f.values < 0):
                raise ContractError("fatigue energy driver must be nonnegative")
        return result

    @abstractmethod
    def _evaluate(self, minimum, maximum) -> FatigueDrivingForceResult:
        pass


class EnergyAmplitudeDriver(FatigueDrivingForce):
    """Ya = 1/2 eps_a^T C0 eps_a, using engineering gamma_xy.

    The supported Tri3 scalar elastic model gives C0 = C(D)/(1-D).
    This is an amplitude-model input, not the full fatigue loss per cycle.
    """

    def _evaluate(self, minimum, maximum) -> FatigueDrivingForceResult:
        layout = maximum.assembly.quadrature_layout
        lo, hi = minimum.state.strain.values, maximum.state.strain.values
        if not np.array_equal(minimum.assembly.tangents, maximum.assembly.tangents):
            raise ContractError("endpoints need identical frozen elastic tangents")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            C0 = maximum.assembly.tangents/(1-maximum.state.damage.values[:, None, None])
            amplitude, mean = (hi-lo)/2, (hi+lo)/2

            def energy(eps):
                return layout.field(0.5*np.einsum("qi,qij,qj->q", eps, C0, eps), "N/mm^2")

            return FatigueDrivingForceResult(
                layout.field(amplitude, "1", metadata={"shear_convention": "engineering"}),
                layout.field(mean, "1", metadata={"shear_convention": "engineering"}),
                energy(amplitude), energy(hi), energy(lo))


class FatigueDamageLaw(ABC):
    def hazard(self, driving: FatigueDrivingForceResult, cycle: FatigueLoadCycle,
               state: MaterialHistoryState) -> FieldData:
        state.layout.assert_field(driving.energy_amplitude)
        driving.energy_amplitude.require(units="N/mm^2", components=(), fully_valid=True)
        if np.any(driving.energy_amplitude.values < 0):
            raise ContractError("negative amplitude energy")
        result = self._hazard(driving, cycle, state)
        state.layout.assert_field(result)
        result.require(units="cycle^-1", components=(), fully_valid=True)
        if np.any(result.values < 0):
            raise ContractError("negative fatigue hazard")
        return result

    @abstractmethod
    def _hazard(self, driving, cycle, state) -> FieldData:
        pass


@dataclass(frozen=True)
class FatigueLawParameters:
    A_per_cycle: float
    Y_ref_n_per_mm2: float
    eta: float = 0
    m: float = 1
    beta_R: float = 0

    def __post_init__(self) -> None:
        for name in ("A_per_cycle", "Y_ref_n_per_mm2", "eta", "m", "beta_R"):
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if self.A_per_cycle < 0 or self.Y_ref_n_per_mm2 <= 0 or self.eta < 0 or self.m <= 0:
            raise ContractError("A >= 0 [cycle^-1], Yref > 0 [N/mm^2], eta >= 0 and m > 0 are required")


class SyntheticFatigueDamageLaw(FatigueDamageLaw):
    """lambda=A*max(Ya/Yref-eta,0)^m*exp(beta_R*R).

    beta_R may be signed: it enters a nonnegative exponential correction.
    Parameters must be calibrated before any real-material interpretation.
    """

    def __init__(self, parameters: Mapping[int, FatigueLawParameters]) -> None:
        if (not isinstance(parameters, Mapping) or not parameters or
                any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0 or
                    not isinstance(v, FatigueLawParameters) for k, v in parameters.items())):
            raise ContractError("fatigue parameters require material-region IDs and FatigueLawParameters")
        self.parameters = MappingProxyType(dict(parameters))

    def _hazard(self, driving, cycle, state) -> FieldData:
        values = np.zeros(state.layout.count)
        if set(np.unique(state.material_ids))-set(self.parameters):
            raise ContractError("fatigue law missing a material region")
        for region, p in self.parameters.items():
            ids = np.flatnonzero(state.material_ids == region)
            if not len(ids) or p.A_per_cycle == 0:
                continue
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    x = np.maximum(driving.energy_amplitude.values[ids]/p.Y_ref_n_per_mm2-p.eta, 0)
                    active = x > 0
                    # Log evaluation avoids intermediate overflows when the final hazard is finite.
                    log_rate = np.log(p.A_per_cycle)+p.m*np.log(x[active])+p.beta_R*cycle.nominal_stress_ratio
                    local = np.zeros(len(ids))
                    local[active] = np.exp(log_rate)
                    values[ids] = local
            except FloatingPointError as exc:
                raise ContractError("nonfinite fatigue hazard; check parameters and amplitude") from exc
        return state.layout.field(values, "cycle^-1")


class ConstantFatigueHazardLaw(FatigueDamageLaw):
    """Verification-only constant hazard, deliberately independent of amplitude."""

    def __init__(self, hazard_per_cycle: float) -> None:
        self.hazard_per_cycle = scalar(hazard_per_cycle, "hazard_per_cycle", minimum=0)

    def _hazard(self, driving, cycle, state) -> FieldData:
        return state.layout.field(np.full(state.layout.count, self.hazard_per_cycle), "cycle^-1")
