"""Future latent policy action affects estimated evolution, never specimen truth."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fatigue_dic_rl.core import FieldData, MechanicsState, StateRole
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.observations import DICObservation


@dataclass(frozen=True, eq=False)
class LatentAction:
    values: np.ndarray
    dimension: int = 8

    def __post_init__(self) -> None:
        values = array(self.values, name="latent action")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            raise ContractError("latent dimension must be a positive integer")
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise ContractError("latent action must be a finite [K] vector")
        object.__setattr__(self, "values", values)


class DamageCorrectionPolicy(ABC):
    def action(self, belief: MechanicsState, observation: DICObservation) -> LatentAction:
        if belief.role != StateRole.BELIEF:
            raise ContractError("policy accepts an estimated belief, never ground truth")
        if not isinstance(observation, DICObservation):
            raise ContractError("policy observations must be DICObservation")
        result = self._action(belief, observation)
        if not isinstance(result, LatentAction):
            raise ContractError("policy must return LatentAction")
        return result

    @abstractmethod
    def _action(self, belief: MechanicsState, observation: DICObservation) -> LatentAction:
        """A future policy estimates a correction from a mechanics belief and DIC."""


class ExponentialHazardCorrection:
    """lambda_hybrid = lambda_physics * exp(alpha * tanh(B @ a)).

    B rows must be explicitly collocated with hazard. The FieldData basis
    contract prevents equal-size node/element/quadrature spaces from mixing.
    No policy, Actor or Critic is instantiated by this algebraic adapter.
    """

    def apply(self, physics_hazard: FieldData, action: LatentAction,
              basis: FieldData, *, alpha: float) -> FieldData:
        physics_hazard.require(units="cycle^-1", components=(), fully_valid=True)
        basis.require(units="1", components=(action.dimension,), fully_valid=True)
        physics_hazard.assert_same_support(basis)
        if np.any(physics_hazard.values < 0):
            raise ContractError("physics hazard must be nonnegative")
        scale = scalar(alpha, "alpha", minimum=0)
        if scale > np.log(np.finfo(float).max):
            raise ContractError("alpha exceeds the finite exponential range")
        try:
            with np.errstate(over="raise", invalid="raise"):
                multiplier = np.exp(scale * np.tanh(basis.values @ action.values))
                result = physics_hazard.values * multiplier
        except FloatingPointError as exc:
            raise ContractError("hazard correction overflow; scale basis/action/alpha explicitly") from exc
        return physics_hazard.with_values(result)
