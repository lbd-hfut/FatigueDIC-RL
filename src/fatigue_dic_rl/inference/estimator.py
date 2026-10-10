"""Belief update contract; no estimator is claimed to identify damage in Phase 0."""

from abc import ABC, abstractmethod
from typing import Sequence

from fatigue_dic_rl.core import MechanicsState, StateRole
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import MeshData
from fatigue_dic_rl.observations import DICObservation


class StateEstimator(ABC):
    def update(self, prior: MechanicsState | None, observations: Sequence[DICObservation],
               mesh: MeshData) -> MechanicsState:
        if prior is not None and (prior.role != StateRole.BELIEF or prior.mesh_id != mesh.mesh_id):
            raise ContractError("estimator prior must be a belief on the target mesh")
        history = tuple(observations)
        if not history or not all(isinstance(o, DICObservation) for o in history):
            raise ContractError("estimator needs a nonempty DIC history")
        if any(a.cycle_count > b.cycle_count for a, b in zip(history, history[1:])):
            raise ContractError("DIC history must be chronological")
        if prior is not None and history[0].cycle_count < prior.cycle_count:
            raise ContractError("DIC update history precedes the prior belief")
        if any(o.coordinate_system != mesh.coordinate_system or o.coordinate_units != mesh.coordinate_units
               for o in history):
            raise ContractError("estimator needs observations registered to the mesh frame/units")
        result = self._update(prior, history, mesh)
        if (not isinstance(result, MechanicsState) or result.role != StateRole.BELIEF or
                result.mesh_id != mesh.mesh_id or result.cycle_count != history[-1].cycle_count):
            raise ContractError("estimator must return a belief at the latest observation cycle")
        return result

    @abstractmethod
    def _update(self, prior: MechanicsState | None, observations: tuple[DICObservation, ...],
                mesh: MeshData) -> MechanicsState:
        """Assimilate displacement observations; do not equate this to damage evolution."""
