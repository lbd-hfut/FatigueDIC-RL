from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import StateRole
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inference import StateEstimator
from fatigue_dic_rl.observations import DICObservation
from fatigue_dic_rl.rl import DamageCorrectionPolicy, ExponentialHazardCorrection, LatentAction


def observation(cycle=0):
    return DICObservation([[0.3, 0.2]], [[0, 0]], [True], [1], cycle, "peak", "load-1", "specimen", "mm", "mm")


def test_positive_bounded_hazard_correction(damage):
    hazard = damage.with_values([0, 0.02], units="cycle^-1")
    basis = damage.with_values(np.ones((2, 8)))
    action = LatentAction(np.ones(8))
    result = ExponentialHazardCorrection().apply(hazard, action, basis, alpha=0.5)
    assert result.values[0] == 0
    assert hazard.values[1]*np.exp(-0.5) <= result.values[1] <= hazard.values[1]*np.exp(0.5)
    np.testing.assert_array_equal(ExponentialHazardCorrection().apply(hazard, action, basis, alpha=0).values, hazard.values)


def test_correction_shape_units_support_and_scaling(damage):
    hazard = damage.with_values([0.01, 0.02], units="cycle^-1")
    basis = damage.with_values(np.ones((2, 8)))
    with pytest.raises(ContractError):
        LatentAction(np.zeros(7))
    for alpha in (-1, np.inf, 710):
        with pytest.raises(ContractError):
            ExponentialHazardCorrection().apply(hazard, LatentAction(np.zeros(8)), basis, alpha=alpha)
    with pytest.raises(ContractError):
        ExponentialHazardCorrection().apply(hazard, LatentAction(np.zeros(8)), damage.with_values(np.zeros((2, 7))), alpha=1)


def test_policy_rejects_truth(state):
    class ZeroPolicy(DamageCorrectionPolicy):
        def _action(self, belief, observation):
            return LatentAction(np.zeros(8))

    assert ZeroPolicy().action(state, observation()).dimension == 8
    with pytest.raises(ContractError):
        ZeroPolicy().action(replace(state, role=StateRole.GROUND_TRUTH), observation())


def test_estimator_role_history_contract(state, mesh):
    class ContractDouble(StateEstimator):
        def _update(self, prior, observations, mesh):
            return replace(state, cycle_count=observations[-1].cycle_count)

    estimator = ContractDouble()
    assert estimator.update(state, [observation(1), observation(2)], mesh).cycle_count == 2
    with pytest.raises(ContractError):
        estimator.update(replace(state, role=StateRole.GROUND_TRUTH), [observation(1)], mesh)
    with pytest.raises(ContractError):
        estimator.update(state, [observation(2), observation(1)], mesh)
    with pytest.raises(ContractError):
        estimator.update(state, [], mesh)
