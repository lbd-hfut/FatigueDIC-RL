from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import CycleLoading, FieldReference
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.materials import ConstantHazardLaw, DamageLaw, DamageLawRegistry
from fatigue_dic_rl.physics import DamageAdmissibility, DamageTransition


def test_law_and_registry(state):
    registry = DamageLawRegistry()
    registry.register("constant", ConstantHazardLaw())
    result = registry.get("constant").hazard(state, {"hazard_per_cycle": 0.01}, CycleLoading(3, state.loading_state))
    assert result.units == "cycle^-1"
    assert result.values.tolist() == [0.01, 0.01]
    with pytest.raises(ContractError):
        registry.register("constant", ConstantHazardLaw())
    with pytest.raises(ContractError):
        registry.get("unknown")


@pytest.mark.parametrize("rate", [-1, np.nan, np.inf, True, "0.1"])
def test_law_rejects_bad_parameters(state, rate):
    with pytest.raises(ContractError):
        ConstantHazardLaw().hazard(state, {"hazard_per_cycle": rate}, CycleLoading(1, state.loading_state))


def test_law_rejects_missing_and_negative_custom_driver(state):
    with pytest.raises(ContractError):
        ConstantHazardLaw().hazard(state, {}, CycleLoading(1, state.loading_state))

    class BadLaw(DamageLaw):
        def _hazard(self, state, material_parameters, loading):
            return state.damage.with_values([-1, 0], units="cycle^-1")

    with pytest.raises(ContractError):
        BadLaw().hazard(state, {}, CycleLoading(1, state.loading_state))


def test_zero_hazard_and_zero_step_exact(damage):
    transition = DamageTransition()
    np.testing.assert_array_equal(transition.apply(damage, damage.with_values([0, 0], units="cycle^-1"), 1e10).values,
                                  damage.values)
    np.testing.assert_array_equal(transition.apply(damage, damage.with_values([3, 4], units="cycle^-1"), 0).values,
                                  damage.values)


@pytest.mark.parametrize("dmax", [0, -1, 1.01, np.nan, np.inf])
def test_invalid_dmax(dmax):
    with pytest.raises(ContractError):
        DamageTransition(dmax)


@pytest.mark.parametrize("step", [-1, np.nan, np.inf, True])
def test_transition_rejects_step(damage, step):
    with pytest.raises(ContractError):
        DamageTransition().apply(damage, damage.with_values([1, 1], units="cycle^-1"), step)


def test_bad_hazard_units_support_and_damage(damage):
    transition = DamageTransition(0.9)
    for bad in (damage.with_values([-1, 0], units="cycle^-1"), damage.with_values([1, 2]),
                damage.with_values([[1, 2], [1, 2]], units="cycle^-1")):
        with pytest.raises(ContractError):
            transition.apply(damage, bad, 1)
    with pytest.raises(ContractError):
        transition.apply(damage.with_values([0, 0.95]), damage.with_values([1, 1], units="cycle^-1"), 1)
    with pytest.raises(ContractError):
        DamageAdmissibility().validate(damage, damage.with_values([0, 0.1]))


def test_transition_deterministic(damage):
    hazard = damage.with_values([0.01, 0.2], units="cycle^-1")
    first = DamageTransition().apply(damage, hazard, 31)
    second = DamageTransition().apply(damage, hazard, 31)
    np.testing.assert_array_equal(first.values, second.values)


def test_invalid_mask_rejected(damage):
    bad = replace(damage, valid_mask=[True, False])
    with pytest.raises(ContractError):
        DamageTransition().apply(bad, bad.with_values([1, 1], units="cycle^-1"), 1)


def test_observation_support_cannot_evolve_damage(damage):
    dic_damage = replace(damage, location="observation",
                         reference=FieldReference("dic", "observation", 2))
    with pytest.raises(ContractError):
        DamageTransition().apply(dic_damage, dic_damage.with_values([1, 1], units="cycle^-1"), 1)
