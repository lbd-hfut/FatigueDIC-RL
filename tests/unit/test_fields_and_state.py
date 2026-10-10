from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import (CycleLoading, FieldData, FieldLocation, FieldReference,
                                 LoadingState, MechanicsState, StateRole)
from fatigue_dic_rl.core.validation import ContractError


@pytest.mark.parametrize("updates", [{"values": 0.1}, {"values": [0, 1, 2]},
                                     {"valid_mask": [1, 1]}, {"valid_mask": [True]},
                                     {"values": [0, np.nan]}, {"units": ""},
                                     {"location": FieldLocation.NODE}])
def test_field_shape_location_validation(damage, updates):
    with pytest.raises(ContractError):
        replace(damage, **updates)


def test_support_and_unit_mixing_rejected(damage):
    with pytest.raises(ContractError):
        damage.assert_compatible(damage.with_values(damage.values, units="MPa"))
    other = replace(damage, reference=FieldReference("other:element", FieldLocation.ELEMENT, 2, "other"))
    with pytest.raises(ContractError):
        damage.assert_same_support(other)
    nodes = FieldData([0, 0.2], "node", "1", reference=FieldReference("node-space", "node", 2, "test-mesh"))
    with pytest.raises(ContractError):
        damage.assert_same_support(nodes)


def test_each_location_and_independent_counts(mesh):
    for location, count in (("node", 4), ("element", 2), ("quadrature", 6), ("observation", 9)):
        field = FieldData(np.zeros(count), location, "1", reference=FieldReference(location, location, count, mesh.mesh_id))
        assert field.count == count


def test_coordinates_only_and_invalid_values():
    field = FieldData([[0, 0], [np.nan, np.nan]], "observation", "mm", coordinates=[[0.1, 0.3], [5, 8]],
                     coordinate_units="mm", coordinate_system="specimen", valid_mask=[True, False])
    assert np.isnan(field.values[1]).all()
    with pytest.raises(ContractError):
        replace(field, coordinate_units=None)
    with pytest.raises(ContractError):
        replace(field, coordinates=None, coordinate_units=None, coordinate_system=None)
    with pytest.raises(ContractError):
        field.assert_same_support(replace(field, coordinate_system="camera"))


@pytest.mark.parametrize("updates", [{"cycle_count": -1}, {"cycle_count": np.inf},
                                     {"mesh_id": "other"}, {"loading_state": "not-a-load"}])
def test_state_invalid_contract(state, updates):
    with pytest.raises(ContractError):
        replace(state, **updates)


def test_state_damage_shape_bounds_units_and_observation_separation(state, damage):
    for bad in (damage.with_values([-0.1, 0.2]), damage.with_values([0.1, 1.1]),
                damage.with_values([[0, 0], [0, 0]]), damage.with_values([0, 0], units="mm"),
                replace(damage, location="observation", reference=FieldReference("dic", "observation", 2, state.mesh_id))):
        with pytest.raises(ContractError):
            replace(state, damage=bad)


def test_mechanics_state_units_and_roles(state, nodal_displacement, mesh, damage):
    strain = damage.with_values(np.zeros((2, 3)))
    stress = strain.with_values(np.zeros((2, 3)), units="Pa")
    full = replace(state, displacement=nodal_displacement, strain=strain, stress=stress,
                   internal_variables={"plastic_strain": strain})
    assert full.role == StateRole.BELIEF
    assert replace(full, role=StateRole.GROUND_TRUTH).role == StateRole.GROUND_TRUTH
    for updates in ({"strain": strain.with_values(strain.values, units="Pa")},
                    {"stress": stress.with_values(stress.values, units="1")},
                    {"displacement": nodal_displacement.with_values(nodal_displacement.values, units="px")},
                    {"internal_variables": {"damage_truth": [1, 2]}}):
        with pytest.raises(ContractError):
            replace(full, **updates)


@pytest.mark.parametrize("step", [-1, np.nan, np.inf, True, "1"])
def test_invalid_cycle_steps(step):
    with pytest.raises(ContractError):
        CycleLoading(step, LoadingState("load", "peak"))


def test_cycle_frequency_validation():
    with pytest.raises(ContractError):
        CycleLoading(1, LoadingState("load", "peak"), frequency_hz=0)
