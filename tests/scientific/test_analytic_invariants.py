import numpy as np
import pytest

from fatigue_dic_rl.materials import IsotropicPlaneStress
from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.geometry import MeshData
from fatigue_dic_rl.observations import ObservationOperator
from fatigue_dic_rl.physics import DamageTransition

pytestmark = pytest.mark.scientific


def test_hazard_bound_irreversibility_over_wide_scales(damage):
    rng = np.random.default_rng(20261008)
    transition = DamageTransition(0.95)
    current = damage
    for step in np.r_[0, np.logspace(-15, 15, 60)]:
        hazard = current.with_values(10**rng.uniform(-15, 15, size=current.count), units="cycle^-1")
        updated = transition.apply(current, hazard, step)
        assert np.all(updated.values >= current.values)
        assert np.all(updated.values >= 0) and np.all(updated.values <= 0.95)
        current = updated


def test_constant_hazard_semigroup_and_exact_solution(damage):
    hazard = damage.with_values([0.001, 0.02], units="cycle^-1")
    transition = DamageTransition(0.95)
    split = transition.apply(transition.apply(damage, hazard, 30), hazard, 70)
    whole = transition.apply(damage, hazard, 100)
    exact = 0.95 - (0.95-damage.values)*np.exp(-100*hazard.values)
    np.testing.assert_allclose(split.values, whole.values, rtol=0, atol=4*np.finfo(float).eps)
    np.testing.assert_allclose(whole.values, exact, rtol=0, atol=4*np.finfo(float).eps)


def test_tiny_exposure_preserves_increment(damage):
    hazard = damage.with_values([1e-20, 0], units="cycle^-1")
    result = DamageTransition().apply(damage, hazard, 1)
    assert result.values[0] > 0
    np.testing.assert_allclose(result.values[0], 1e-20, rtol=1e-15, atol=0)


def test_overflow_exposure_uses_saturation_limit(damage):
    hazard = damage.with_values([1e308, 0], units="cycle^-1")
    result = DamageTransition(0.95).apply(damage, hazard, 1e308)
    assert result.values.tolist() == [0.95, 0.2]


def test_saturated_damage_is_fixed_point(damage):
    saturated = damage.with_values([0.95, 0.95])
    result = DamageTransition(0.95).apply(saturated, damage.with_values([1, 5], units="cycle^-1"), 1e6)
    np.testing.assert_array_equal(result.values, saturated.values)


def test_plane_stress_example_stiffness_and_energy(damage):
    strain = damage.with_values([[0.001, 0, 0.002], [0.001, 0, 0.002]])
    model = IsotropicPlaneStress()
    response = model.evaluate(strain, damage, {"young_modulus_pa": 200e9, "poisson_ratio": 0.3})
    base = 200e9/(1-0.3**2)*np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 0.35]])
    np.testing.assert_allclose(response.tangent[0], base)
    np.testing.assert_allclose(response.stress.values[1], 0.8*response.stress.values[0])
    assert np.all(np.linalg.eigvalsh(response.tangent) > 0)
    energy = 0.5*np.einsum("ni,ni->n", strain.values, response.stress.values)
    assert np.all(energy >= 0)


@pytest.mark.parametrize("parameters", [{"young_modulus_pa": -1, "poisson_ratio": 0.3},
                                        {"young_modulus_pa": 200e9, "poisson_ratio": 0.5},
                                        {"young_modulus_pa": 200e9, "poisson_ratio": -1},
                                        {"young_modulus_pa": np.nan, "poisson_ratio": 0.3}, {}])
def test_invalid_constitutive_parameters(damage, parameters):
    with pytest.raises(ValueError):
        IsotropicPlaneStress().evaluate(damage.with_values(np.zeros((2, 3))), damage, parameters)


@pytest.mark.parametrize("scale", [1e-100, 1e-3, 1, 1e100])
@pytest.mark.parametrize("reverse", [False, True])
def test_triangle_shape_functions_across_scales_and_orientation(scale, reverse):
    nodes = scale*np.array([[0.2, -0.1], [1.6, 0.3], [0.1, 1.2]])
    cell = [2, 1, 0] if reverse else [0, 1, 2]
    mesh = MeshData(nodes, [cell], ["tri3"], [0], "m", "specimen")
    barycentric = np.random.default_rng(47).dirichlet([1, 1, 1], 40)
    points = barycentric @ nodes
    values = np.array([[0.3, 0.8], [-0.2, 1.4], [0.7, -0.1]])
    field = FieldData(values, "node", "m", reference=mesh.field_reference("node"))
    result = ObservationOperator(mesh).prepare(points, coordinate_units="m", coordinate_system="specimen").apply(field)
    assert result.valid_mask.all()
    np.testing.assert_allclose(result.values, barycentric @ values, rtol=0, atol=16*np.finfo(float).eps)
