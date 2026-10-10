from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.tri3_verification import tension_request
from fatigue_dic_rl.fatigue import (EnergyAmplitudeDriver, FatigueEvolutionSolver, FatigueLawParameters,
    FatigueLoadCycle, FatigueSolverConfig, FatigueStepRequest, HistoryStatus, LoadingMode,
    MaterialHistoryState, SyntheticFatigueDamageLaw)
from fatigue_dic_rl.geometry import BoundaryTag, MeshData, QuadratureLayout
from fatigue_dic_rl.geometry.triangles import tri3_shape_functions
from fatigue_dic_rl.mechanics import DisplacementBC, NodalForceBC
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverRequest


@pytest.fixture
def pattern():
    return tension_request(rectangle_mesh(2, 1, width_mm=2), 2000, .3, .8, 10)


@pytest.fixture
def cycle():
    return FatigueLoadCycle(0, 10, 1, -1, -1, ("left", "right"), LoadingMode.LOAD)


def test_history_copies_and_binds_passive_variables(pattern):
    layout = pattern.quadrature_layout
    source = np.full(layout.count, .1)
    h = MaterialHistoryState.initialize(layout, layout.field(source, "1"), damage_max=.95,
        history=layout.field(np.ones((layout.count, 1)), "1"), history_names=("passive_marker",))
    source[:] = .8
    trial = h.trial(h.damage.with_values(np.full(layout.count, .2)), 7)
    assert h.cycle_count == 0 and trial.cycle_count == 7
    assert trial.status == HistoryStatus.TRIAL and trial.commit().status == HistoryStatus.COMMITTED
    np.testing.assert_array_equal(h.damage.values, .1)
    np.testing.assert_allclose(trial.accumulated_damage_increment.values, .1)
    np.testing.assert_array_equal(h.history.values, trial.history.values)
    for value in (h.damage.values, h.history.values, h.material_ids, h.accumulated_cycles.values):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(ContractError):
        h.commit()


@pytest.mark.parametrize("kind", ["element", "node", "other_layout"])
def test_history_refuses_location_or_reference_alias(pattern, kind):
    layout = pattern.quadrature_layout
    if kind == "other_layout":
        wrong = replace(layout, rule_name="different-rule").field(np.zeros(layout.count), "1")
    else:
        wrong = FieldData(np.zeros(pattern.mesh.field_reference(kind).count), kind, "1",
                          reference=pattern.mesh.field_reference(kind))
    with pytest.raises(ContractError):
        MaterialHistoryState.initialize(layout, wrong, damage_max=.95)


def test_history_rejects_permuted_state_and_region_ids(pattern):
    layout = pattern.quadrature_layout
    permuted = replace(layout, material_state_indices=np.arange(layout.count)[::-1])
    with pytest.raises(NotImplementedError, match="identity"):
        MaterialHistoryState.initialize(permuted, permuted.field(np.zeros(layout.count), "1"), damage_max=.95)
    h = MaterialHistoryState.initialize(layout, pattern.damage, damage_max=.95)
    with pytest.raises(ContractError, match="material IDs"):
        replace(h, material_ids=np.ones(layout.count, dtype=int))
    with pytest.raises(ContractError, match="quadrature IDs"):
        replace(h, quadrature_ids=np.arange(layout.count)[::-1])
    with pytest.raises(ContractError):
        replace(h, accumulated_cycles=layout.field(np.arange(layout.count), "cycle"))


@pytest.mark.parametrize("cap", [0, 1, -.1, np.nan, np.inf])
def test_strict_history_bounds(pattern, cap):
    with pytest.raises(ContractError):
        MaterialHistoryState.initialize(pattern.quadrature_layout, pattern.damage, damage_max=cap)


def test_history_rejects_saturation_and_unrepresentable_cycles(pattern):
    l = pattern.quadrature_layout
    h = MaterialHistoryState.initialize(l, pattern.damage, damage_max=.95, cycle_count=1e20)
    with pytest.raises(ContractError):
        h.trial(h.damage, 1)
    with pytest.raises(ContractError):
        h.trial(l.field(np.full(l.count, .95), "1"), 1e10)


@pytest.mark.parametrize("updates", [{"delta_N": 0}, {"delta_N": -1}, {"delta_N": np.inf},
    {"nominal_stress_ratio": 0}, {"load_max": 0}, {"load_min": 2}, {"N_start": -1},
    {"amplitude_units": "N"}, {"boundary_condition_refs": ()}, {"load_max": True}])
def test_invalid_cycle(cycle, updates):
    with pytest.raises((ContractError, ValueError)):
        replace(cycle, **updates)


def test_nonproportional_and_mixed_control_rejected(pattern, cycle):
    with pytest.raises(NotImplementedError):
        replace(cycle, loading_path="multiaxial-loop")
    with pytest.raises(NotImplementedError):
        replace(cycle, loading_mode=LoadingMode.DISPLACEMENT).request(pattern, pattern.damage, 0, "max")
    mixed = replace(pattern, displacement_bcs=pattern.displacement_bcs+(
        DisplacementBC([1], [0], [.001], "mm"),))
    with pytest.raises(NotImplementedError):
        cycle.request(mixed, pattern.damage, 0, "max")
    with pytest.raises(ContractError):
        replace(cycle, boundary_condition_refs=("absent",)).request(pattern, pattern.damage, 0, "max")


@pytest.mark.parametrize("params", [{"A_per_cycle": -1}, {"Y_ref_n_per_mm2": 0}, {"eta": -.1},
    {"m": 0}, {"m": -1}, {"beta_R": np.nan}, {"A_per_cycle": True}])
def test_illegal_law_parameters(params):
    defaults = dict(A_per_cycle=.001, Y_ref_n_per_mm2=.025)
    defaults.update(params)
    with pytest.raises(ContractError):
        FatigueLawParameters(**defaults)


def endpoints(pattern, cycle):
    solver = Tri3MechanicsSolver()
    return tuple(solver.solve(cycle.request(pattern, pattern.damage, 0, phase)) for phase in ("min", "max"))


def test_driver_reversed_energy_and_engineering_shear(pattern, cycle):
    lo, hi = endpoints(pattern, cycle)
    driver = EnergyAmplitudeDriver().evaluate(lo, hi)
    np.testing.assert_allclose(driver.Y_max.values, driver.Y_min.values, atol=1e-15)
    np.testing.assert_allclose(driver.energy_amplitude.values, .025, atol=1e-14)
    np.testing.assert_allclose(driver.strain_mean.values, 0, atol=1e-15)
    # Independently prescribed pure shear: gamma_a=.004, G=E/[2(1+nu)].
    ids = np.arange(pattern.mesh.n_nodes)
    u = np.column_stack((.004*pattern.mesh.nodes[:, 1], np.zeros(len(ids))))
    shear = replace(pattern, traction_bcs=(), displacement_bcs=(
        DisplacementBC(np.repeat(ids, 2), np.tile([0, 1], len(ids)), u.ravel(), "mm"),))
    shear_cycle = replace(cycle, loading_mode=LoadingMode.DISPLACEMENT)
    shear_driver = EnergyAmplitudeDriver().evaluate(*endpoints(shear, shear_cycle))
    np.testing.assert_allclose(shear_driver.energy_amplitude.values, .5*(2000/(2*1.3))*.004**2, atol=1e-14)


def test_zero_threshold_mean_correction_and_nonfinite_rejection(pattern, cycle):
    h = MaterialHistoryState.initialize(pattern.quadrature_layout, pattern.damage, damage_max=.95)
    driver = EnergyAmplitudeDriver().evaluate(*endpoints(pattern, cycle))
    law = SyntheticFatigueDamageLaw({0: FatigueLawParameters(.001, .025, beta_R=.4)})
    rate = law.hazard(driver, cycle, h)
    np.testing.assert_allclose(rate.values, .001*np.exp(-.4), rtol=1e-12)
    inactive = replace(driver, energy_amplitude=driver.energy_amplitude.with_values(np.zeros(h.layout.count)))
    np.testing.assert_array_equal(law.hazard(inactive, cycle, h).values, 0)
    threshold = SyntheticFatigueDamageLaw({0: FatigueLawParameters(.001, .025, eta=2)})
    np.testing.assert_array_equal(threshold.hazard(driver, cycle, h).values, 0)
    with pytest.raises(ContractError, match="nonfinite"):
        SyntheticFatigueDamageLaw({0: FatigueLawParameters(.001, .025, beta_R=-1e5)}).hazard(driver, cycle, h)
    with pytest.raises(ContractError, match="missing"):
        SyntheticFatigueDamageLaw({1: FatigueLawParameters(.001, .025)}).hazard(driver, cycle, h)
    with pytest.raises(ContractError):
        law.hazard(replace(driver, energy_amplitude=driver.energy_amplitude.with_values(
            np.ones(h.layout.count), units="Pa")), cycle, h)


def test_solver_refuses_layout_alias(pattern):
    copied = QuadratureLayout.tri3_centroid(pattern.mesh)
    h = MaterialHistoryState.initialize(copied, copied.field(np.zeros(copied.count), "1"), damage_max=.95)
    with pytest.raises(ContractError, match="exact QuadratureLayout"):
        FatigueEvolutionSolver(pattern, h, SyntheticFatigueDamageLaw({0: FatigueLawParameters(.001, .025)}))


@pytest.mark.parametrize("updates", [{"max_delta_damage_per_step": 0}, {"max_delta_damage_per_step": 1},
    {"max_step_subdivisions": -1}, {"max_step_subdivisions": 1.5}, {"max_step_subdivisions": True},
    {"minimum_cycle_increment": 0}, {"minimum_cycle_increment": np.inf}, {"adaptive_subdivision": 1},
    {"energy_identity_rtol": .1}])
def test_illegal_evolution_configuration(updates):
    with pytest.raises(ContractError):
        FatigueSolverConfig(**updates)


def test_multiple_quadrature_states_and_passive_history_evolve_independently(pattern, cycle):
    mesh = MeshData([[0, 0], [2, 0], [0, 1]], [[0, 1, 2]], ["tri3"], [0], "mm", "specimen",
        boundary_tags={"bottom": BoundaryTag([0, 1], [[0, 1]])}, mesh_id="fatigue-multi-q")
    points = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    layout = QuadratureLayout(mesh, "three-point", points, np.full(3, 1/6), tri3_shape_functions(points) @ mesh.nodes,
        np.tile([[2, 0], [0, 1]], (3, 1, 1)), np.full(3, 2), np.zeros(3, dtype=int), (np.arange(3),), np.arange(3))
    damage = layout.field([.1, .2, .3], "1")
    p = Tri3SolverRequest.from_mm(mesh, pattern.materials, pattern.loading_state, thickness_mm=.8,
        quadrature_layout=layout, damage=damage,
        displacement_bcs=(DisplacementBC([0, 0, 1], [0, 1, 1], [0, 0, 0], "mm"),),
        nodal_force_bcs=(NodalForceBC([2], [[1, 0]]),))
    h = MaterialHistoryState.initialize(layout, damage, damage_max=.95,
        history=layout.field([[1], [2], [3]], "1"), history_names=("passive_marker",))
    solver = FatigueEvolutionSolver(p, h, SyntheticFatigueDamageLaw({0: FatigueLawParameters(.001, .025)}))
    row = solver.advance(FatigueStepRequest(replace(cycle, boundary_condition_refs=("bottom",)))).substeps[-1]
    assert row.maximum.state.stress.values.shape == (3, 3) and mesh.n_elements == 1
    assert np.all(row.damage_increment.values > 0)
    assert np.ptp(row.damage_increment.values) > 0
    np.testing.assert_array_equal(row.state_after.history.values, [[1], [2], [3]])


def test_region_hazard_mapping_is_used_in_actual_coupling(pattern, cycle):
    mesh = replace(pattern.mesh, material_region_ids=np.array([0, 1, 0, 1]))
    p = tension_request(pattern.mesh, 2000, .3, .8, 10, control="displacement", extension_mm=.01)
    layout = QuadratureLayout.tri3_centroid(mesh)
    p = replace(p, mesh=mesh, quadrature_layout=layout, damage=layout.field(np.zeros(layout.count), "1"),
        materials={0: pattern.materials[0], 1: pattern.materials[0]})
    h = MaterialHistoryState.initialize(p.quadrature_layout, p.damage, damage_max=.95)
    law = SyntheticFatigueDamageLaw({0: FatigueLawParameters(.001, .025), 1: FatigueLawParameters(.002, .025)})
    solver = FatigueEvolutionSolver(p, h, law)
    row = solver.advance(FatigueStepRequest(replace(cycle, loading_mode=LoadingMode.DISPLACEMENT))).substeps[0]
    np.testing.assert_allclose(row.hazard_used.values[h.material_ids == 0], .001, rtol=1e-12)
    np.testing.assert_allclose(row.hazard_used.values[h.material_ids == 1], .002, rtol=1e-12)
    assert row.state_after.damage.values[h.material_ids == 1].mean() > row.state_after.damage.values[h.material_ids == 0].mean()
