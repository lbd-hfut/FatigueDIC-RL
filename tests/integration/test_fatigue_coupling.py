from dataclasses import replace
import json

import numpy as np
import pytest

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.tri3_verification import tension_request
from fatigue_dic_rl.fatigue import (ConstantFatigueHazardLaw, FatigueDamageLaw, FatigueEvolutionSolver,
    FatigueLawParameters, FatigueLoadCycle, FatigueSolverConfig, FatigueStepError, FatigueStepRequest,
    LoadingMode, MaterialHistoryState, SyntheticFatigueDamageLaw, load_observations, run_trajectory)
from fatigue_dic_rl.mechanics import MechanicsConvergenceError, SolverDiagnostics
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.observations import ObservationOperator


def setup(control="load", config=None, mechanics=None):
    p = tension_request(rectangle_mesh(2, 1, width_mm=2), 2000, .3, .8, 10,
                        control=control, extension_mm=.01)
    h = MaterialHistoryState.initialize(p.quadrature_layout, p.damage, damage_max=.95)
    mapping = ObservationOperator(p.mesh).prepare([[.13, .24], [.83, .67], [3, .2]],
        coordinate_units="mm", coordinate_system="specimen")
    s = FatigueEvolutionSolver(p, h, SyntheticFatigueDamageLaw({0: FatigueLawParameters(.002, .025)}),
                              config=config, mechanics=mechanics, observation_mapping=mapping)
    c = FatigueLoadCycle(0, 10, 1, -1, -1, ("left", "right"),
        LoadingMode.LOAD if control == "load" else LoadingMode.DISPLACEMENT)
    return s, c


@pytest.mark.parametrize("control", ["load", "displacement"])
def test_final_response_refreshed_once_and_no_cycle_double_count(control):
    s, c = setup(control)
    original = s.committed_state
    r = s.advance(FatigueStepRequest(c))
    assert original.cycle_count == 0 and s.committed_state.cycle_count == 10
    row = r.substeps[-1]
    direct = s.mechanics.solve(c.request(s.pattern, s.committed_state.damage, c.N_end, "max"))
    np.testing.assert_array_equal(row.maximum.state.displacement.values, direct.state.displacement.values)
    np.testing.assert_array_equal(row.maximum.state.stress.values, direct.state.stress.values)
    np.testing.assert_array_equal(s.committed_state.accumulated_damage_increment.values,
                                  s.committed_state.damage.values-original.damage.values)
    if control == "load":
        assert row.diagnostics["max_elastic_energy_change_n_mm"] > 0
    else:
        assert row.diagnostics["max_elastic_energy_change_n_mm"] < 0
    assert row.diagnostics["max_damage_dissipation_estimate_n_mm"] > 0
    assert row.observation.valid_mask.tolist() == [True, True, False]
    with pytest.raises(FatigueStepError, match="N_start"):
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is r.committed_state


@pytest.mark.parametrize("control", ["load", "displacement"])
def test_zero_amplitude_invariance(control):
    s, c = setup(control)
    c = replace(c, load_min=1, nominal_stress_ratio=1)
    before = s.committed_state
    row = s.advance(FatigueStepRequest(c)).substeps[0]
    np.testing.assert_array_equal(row.driving_used.strain_amplitude.values, 0)
    np.testing.assert_array_equal(row.driving_used.energy_amplitude.values, 0)
    np.testing.assert_array_equal(s.committed_state.damage.values, before.damage.values)
    assert s.committed_state.cycle_count == 10


def test_subdivision_is_recomputed_and_atomic():
    s, c = setup(config=FatigueSolverConfig(max_delta_damage_per_step=.004))
    s.damage_law = ConstantFatigueHazardLaw(.002)
    r = s.advance(FatigueStepRequest(c))
    assert len(r.substeps) > 1 and r.rejected_attempts
    assert sum(row.cycle.delta_N for row in r.substeps) == 10
    assert all(row.damage_increment.values.max() <= .004 for row in r.substeps)
    np.testing.assert_allclose(s.committed_state.damage.values, .95*(-np.expm1(-.002*10)), atol=1e-15)
    assert s.committed_state.cycle_count == 10


def test_adaptive_nonlinear_hazard_matches_explicit_actual_substeps():
    s, c = setup(config=FatigueSolverConfig(max_delta_damage_per_step=.005))
    r = s.advance(FatigueStepRequest(c))
    assert len(r.substeps) >= 4
    assert r.substeps[-1].hazard_used.values.mean() > r.substeps[0].hazard_used.values.mean()
    manual, _ = setup(config=FatigueSolverConfig(max_delta_damage_per_step=.005))
    for row in r.substeps:
        manual.advance(FatigueStepRequest(row.cycle))
    np.testing.assert_array_equal(manual.committed_state.damage.values, s.committed_state.damage.values)
    coarse, _ = setup()
    coarse.advance(FatigueStepRequest(c))
    assert np.linalg.norm(s.committed_state.damage.values-coarse.committed_state.damage.values) > 1e-6


def test_changed_material_ownership_binding_rejected():
    s, c = setup()
    before = s.committed_state
    s.pattern = replace(s.pattern, mesh=replace(s.pattern.mesh,
        material_region_ids=np.ones(s.pattern.mesh.n_elements, dtype=int)),
        materials={1: s.pattern.materials[0]})
    with pytest.raises(FatigueStepError, match="binding changed"):
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is before


@pytest.mark.parametrize("configuration", [FatigueSolverConfig(max_delta_damage_per_step=.001, adaptive_subdivision=False),
    FatigueSolverConfig(max_delta_damage_per_step=.001, max_step_subdivisions=0),
    FatigueSolverConfig(max_delta_damage_per_step=.001, minimum_cycle_increment=6)])
def test_failed_increment_preserves_history(configuration):
    s, c = setup(config=configuration)
    original = s.committed_state
    with pytest.raises(FatigueStepError) as failure:
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is original
    assert failure.value.failures and s.failure_log


def test_end_solve_failure_rolls_back_after_internal_success():
    class FailLater(Tri3MechanicsSolver):
        def _solve(self, request):
            # First half can succeed; reject maximum endpoint of the second half.
            if request.cycle_count >= 7.5 and request.loading_state.phase == "max":
                raise MechanicsConvergenceError(SolverDiagnostics(False, "injected-end-failure", 1, 1, "injected"))
            return super()._solve(request)
    s, c = setup(config=FatigueSolverConfig(max_delta_damage_per_step=.01), mechanics=FailLater())
    before = s.committed_state
    with pytest.raises(FatigueStepError) as exc:
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is before
    assert exc.value.failures[-1].stage == "end_mechanics"
    assert exc.value.failures[-1].mechanics_diagnostics.converged is False
    assert before.cycle_count == 0 and np.all(before.damage.values == 0)


def test_real_underconstraint_failure_and_bad_material():
    s, c = setup()
    before = s.committed_state
    s.pattern = replace(s.pattern, displacement_bcs=())
    with pytest.raises(FatigueStepError, match="rigid-body"):
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is before
    s, c = setup()
    bad_assignment = replace(s.pattern.materials[0], parameters={"young_modulus_n_per_mm2": -1, "poisson_ratio": .3})
    s.pattern = replace(s.pattern, materials={0: bad_assignment})
    before = s.committed_state
    with pytest.raises(FatigueStepError):
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is before


@pytest.mark.parametrize("rate", [np.inf, np.nan, -1])
def test_nonfinite_and_negative_driver_rollback(rate):
    class BadLaw(FatigueDamageLaw):
        def _hazard(self, driving, cycle, state):
            return state.layout.field(np.full(state.layout.count, rate), "cycle^-1")
    s, c = setup()
    s.damage_law = BadLaw()
    before = s.committed_state
    with pytest.raises(FatigueStepError):
        s.advance(FatigueStepRequest(c))
    assert s.committed_state is before


def test_minimum_increment_rejection():
    s, c = setup()
    before = s.committed_state
    with pytest.raises(FatigueStepError, match="minimum_cycle_increment"):
        s.advance(FatigueStepRequest(replace(c, delta_N=1e-10)))
    assert s.committed_state is before


def test_trajectory_truth_observation_separation_and_reproducibility(tmp_path):
    trajectories = []
    for i in range(2):
        s, c = setup()
        t = run_trajectory(s, (c, replace(c, N_start=10)), random_seed=19, code_version="test-version")
        t.save(tmp_path/f"t{i}")
        provenance = json.loads(t.provenance_json)
        assert provenance["boundary_tags"]["right"]["facets"]
        trajectories.append(t)
    for rel in ("truth/trajectory.npz", "observations/displacement.npz"):
        with np.load(tmp_path/"t0"/rel) as a, np.load(tmp_path/"t1"/rel) as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k])
    observations = load_observations(tmp_path/"t0"/"observations")
    assert [o.cycle_count for o in observations] == [0, 10, 20]
    with np.load(tmp_path/"t0"/"observations/displacement.npz") as a:
        assert set(a.files) == {"N", "coordinates_mm", "displacement_mm", "valid_mask", "confidence"}
    with pytest.raises(FileExistsError):
        trajectories[0].save(tmp_path/"t0")
    assert not list(tmp_path.glob(".*-trial-*"))


def test_illegal_future_load_prevalidated():
    s, c = setup()
    before = s.committed_state
    with pytest.raises(ContractError, match="contiguous"):
        run_trajectory(s, (c, replace(c, N_start=11)), random_seed=19, code_version="test")
    assert s.committed_state is before


def test_storage_failure_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    s, c = setup()
    t = run_trajectory(s, (c,), random_seed=19, code_version="test")

    def fail(*args):
        raise OSError("injected storage failure")

    monkeypatch.setattr("fatigue_dic_rl.fatigue.trajectory.save_observations", fail)
    with pytest.raises(OSError, match="injected storage"):
        t.save(tmp_path/"trajectory")
    assert not (tmp_path/"trajectory").exists() and not list(tmp_path.glob(".*-trial-*"))
