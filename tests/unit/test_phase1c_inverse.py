import ast
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inverse.parameterizations import UniformDamage, GaussianDamage, LowRankDamage, Parameter, ParameterSpace
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian
from fatigue_dic_rl.inverse.identifiability import fisher_diagnostics
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter, StaticLoadCase
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.evaluation.identifiability_verification import mesh_for, pattern_for, make_case, points, GAUSSIAN_TRUTH


@pytest.mark.parametrize("count", [4, 8, 16])
def test_low_rank_damage_box_and_coordinate_sampling(count):
    model = LowRankDamage(points(count))
    space = ParameterSpace(model.parameters)
    x = points(64, "regular")
    for z in (np.ones(count), -np.ones(count), np.zeros(count)):
        d = model.evaluate(x, space.decode(z))
        assert np.all(d >= 0) and np.all(d < model.damage_max)
    np.testing.assert_allclose(model.evaluate(x[::-1], space.decode(np.ones(count))), model.evaluate(x, space.decode(np.ones(count)))[::-1])


def test_parameter_scales_roundtrip_and_fixed_metadata():
    space = ParameterSpace(GaussianDamage().parameters, {"E0": 2000.})
    z = np.linspace(-.7, .7, 6)
    np.testing.assert_allclose(space.encode(space.decode(z)), z)
    assert space.decode(z)["E0"] == 2000.
    with pytest.raises(TypeError): space.fixed["E0"] = 3000.
    with pytest.raises(ContractError): space.decode([0.]*5+[1.01])


@pytest.mark.parametrize("spec", [("bad", 1, 1, "1"), ("bad", 0, 1, "pixel"), ("", 0, 1, "1")])
def test_invalid_parameter_spec(spec):
    with pytest.raises(ContractError): Parameter(*spec)


@pytest.mark.parametrize("change", [{"amplitude": -.1}, {"sx": 0.}, {"sy": np.nan}, {"d_base": .5}])
def test_invalid_gaussian_candidate_is_rejected_not_clipped(change):
    with pytest.raises(ContractError): GaussianDamage().evaluate(points(16), dict(GAUSSIAN_TRUTH, **change))


def test_uniform_bad_damage_and_basis_invalid_global_box():
    for value in (-.1, .95, np.inf):
        with pytest.raises(ContractError): UniformDamage().evaluate(points(4), {"d0": value})
    with pytest.raises(ContractError): LowRankDamage(points(16), base=.1, coefficient_bound=.02)


def test_multiscale_fd_against_independent_analytic_function():
    def function(z): return np.array([z[0]**3+2*z[1], np.sin(z[1]), z[0]*z[1]])
    z = np.array([.2, -.3])
    s = centered_jacobian(function, z)
    exact = np.array([[3*z[0]**2, 2.], [0., np.cos(z[1])], [z[1], z[0]]])
    np.testing.assert_allclose(s.jacobian, exact, atol=3e-7)
    assert s.converged and s.absolute_error_norm > 0
    assert np.all(s.steps < 1e-3)


def test_fd_rejects_boundary_and_nonfinite_response():
    with pytest.raises(ContractError): centered_jacobian(lambda z: z, [1.])
    with pytest.raises(ContractError): centered_jacobian(lambda z: np.array([np.nan]), [0.])


def test_fisher_retains_exact_nullspace_and_does_not_invert_it():
    s = centered_jacobian(lambda z: np.array([z[0]+2*z[1], 3*z[0]+6*z[1]]), np.array([0., 0.]))
    f = fisher_diagnostics(s)
    assert f.effective_rank == 1 and np.isinf(f.condition_number)
    np.testing.assert_allclose(s.jacobian @ f.null_directions.T, 0., atol=1e-12)
    np.testing.assert_allclose(f.identifiable_covariance @ f.null_directions.T, 0., atol=1e-12)
    assert f.identifiable_covariance.shape == (2, 2)


def test_fisher_handles_more_parameters_than_observation_components():
    s = centered_jacobian(lambda z: np.array([z.sum()]), np.zeros(4))
    f = fisher_diagnostics(s)
    assert f.effective_rank == 1 and f.null_directions.shape == (3, 4)


def fixture_adapter(model=None, space=None):
    model = model or UniformDamage()
    mesh = mesh_for({"geometry": {"nx": 2, "ny": 1, "seed": 19}})
    p = pattern_for(mesh)
    physical = {"d0": .2} if isinstance(model, UniformDamage) else GAUSSIAN_TRUTH
    c = make_case(p, model, physical, points(16))
    return StaticForwardAdapter(model, space or ParameterSpace(model.parameters), [c]), physical


def test_confidence_changes_do_not_change_likelihood_weights():
    adapter, physical = fixture_adapter()
    c = adapter.cases[0]
    changed = DICMeasurement(replace(c.measurement.observation, confidence=np.ones(16)*.05), c.measurement.covariance, c.measurement.roi_mask)
    other = StaticForwardAdapter(adapter.damage_model, adapter.parameter_space, [replace(c, measurement=changed)])
    np.testing.assert_array_equal(adapter.residual([0.]), other.residual([0.]))


def test_forward_rejects_truth_in_public_pattern_and_active_outside_domain():
    adapter, physical = fixture_adapter()
    c = adapter.cases[0]; layout = c.pattern.quadrature_layout
    with pytest.raises(ContractError): StaticForwardAdapter(adapter.damage_model, adapter.parameter_space,
        [replace(c, pattern=replace(c.pattern, damage=layout.field(np.full(layout.count, .2), "1")))])
    observation = replace(c.measurement.observation, coordinates=np.full((16, 2), 3.))
    measurement = DICMeasurement(observation, c.measurement.covariance, np.ones(16, dtype=bool))
    with pytest.raises(ContractError): StaticForwardAdapter(adapter.damage_model, adapter.parameter_space, [replace(c, measurement=measurement)])


def test_unknown_E_is_removed_from_retained_public_metadata():
    adapter, _ = fixture_adapter()
    space = ParameterSpace(adapter.damage_model.parameters+(Parameter("E0", 1200., 3000., "N/mm^2"),))
    unknown = StaticForwardAdapter(adapter.damage_model, space, adapter.cases)
    assert unknown.cases[0].pattern.materials[0].parameters["young_modulus_n_per_mm2"] == 2100.
    assert unknown.cases[0].pattern.materials[0].parameters["young_modulus_n_per_mm2"] != 2000.
    p = {"d0": .2, "E0": 2000.}
    np.testing.assert_allclose(unknown.displacement_prediction(space.encode(p)), adapter.displacement_prediction(adapter.parameter_space.encode({"d0": .2})))


def test_unused_forward_parameter_is_rejected():
    adapter, _ = fixture_adapter()
    space = ParameterSpace(adapter.damage_model.parameters+(Parameter("hidden", 0., 1., "1"),))
    with pytest.raises(ContractError): StaticForwardAdapter(adapter.damage_model, space, adapter.cases)


@pytest.mark.parametrize("name,low,high,units", [("E0", 1200., 3000., "1"), ("tx", -.1, .1, "rad"),
                                               ("load_factor", -1., 1., "1"), ("theta", -.2, .2, "rad")])
def test_forward_rejects_illegal_nuisance_units_and_bounds(name, low, high, units):
    adapter, _ = fixture_adapter()
    space = ParameterSpace(adapter.damage_model.parameters+(Parameter(name, low, high, units),))
    with pytest.raises(ContractError): StaticForwardAdapter(adapter.damage_model, space, adapter.cases)


def test_prior_validation_and_bounded_multistart_recovery():
    adapter, physical = fixture_adapter()
    with pytest.raises(ContractError): InverseObjective(adapter, prior_mean=[0.])
    with pytest.raises(ContractError): InverseObjective(adapter, prior_mean=[0.], prior_standard_deviation=[0.])
    objective = InverseObjective(adapter)
    fit = invert(objective, [[-.4], [0.], [.4]])
    assert len(fit.starts) == 3 and all(r.converged for r in fit.starts)
    assert all(r.sensitivity_checked for r in fit.starts)
    assert adapter.parameter_space.decode(fit.best.normalized)["d0"] == pytest.approx(.2, abs=1e-8)
    assert all(len(r.history) > 1 for r in fit.starts)


def test_inverse_failed_solver_start_keeps_failure_and_does_not_use_truth():
    adapter, _ = fixture_adapter()
    def fail(z): raise RuntimeError("deliberate FEM failure")
    adapter.residual = fail
    fit = invert(InverseObjective(adapter), [[0.]])
    assert not fit.starts[0].converged and "deliberate FEM failure" in fit.starts[0].message
    assert "failure" in fit.starts[0].history[0]
    with pytest.raises(ContractError): fit.best


def test_actual_singular_FE_failure_preserves_diagnostics_and_public_state():
    adapter, _ = fixture_adapter()
    c = adapter.cases[0]
    singular = replace(c.pattern, displacement_bcs=())
    bad = StaticForwardAdapter(adapter.damage_model, adapter.parameter_space, [replace(c, pattern=singular)])
    fit = invert(InverseObjective(bad), [[0.]])
    assert not fit.starts[0].converged
    assert "mechanics_diagnostics" in fit.starts[0].history[0]
    assert np.count_nonzero(singular.damage.values) == 0 and singular.cycle_count == 0


def test_inverse_import_graph_has_no_truth_or_evaluation_dependency():
    root = Path(__file__).resolve().parents[2]/"src"/"fatigue_dic_rl"/"inverse"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not ({"evaluation", "truth", "networks", "rl", "training"} & set((node.module or "").split(".")))
        assert "SyntheticTruth" not in path.read_text()
