import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian
from fatigue_dic_rl.inverse.parameterizations import Parameter, ParameterSpace
from fatigue_dic_rl.inference.belief_state import GaussianPrior, MechanicsBelief, EstimateRole
from fatigue_dic_rl.inference.uncertainty import posterior_uncertainty, checked_local_jacobian, sample_gaussian_parameters
from fatigue_dic_rl.evaluation.belief_metrics import coverage_summary
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup


def test_full_rank_covariance_matches_independent_linear_algebra():
    j = np.array([[3., 1.], [-2., 4.], [1., -1.]])
    prior = GaussianPrior([0., 0.], [.7, .9])
    s = centered_jacobian(lambda z: j @ z, [0., 0.])
    u = posterior_uncertainty(s, [2., .01], prior)
    F = j.T @ j+np.diag([1/.7**2, 1/.9**2])
    np.testing.assert_allclose(u.posterior_fisher, F, atol=1e-12)
    np.testing.assert_allclose(u.covariance_normalized, np.linalg.solve(F, np.eye(2)), rtol=1e-12)
    np.testing.assert_allclose(u.covariance_physical, np.diag([2., .01]) @ np.linalg.solve(F, np.diag([2., .01])), rtol=1e-12)
    assert u.data_rank == u.posterior_rank == 2


def test_nullspace_has_no_full_covariance_or_fictitious_finite_marginals():
    j = np.array([[1., 2.], [2., 4.]])
    u = posterior_uncertainty(centered_jacobian(lambda z: j @ z, [0., 0.]), [1., 2.])
    assert u.data_rank == u.posterior_rank == 1
    assert u.covariance_normalized is None and u.covariance_physical is None
    assert np.isinf(u.marginal_standard_deviation_physical).all()
    assert u.posterior_null_directions.shape == (1, 2)
    np.testing.assert_allclose(j @ u.posterior_null_directions.T, 0., atol=1e-12)
    assert not u.reliable_for_propagation


def test_prior_closes_nullspace_without_increasing_data_rank():
    s = centered_jacobian(lambda z: np.array([z[0]+z[1]]), [0., 0.])
    u = posterior_uncertainty(s, [1., 1.], GaussianPrior([0., 0.], [.1, .1]))
    assert u.data_rank == 1 and u.posterior_rank == 2
    assert u.covariance_normalized is not None
    assert "data_rank_deficient_prior_is_not_DIC_information" in u.warnings


def test_empty_data_prior_and_rectangular_jacobian_are_stable():
    u = posterior_uncertainty(None, [1., 1., 1.])
    assert u.data_rank == u.posterior_rank == 0
    assert u.posterior_null_directions.shape == (3, 3)
    u = posterior_uncertainty(None, [1., 1., 1.], GaussianPrior([0.]*3, [.1]*3))
    assert u.data_rank == 0 and u.posterior_rank == 3
    np.testing.assert_allclose(u.covariance_normalized, np.eye(3)*.01)


@pytest.mark.parametrize("bad", [([0.], [0.]), ([2.], [1.]), ([0.], [np.nan]), ([0., 0.], [1.])])
def test_invalid_gaussian_prior(bad):
    with pytest.raises(ContractError): GaussianPrior(*bad)


def test_active_bound_derivative_is_checked_and_laplace_is_flagged():
    s, method = checked_local_jacobian(lambda z: np.array([z[0]**3, 2*z[0]]), [1.])
    assert s.converged and "one_sided" in method
    np.testing.assert_allclose(s.jacobian[:, 0], [3., 2.], atol=1e-6)
    u = posterior_uncertainty(s, [1.], normalized=np.ones(1), derivative_method=method)
    assert u.data_rank == 1 and u.status == "unreliable_local_approximation"
    assert "active_parameter_bound_Laplace_unreliable" in u.warnings


def test_nonconvergence_and_boundary_mass_do_not_pass_as_reliable_covariance():
    s = centered_jacobian(lambda z: z, [0.])
    for arguments in ({"converged": False}, {"normalized": [.95]}):
        u = posterior_uncertainty(s, [1.], **arguments)
        assert not u.reliable_for_propagation


def test_unchecked_derivative_cannot_supply_full_covariance():
    from dataclasses import replace
    s = replace(centered_jacobian(lambda z: 10*z, [0.]), converged=False)
    u = posterior_uncertainty(s, [1.], GaussianPrior([0.], [.1]))
    assert not u.reliable_for_propagation and u.covariance_normalized is None


def test_fixed_linearization_information_increment_is_psd():
    rng = np.random.default_rng(101)
    rows = rng.normal(size=(5, 4, 3))
    old = np.zeros((3, 3))
    for end in range(1, 6):
        j = rows[:end].reshape(-1, 3)
        u = posterior_uncertainty(centered_jacobian(lambda z: j @ z, np.zeros(3)), np.ones(3))
        assert np.linalg.eigvalsh(u.data_fisher-old).min() > -1e-12
        old = u.data_fisher


def test_analytic_linear_gaussian_coverage_uses_independent_seeds():
    rng = np.random.default_rng(819)
    j = np.array([[3., 1.], [2., -4.], [1., 2.]])
    c = np.linalg.solve(j.T @ j, np.eye(2))
    errors = rng.normal(size=(1000, 3)) @ j @ c
    for i in range(2):
        stats = coverage_summary(errors[:, i], np.full(1000, np.sqrt(c[i, i])))
        assert .92 < stats["coverage"] < .98
        assert stats["valid_count"] == 1000


def test_sampling_is_deterministic_and_rejects_rank_deficient_belief():
    _, space, _, predictor, _ = uniform_setup({"geometry": {"nx": 2, "ny": 1, "seed": 19}})
    damage = predictor.damage_states([0.], [0.])[0.].damage
    s = centered_jacobian(lambda z: 100*z, [0.])
    u = posterior_uncertainty(s, space.scales, normalized=[0.])
    b = MechanicsBelief(space.names, [0.], [.38], u, damage, 0., EstimateRole.FILTERED, (), None, "{}")
    a = sample_gaussian_parameters(b, space, count=20, seed=12)
    other = sample_gaussian_parameters(b, space, count=20, seed=12)
    np.testing.assert_array_equal(a.normalized, other.normalized)
    assert not a.normalized.flags.writeable
    b = MechanicsBelief(space.names, [0.], [.38], posterior_uncertainty(None, space.scales), damage,
                       0., EstimateRole.PRIOR, (), None, "{}")
    with pytest.raises(ContractError): sample_gaussian_parameters(b, space, count=20, seed=12)
