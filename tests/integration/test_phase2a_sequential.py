from dataclasses import replace
import json
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inverse.fatigue_adapter import FatigueForwardAdapter
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from fatigue_dic_rl.inference.belief_state import ObservationBatch, EstimateRole, GaussianPrior
from fatigue_dic_rl.inference.sequential_map import PhysicsBasedBeliefEstimator, SequentialMAPConfig
from fatigue_dic_rl.inference.physics_predictor import PhysicsPredictor
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup, generate_truth, observation_series, fatigue_loading


@pytest.fixture(scope="module")
def experiment():
    model, space, pattern, predictor, x = uniform_setup({"geometry": {"nx": 2, "ny": 1, "seed": 19}}, fatigue=True)
    actual = {"d0": .12, "A": .0015}
    truths = generate_truth(pattern, model, actual, [0., 10., 20., 30., 40.], x, load_history=fatigue_loading(pattern))
    frames = observation_series(pattern, truths, x, sigma_mm=0.)
    return model, space, pattern, predictor, truths, frames, actual


@pytest.mark.scientific
def test_noiseless_sequential_recovery_rank_transition_and_strict_forecast(experiment):
    _, space, _, predictor, truths, fs, actual = experiment
    est = PhysicsBasedBeliefEstimator(predictor, initializations=[np.zeros(2)], config=SequentialMAPConfig(assimilation_cutoff=20.))
    for i, frame in enumerate(fs[:3]):
        prior_ids = est.observed_ids
        result = est.assimilate_observation(frame)
        assert result.status == "CONVERGED"
        assert result.predictions_before_update[0].role == EstimateRole.PREDICTION
        assert result.predictions_before_update[0].conditioned_observation_ids == prior_ids
        assert est.belief.role == EstimateRole.FILTERED
        assert est.belief.uncertainty.data_rank == (1 if i == 0 else 2)
        if i: np.testing.assert_allclose(est.belief.normalized_mean, space.encode(actual), atol=1e-6)
        np.testing.assert_allclose(est.belief.current_damage.values, truths[i].damage, atol=1e-8)
    future = est.forecast_future([f.design for f in fs[3:]], propagate_uncertainty=False)
    for p, t in zip(future.predictions, truths[3:]):
        assert p.role == EstimateRole.FORECAST and p.conditioned_through_cycle == 20.
        np.testing.assert_allclose(p.damage.values, t.damage, atol=1e-8)
        np.testing.assert_allclose(p.observations[0].displacement_mm, t.displacement_dic_mm, atol=1e-10)
    before = est.belief
    with pytest.raises(ContractError, match="future"): est.assimilate_observation(fs[3])
    assert est.belief is before


@pytest.mark.scientific
def test_prefix_is_equivalent_to_unmodified_phase1c_batch_and_warm_cold_costs(experiment):
    model, space, _, predictor, _, fs, _ = experiment
    batch = invert(InverseObjective(FatigueForwardAdapter(model, space, [f.case for f in fs[:3]])), [np.zeros(2)])
    cold = PhysicsBasedBeliefEstimator(predictor, initializations=[np.zeros(2)], config=SequentialMAPConfig(warm_start=False))
    warm = PhysicsBasedBeliefEstimator(predictor, initializations=[np.zeros(2)])
    for f in fs[:3]:
        cold.assimilate_observation(f); warm.assimilate_observation(f)
    np.testing.assert_allclose(warm.belief.normalized_mean, batch.best.normalized, atol=1e-6)
    assert abs(cold.attempts[-1].selected.cost-warm.attempts[-1].selected.cost) < 1e-12
    assert len(warm.attempts) == 3 and all(r.selected.history for r in warm.attempts)


def test_original_prior_is_used_once_per_complete_prefix(experiment):
    _, space, _, predictor, _, fs, _ = experiment
    prior = GaussianPrior([0., 0.], [.2, .2])
    seq = PhysicsBasedBeliefEstimator(predictor, prior=prior, initializations=[np.zeros(2)])
    for f in fs[:2]: seq.assimilate_observation(f)
    direct = PhysicsBasedBeliefEstimator(predictor, prior=prior, initializations=[np.zeros(2)])
    direct.assimilate_observation(ObservationBatch(fs[:2]))
    np.testing.assert_allclose(seq.belief.normalized_mean, direct.belief.normalized_mean, atol=1e-6)
    np.testing.assert_allclose(seq.belief.uncertainty.prior_fisher, prior.precision, rtol=1e-15)
    provenance = json.loads(seq.belief.provenance_json)
    assert provenance["fixed_prior_terms"] == 1 and provenance["likelihood_terms"] == 2
    assert not provenance["previous_posterior_as_prior"]


def test_real_fe_failure_rolls_back_belief_and_preserves_failed_start_diagnostics():
    model, space, pattern, predictor, x = uniform_setup({"geometry": {"nx": 2, "ny": 1, "seed": 19}})
    truths = generate_truth(pattern, model, {"d0": .2}, [0.], x)
    f = observation_series(pattern, truths, x, sigma_mm=0.)[0]
    est = PhysicsBasedBeliefEstimator(predictor, initializations=[[0.]])
    est.assimilate_observation(f)
    before = est.belief
    singular = replace(f, observation_id="failed", case=replace(f.case, pattern=replace(pattern, displacement_bcs=())))
    result = est.assimilate_observation(singular)
    assert result.status == "FAILED_ROLLED_BACK" and result.filtered_belief is None
    assert est.belief is before and est.observed_ids == (f.observation_id,)
    assert not result.optimization.starts[0].converged
    assert "mechanics_diagnostics" in result.optimization.starts[0].history[0]


def test_unconverged_result_is_tentative_and_boundary_laplace_is_flagged():
    from fatigue_dic_rl.evaluation.sequential_verification import run_numerical_stability
    result = run_numerical_stability({"geometry": {"nx": 2, "ny": 1, "seed": 19}})
    assert result["status"] == "PASSED"
    assert result["rows"][1]["status"] == "UNCONVERGED_TENTATIVE"
