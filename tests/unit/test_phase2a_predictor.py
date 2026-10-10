from dataclasses import replace
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inverse.fatigue_adapter import FatigueForwardAdapter
from fatigue_dic_rl.inverse.parameterizations import Parameter
from fatigue_dic_rl.inference.belief_state import ObservationBatch, ObservationFrame, EstimateRole
from fatigue_dic_rl.inference.physics_predictor import PhysicsPredictor, PhysicsForwardAdapter
from fatigue_dic_rl.inference.sequential_map import PhysicsBasedBeliefEstimator
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup, generate_truth, observation_series, fatigue_loading


@pytest.fixture
def setup():
    return uniform_setup({"geometry": {"nx": 2, "ny": 1, "seed": 19}}, fatigue=True)


def test_actual_phase1b_truth_and_phase1c_forward_predictions_agree(setup):
    model, space, pattern, predictor, x = setup
    actual = {"d0": .12, "A": .0015}
    truths = generate_truth(pattern, model, actual, [0., 10., 20.], x, load_history=fatigue_loading(pattern))
    frames = observation_series(pattern, truths, x, sigma_mm=0.)
    new = PhysicsForwardAdapter(predictor, ObservationBatch(frames))
    old = FatigueForwardAdapter(model, space, [f.case for f in frames])
    z = space.encode(actual)
    np.testing.assert_allclose(new.displacement_prediction(z), old.displacement_prediction(z), atol=1e-15)
    for a, t in zip(new.predict(z), truths):
        np.testing.assert_allclose(a[0], t.displacement_dic_mm, atol=1e-15)


def test_prediction_recomputes_fem_with_new_damage(setup):
    model, space, pattern, predictor, x = setup
    truths = generate_truth(pattern, model, {"d0": .12, "A": .0015}, [0., 10.], x, load_history=fatigue_loading(pattern))
    frames = observation_series(pattern, truths, x, sigma_mm=0.)
    z = space.encode({"d0": .12, "A": .0015})
    observed = []
    original = predictor.mechanics.solve
    def record(request):
        if request.cycle_count == 10.:
            observed.append(request.damage.values.copy())
        return original(request)
    predictor.mechanics.solve = record
    states = predictor.damage_states(z, [0., 10.])
    at10 = predictor.observe(z, frames[1].design, states[10.])
    np.testing.assert_allclose(at10.displacement_mm, truths[1].displacement_dic_mm)
    np.testing.assert_array_equal(observed[-1], states[10.].damage.values)
    assert np.linalg.norm(at10.displacement_mm-truths[0].displacement_dic_mm) > 1e-5


def test_candidates_and_off_grid_queries_do_not_change_other_trajectories(setup):
    _, space, _, predictor, _ = setup
    za, zb = space.encode({"d0": .12, "A": .0015}), space.encode({"d0": .25, "A": .002})
    a = predictor.damage_states(za, [20.])[20.].damage.values
    predictor.damage_states(zb, [20.])
    another = predictor.damage_states(za, [1.5, 7.2, 20.])[20.].damage.values
    np.testing.assert_array_equal(a, another)
    fresh = PhysicsPredictor(predictor.damage_model, space, predictor.pattern, load_history=predictor.load_history)
    np.testing.assert_array_equal(a, fresh.damage_states(za, [20.])[20.].damage.values)


def test_explicit_history_from_another_candidate_is_rejected(setup):
    model, space, pattern, predictor, x = setup
    za, zb = space.encode({"d0": .12, "A": .0015}), space.encode({"d0": .25, "A": .002})
    truth = generate_truth(pattern, model, {"d0": .12, "A": .0015}, [10.], x, load_history=fatigue_loading(pattern))
    design = observation_series(pattern, truth, x, sigma_mm=0.)[0].design
    wrong = predictor.damage_states(zb, [10.])[10.]
    with pytest.raises(ContractError, match="parameter candidate"): predictor.observe(za, design, wrong)


def test_failed_fatigue_rollout_is_not_cached_and_does_not_poison_replay(setup):
    _, space, _, predictor, _ = setup
    z = space.encode({"d0": .12, "A": .0015})
    before = predictor.damage_states(z, [0.])[0.]
    original = predictor.mechanics.solve
    def fail(request):
        if request.cycle_count >= 6.:
            raise RuntimeError("injected endpoint FEM failure")
        return original(request)
    predictor.mechanics.solve = fail
    with pytest.raises(Exception, match="rolled back"): predictor.damage_states(z, [10.])
    assert predictor.failure_diagnostics and before.cycle_count == 0
    assert not any(key[1] == (10.,) for key in predictor._history_cache)
    predictor.mechanics.solve = original
    repaired = predictor.damage_states(z, [10.])[10.]
    fresh = PhysicsPredictor(predictor.damage_model, space, predictor.pattern, load_history=predictor.load_history)
    np.testing.assert_array_equal(repaired.damage.values, fresh.damage_states(z, [10.])[10.].damage.values)


def test_unknown_material_nominal_is_bounds_derived_and_shared_registration_is_explicit():
    setup = uniform_setup({"geometry": {"nx": 2, "ny": 1, "seed": 19}},
                          extra_parameters=(Parameter("E0", 1200., 3000., "N/mm^2"),))
    predictor = setup[3]
    assert predictor.pattern.materials[0].parameters["young_modulus_n_per_mm2"] == 2100.
    with pytest.raises(ContractError, match="shared"): PhysicsBasedBeliefEstimator(predictor, noise_contract="independent_given_shared_registration")


@pytest.mark.parametrize("target", [-1., 41., np.inf])
def test_prediction_rejects_unsupported_time(setup, target):
    predictor = setup[3]
    with pytest.raises(ContractError): predictor.damage_states([0., 0.], [target])


def test_public_model_cannot_carry_hidden_damage(setup):
    model, space, pattern, _, _ = setup
    layout = pattern.quadrature_layout
    hidden = replace(pattern, damage=layout.field(np.full(layout.count, .1), "1"))
    with pytest.raises(ContractError): PhysicsPredictor(model, space, hidden, load_history=fatigue_loading(pattern))
