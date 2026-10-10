from dataclasses import replace
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup,generate_truth,observation_series,fatigue_loading
from fatigue_dic_rl.inference.neural_belief_estimator import NeuralBeliefEstimator
from fatigue_dic_rl.inference.belief_state import ObservationBatch,ObservationFrame,EstimateRole
from fatigue_dic_rl.networks.temporal_gru import GraphRecurrentEstimator
from fatigue_dic_rl.networks.graph_features import build_graph_frame


def setup():
    torch.set_num_threads(1);torch.manual_seed(9);c={'geometry':{'nx':2,'ny':1,'seed':19},'observation':{'count':16}}
    model,space,pattern,predictor,x=uniform_setup(c,fatigue=True);truths=generate_truth(pattern,model,{'d0':.12,'A':.0015},[0.,10.,20.,30.,40.],x,load_history=fatigue_loading(pattern));frames=observation_series(pattern,truths,x,sigma_mm=5e-5)
    return predictor,frames,NeuralBeliefEstimator(predictor,GraphRecurrentEstimator())


def test_protocol_roles_physics_consistency_and_time_holdout():
    p,f,e=setup();r=e.assimilate_observation(ObservationBatch(f[:3]));assert r.status=='SUCCESS';assert e.belief.cycle_count==20.
    assert r.predictions_before_update[-1].role==EstimateRole.PREDICTION
    np.testing.assert_allclose(r.innovations_before_update[-1].displacement_residual_mm,(f[2].case.measurement.observation.displacement-r.predictions_before_update[-1].observations[0].displacement_mm).ravel())
    state=p.damage_states(e.belief.normalized_mean,[20.])[20.];np.testing.assert_array_equal(state.damage.values,e.belief.current_damage.values)
    before=e.belief;forecast=e.forecast_future([a.design for a in f[3:]],propagate_uncertainty=False)
    assert e.belief is before and all(a.role==EstimateRole.FORECAST for a in forecast.predictions)
    with pytest.raises(ContractError):e.assimilate_observation(f[3])
    with pytest.raises(ContractError):e.assimilate_observation(f[0])
    with pytest.raises(ContractError):e.forecast_future([f[3]])
    assert e.belief is before


def test_neural_fe_failure_rolls_back_memory_and_prefix(monkeypatch):
    p,f,e=setup();old=e.belief;solve=p.mechanics.solve
    def failed(request):raise RuntimeError('injected FE failure')
    monkeypatch.setattr(p.mechanics,'solve',failed);r=e.assimilate_observation(f[0]);assert r.status=='FAILED_ROLLED_BACK';assert e.belief is old and e.hidden is None and not e.frames
    monkeypatch.setattr(p.mechanics,'solve',solve);r=e.assimilate_observation(f[0]);assert r.status=='SUCCESS'


def test_neural_deterministic_replay_and_single_frame_fisher():
    p,f,e=setup();p2,f2,e2=setup();e.diagnose_identifiability=True
    r=e.assimilate_observation(f[0]);s=e2.assimilate_observation(f2[0]);np.testing.assert_array_equal(r.filtered_belief.normalized_mean,s.filtered_belief.normalized_mean)
    assert e.belief.uncertainty.data_rank==1 and any('data_rank_deficient' in w for w in e.belief.uncertainty.warnings)


def test_frame_permutation_and_roi_nan_isolation():
    p,f,e=setup();a=f[0];m=a.case.measurement;o=m.observation;q=len(o.coordinates);order=np.random.default_rng(1).permutation(q);idx=np.column_stack([2*order,2*order+1]).ravel()
    from fatigue_dic_rl.observations.covariance import DisplacementCovariance
    mm=replace(m,observation=replace(o,coordinates=o.coordinates[order],displacement=o.displacement[order],valid_mask=o.valid_mask[order],confidence=o.confidence[order]),covariance=DisplacementCovariance(m.covariance.matrix[np.ix_(idx,idx)]),roi_mask=m.roi_mask[order])
    b=ObservationFrame('permuted',replace(a.case,measurement=mm));g1=build_graph_frame(a,e.belief.normalized_mean,predictor=p,physics=True);g2=build_graph_frame(b,e.belief.normalized_mean,predictor=p,physics=True)
    np.testing.assert_allclose(g2.dic.features,g1.dic.features[order],atol=1e-11);np.testing.assert_allclose(g1.global_features,g2.global_features,atol=1e-11)


def test_empty_roi_and_shared_registration_contract_rejected():
    p,f,e=setup();m=replace(f[0].case.measurement,roi_mask=np.zeros(16,dtype=bool));bad=ObservationFrame('empty',replace(f[0].case,measurement=m))
    with pytest.raises(ContractError):e.assimilate_observation(bad)
    with pytest.raises(ContractError):e.assimilate_observation(ObservationBatch((f[0],),'independent_given_shared_registration'))


def test_unknown_E0_refuses_unique_damage_posterior():
    p,f,e=setup()
    with pytest.raises(ContractError,match='unknown E0'):NeuralBeliefEstimator(p,GraphRecurrentEstimator(),known_E0=False)
