from dataclasses import replace
from pathlib import Path
import json,ast
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.training.correction_dataset import generate_correction_case,load_privileged_prefix
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter,fit_correction
from fatigue_dic_rl.correction.hybrid_damage_law import LawSpecification
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty
from fatigue_dic_rl.inverse.parameterizations import Parameter,ParameterSpace,UniformDamage
from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    p=tmp_path_factory.mktemp('phase2c')/'case';generate_correction_case(p,seed=190,split='train',scenario='wrong_Am',sigma_mm=0,scales=(.35,1.4,2.8),d0=.12);return p


def test_matched_parameter_refit_uses_only_prefix_and_recovers(case):
    a=load_correction_acquisition(case,unknown=('d0','A','m'));adapter=MultiTrajectoryAdapter(a.predictors,a.batches);r=fit_correction(adapter,initializations=[adapter.parameter_space.encode({'d0':.16,'A':8e-5,'m':1.})])
    assert r.status=='CONVERGED' and r.uncertainty.data_rank==3
    assert abs(r.physical['d0']-.12)<2e-7 and abs(r.physical['A']-.000128)<2e-8 and abs(r.physical['m']-1.18)<2e-5
    assert all(f.cycle_count<=20 for b in a.batches for f in b.frames)


def test_candidate_history_and_query_invariance(case):
    a=load_correction_acquisition(case,mode='hybrid');p=a.predictors[0];z=p.parameter_space.encode({'d0':.12,'c0':.1,'c1':0})
    direct=p.damage_states(z,[40.])[40.];many=p.damage_states(z,[0.,7.,20.,33.,40.])[40.];np.testing.assert_array_equal(direct.damage.values,many.damage.values)
    z2=z.copy();z2[0]+=.05
    with pytest.raises(ContractError):p.observe(z2,replace(a.batches[0].frames[-1].design,cycle_count=40.),direct)
    assert p.model_id!=load_correction_acquisition(case).predictors[0].model_id


def test_loader_prefix_io_and_truth_import_isolation(case,monkeypatch):
    from fatigue_dic_rl.observations.noise import DICMeasurement
    original=DICMeasurement.load;seen=[]
    def guarded(path):
        seen.append(str(path));assert int(Path(path).name)<=2;return original(path)
    monkeypatch.setattr(DICMeasurement,'load',guarded);a=load_correction_acquisition(case);assert len(seen)==9
    assert all(f.cycle_count>20 for seq in a.future_designs(cycles=[30.,40.]) for f in seq)
    with pytest.raises(ContractError):load_correction_acquisition(case,cutoff=30.)
    for path in ('acquisition.py','physics_predictor.py','correction_optimizer.py'):
        tree=ast.parse((Path('src/fatigue_dic_rl/correction')/path).read_text())
        assert all(not isinstance(n,ast.ImportFrom) or not any(x in (n.module or '').split('.') for x in ('training','evaluation','rl','torch')) for n in ast.walk(tree))


def test_shared_nuisance_cross_information_and_prior_separation(case):
    a=load_correction_acquisition(case,mode='hybrid',shared_registration=True);ad=MultiTrajectoryAdapter(a.predictors,a.batches);z=ad.parameter_space.encode({'d0':.12,'c0':.1,'c1':0,'tx':0,'ty':0,'theta':0})
    s,m=checked_local_jacobian(ad.whitened_prediction,z);u=posterior_uncertainty(s,ad.parameter_space.scales,normalized=z,derivative_method=m)
    assert np.linalg.norm(u.data_fisher[:3,3:])>0
    np.testing.assert_array_equal(u.data_fisher,u.jacobian_whitened.T@u.jacobian_whitened);assert np.all(u.prior_fisher==0)
    assert all(b.noise_contract=='independent_given_shared_registration' for b in a.batches)


def test_global_correction_A_m_redundancy_is_reported(case):
    a=load_correction_acquisition(case,mode='hybrid',unknown=('d0','A','m'),law_spec=LawSpecification(mode='hybrid',basis_kind='global_diagnostic'))
    ad=MultiTrajectoryAdapter(a.predictors,a.batches);z=ad.parameter_space.encode({'d0':.12,'A':8e-5,'m':1.,'c0':0.,'c1':0.})
    s,m=checked_local_jacobian(ad.whitened_prediction,z);u=posterior_uncertainty(s,ad.parameter_space.scales,normalized=z,derivative_method=m)
    assert u.data_rank==3 and u.covariance_normalized is None and len(u.data_null_directions)==2


def test_failed_budget_retained_as_tentative(case):
    a=load_correction_acquisition(case);r=fit_correction(MultiTrajectoryAdapter(a.predictors,a.batches),max_nfev=1)
    assert r.status=='UNCONVERGED_TENTATIVE' and len(r.optimization.starts[0].history)>0
    assert not r.uncertainty.reliable_for_propagation


def test_privileged_train_loader_rejects_test_and_no_epsilon(case,tmp_path):
    phi,y=load_privileged_prefix(case);assert phi.shape[1]==2 and np.isfinite(y).all()
    p=tmp_path/'test';p.mkdir();meta=json.loads((case/'public.json').read_text());meta['split']='test';(p/'public.json').write_text(json.dumps(meta))
    with pytest.raises(ContractError):load_privileged_prefix(p)


def test_public_schema_rejects_hidden_damage(case,tmp_path):
    meta=json.loads((case/'public.json').read_text());meta['damage']=[.12];p=tmp_path/'hidden';p.mkdir();(p/'public.json').write_text(json.dumps(meta))
    with pytest.raises(ContractError):load_correction_acquisition(p)


def test_existing_phase2a_protocol_accepts_fixed_corrected_law(case):
    from fatigue_dic_rl.inference.sequential_map import PhysicsBasedBeliefEstimator,SequentialMAPConfig
    from fatigue_dic_rl.inference.belief_state import EstimateRole
    a=load_correction_acquisition(case,mode='hybrid',coefficients=[.4,-.1]);p=a.predictors[0]
    est=PhysicsBasedBeliefEstimator(p,initializations=[p.parameter_space.encode({'d0':.12})],config=SequentialMAPConfig(assimilation_cutoff=20.))
    result=est.assimilate_observation(a.batches[0]);assert result.filtered_belief is not None
    forecast=est.forecast_future(a.future_designs(cycles=[30.,40.])[0],propagate_uncertainty=False)
    assert forecast.origin_cycle_count==20 and all(v.role==EstimateRole.FORECAST for v in forecast.predictions)
    assert forecast.conditioned_observation_ids==a.batches[0].ids


def test_public_plan_mutation_is_rejected(case):
    a=load_correction_acquisition(case);a.metadata['future_load_factor']=.5
    with pytest.raises(ContractError):a.future_designs(cycles=[30.])


def test_fixed_law_knobs_cannot_silently_override_candidate_law(case):
    a=load_correction_acquisition(case);p=a.predictors[0]
    with pytest.raises(ContractError):CorrectionPredictor(UniformDamage(),p.parameter_space,p.pattern,load_history=p.load_history,law_exponent=2.)
