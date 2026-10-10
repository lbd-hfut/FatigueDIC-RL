import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference,whitened_model_difference,conditional_noise_monte_carlo
from fatigue_dic_rl.correction.experimental_design import DesignScore,choose_experiment
from fatigue_dic_rl.correction.designed_acquisition import validate_options,loading
from fatigue_dic_rl.inference.designed_model_estimator import projected_validation_choice
from fatigue_dic_rl.observations.covariance import DisplacementCovariance


def test_projection_preserves_cross_terms_and_pythagorean_norm():
    rng=np.random.default_rng(42);j=rng.normal(size=(30,3));d=rng.normal(size=30);p=project_whitened_difference(d,np.c_[j,j[:,0]+j[:,1]])
    assert p.nuisance_rank==3
    np.testing.assert_allclose(j.T@p.projected_difference,0,atol=1e-12)
    assert p.raw_squared_distance==pytest.approx(p.absorbed_squared_distance+p.projected_squared_distance)
    np.testing.assert_allclose(p.projected_difference,d-j@np.linalg.lstsq(j,d,rcond=None)[0],atol=1e-12)


def test_covariance_whitening_matches_mahalanobis_geometry():
    c=DisplacementCovariance(np.array([[4.,1.],[1.,2.]]));f=c.factor(np.ones(1,bool));d=np.array([2.,1.]);j=np.array([[1.],[2.]])
    p=whitened_model_difference(d,np.zeros(2),j,f)
    assert p.raw_squared_distance==pytest.approx(d@np.linalg.solve(c.matrix,d))
    physical=d-j@np.linalg.solve(j.T@np.linalg.solve(c.matrix,j),j.T@np.linalg.solve(c.matrix,d))
    assert p.projected_squared_distance==pytest.approx(physical@np.linalg.solve(c.matrix,physical))


def test_projection_is_permutation_invariant_in_whitened_rows():
    rng=np.random.default_rng(12);d=rng.normal(size=20);j=rng.normal(size=(20,2));idx=rng.permutation(20);a=project_whitened_difference(d,j);b=project_whitened_difference(d[idx],j[idx]);np.testing.assert_allclose(a.projected_difference[idx],b.projected_difference,atol=1e-12)


def test_zero_and_unresolved_nuisance_directions_are_explicit():
    a=project_whitened_difference(np.ones(3),np.ones((3,1)));assert np.linalg.norm(a.projected_difference)<1e-12
    mc=conditional_noise_monte_carlo(a,seed=9);assert mc['abstained_per_model']==2000 and mc['independent_physical_cases']==0
    b=project_whitened_difference(np.ones(3),np.ones((3,1))*1e-10,numerical_resolution=1e-8);assert b.nuisance_rank==0 and b.projected_squared_distance==pytest.approx(3.)


@pytest.mark.parametrize('d,j',[(np.array([np.nan]),np.zeros((1,0))),(np.ones(2),np.ones((3,1))),(np.ones((2,1)),np.ones((2,0)))])
def test_projection_rejects_invalid_contracts(d,j):
    with pytest.raises(ContractError):project_whitened_difference(d,j)


def scores():return [DesignScore('a',4.,1.,160,10.,True,True),DesignScore('b',1.,5.,160,10.,True,True),DesignScore('unsafe',100.,100.,160,10.,False,True)]


def test_design_strategy_safety_budget_and_repeatability():
    assert choose_experiment(scores(),strategy='fisher')['design_id']=='a'
    assert choose_experiment(scores(),strategy='discrimination')['design_id']=='b'
    assert choose_experiment(scores(),strategy='random',seed=3)==choose_experiment(scores()[::-1],strategy='random',seed=3)
    assert choose_experiment([scores()[2]],strategy='fisher')['status']=='ABSTAIN'
    with pytest.raises(ContractError):choose_experiment(scores()+[DesignScore('c',1.,1.,640,10.,True,True)],strategy='fisher')


def test_selector_abstains_with_weak_parameter_information():
    def row(square,singular):return dict(validation_square=square,residual=np.ones(10)*np.sqrt(square/10),prediction=np.zeros(10),nuisance_jacobian=np.empty((10,0)),combined_minimum_singular=singular,dimension=3,derivative_checked=True,converged=True,boundary=False)
    rows={n:row(10.,.5) for n in ('B0','B1','B2','B5')};r=projected_validation_choice(rows);assert r['decision']=='ABSTAIN' and r['deployed_model']=='B0'
    rows['B2']['combined_minimum_singular']=10.;assert projected_validation_choice(rows)['decision']=='KEEP_PHYSICS'


def test_nonidentical_strong_history_evidence_can_activate_correction():
    d=np.ones(10)*2;param=np.zeros(10);obs=d
    rows={n:dict(validation_square=40.,residual=param-obs,prediction=param,nuisance_jacobian=np.empty((10,0)),combined_minimum_singular=10.,dimension=3,derivative_checked=True,converged=True,boundary=False) for n in ('B0','B1','B2','B5')};rows['B5'].update(validation_square=0.,residual=d-obs,prediction=d)
    r=projected_validation_choice(rows);assert r['decision']=='CORRECTION_EVIDENCE' and r['deployed_model']=='B5'
    rows['B5']['boundary']=True;assert projected_validation_choice(rows)['decision']=='ABSTAIN'


def test_truth_conditions_are_not_accepted_as_public_nuisance_features():
    with pytest.raises(ContractError):validate_options({'family':'S1'})
    with pytest.raises(ContractError):validate_options({'load_unknown':1})


def test_candidate_loading_is_real_time_contiguous_not_frame_index():
    class P:
        class loading_state:load_id='public'
    h=loading(P(),.7,0.,future_factor=.25,future_R=-1.)
    assert [(s.N_start,s.N_end) for s in h]==[(0.,10.),(10.,20.),(20.,220.)]
    assert h[1].load_min==0 and h[2].load_min==-.25


def test_conditional_noise_validation_is_reproducible_and_not_world_count():
    p=project_whitened_difference(np.array([3.,4.]),np.empty((2,0)));a=conditional_noise_monte_carlo(p,seed=123);b=conditional_noise_monte_carlo(p,seed=123);assert a==b and a['independent_physical_cases']==0
    assert all(x>1000 for x in a['correct_per_model'])
