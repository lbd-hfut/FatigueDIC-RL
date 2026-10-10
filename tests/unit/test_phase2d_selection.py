from dataclasses import replace
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.model_selection import HistoricalCandidate, select_model,choose_shrinkage
from fatigue_dic_rl.correction.identifiability_gate import conditional_singular_values


def c(name,square,rank=3,dim=3):return HistoricalCandidate(name,square,160,rank,dim,True,False,True)

def test_correct_physics_does_not_activate_on_small_history_fluctuations():
    r=select_model([c('B0',160,1,1),c('B1',153,2,2),c('B2',152),c('B5',151)])
    assert r['selected']=='B0'

def test_refit_is_preferred_when_it_explains_history():
    r=select_model([c('B0',500,1,1),c('B1',250,2,2),c('B2',150),c('B5',151)])
    assert r['selected']=='B2'

def test_identifiable_structural_correction_requires_independent_history_gain():
    r=select_model([c('B0',500,1,1),c('B1',300,2,2),c('B2',250),c('B5',150)])
    assert r['selected']=='B5'

@pytest.mark.parametrize('changed',[dict(practical_rank=2),dict(boundary=True),dict(converged=False),dict(derivative_checked=False)])
def test_invalid_or_weak_correction_cannot_activate(changed):
    corr=replace(c('B5',0),**changed);r=select_model([c('B0',500,1,1),c('B2',250),corr]);assert r['selected']=='B2'

def test_future_or_fitted_validation_frame_rejected():
    with pytest.raises(ContractError):replace(c('B5',0),validation_cycle=30.)
    with pytest.raises(ContractError):replace(c('B5',0),fitted_through_cycle=20.)

def test_gate_grid_uses_history_only_and_can_abstain():
    r=select_model([c('B0',500,1,1),c('B5',150)])
    a=choose_shrinkage(r,[500,200,150,160,180],coefficient_norm_square=4,components=160)
    assert a['gate']==.5
    b=choose_shrinkage(r,[150,149,148,147,146],coefficient_norm_square=4,components=160)
    assert b['gate']==0

def test_nuisance_projection_includes_cross_information():
    j=np.array([[1.,1.],[2.,2.],[3.,3.]])
    assert conditional_singular_values(j,[1])[0]<1e-14
    independent=np.c_[j[:,0],[1.,0.,-1.]]
    assert conditional_singular_values(independent,[1])[0]>1

def test_validation_covariance_dimension_mismatch_rejected():
    with pytest.raises(ContractError):select_model([c('B0',160,1,1),replace(c('B5',100),validation_components=32)])
