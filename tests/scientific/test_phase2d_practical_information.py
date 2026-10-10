import numpy as np
from fatigue_dic_rl.inverse.sensitivity import SensitivityResult
from fatigue_dic_rl.inference.uncertainty import posterior_uncertainty
from fatigue_dic_rl.correction.identifiability_gate import observability
from fatigue_dic_rl.evaluation.model_selection_metrics import wilson,paired_mean_interval


def info(j):
    k=j.shape[1];s=SensitivityResult(j,j,np.ones(k)*.001,np.zeros(k),0.,True);return posterior_uncertainty(s,np.ones(k),normalized=np.zeros(k))

def test_full_numerical_rank_can_be_practically_unobservable():
    u=info(np.diag([100.,10.,.1]));o=observability(u,np.zeros(3))
    assert o['numerical_rank']==3 and o['practical_rank']==2 and not o['eligible']

def test_noise_increases_do_not_create_information():
    j=np.diag([100.,10.,5.]);a=observability(info(j),np.zeros(3));b=observability(info(j/100),np.zeros(3))
    assert a['eligible'] and not b['eligible']
    assert np.linalg.eigvalsh(info(j).data_fisher-info(j/100).data_fisher).min()>=0

def test_interval_uses_cases_and_resolves_zero_activation_precision():
    a=wilson(0,40);assert 0.08<a[1]<.1
    # Eight cases per stratum are insufficient for a 10% false activation upper bound.
    assert wilson(0,8)[1]>.1
    b=paired_mean_interval(np.zeros(40));assert b['bootstrap95']==[0.,0.]

def test_unconverged_full_prefix_cannot_be_deployed_despite_strong_fisher():
    j=np.diag([100.,10.,5.]);k=3;s=SensitivityResult(j,j,np.ones(k)*.001,np.zeros(k),0.,True)
    u=posterior_uncertainty(s,np.ones(k),normalized=np.zeros(k),converged=False)
    o=observability(u,np.zeros(k))
    assert o['numerical_rank']==o['practical_rank']==3 and not o['eligible']
