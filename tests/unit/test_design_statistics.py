import tomllib
from pathlib import Path
import numpy as np
from fatigue_dic_rl.evaluation.design_metrics import summarize_worlds
from fatigue_dic_rl.evaluation.model_selection_metrics import wilson


def world(i,family,failed=False):
    c=tomllib.loads(Path('configs/phase2e/study.toml').read_text());arms={}
    for strategy in c['design']['strategies']:
        arms[strategy]={'status':'FAILED'} if failed else dict(status='COMPLETED',methods={n:dict(forecast_damage_rmse=[.01]) for n in ('B0','B2','B5','B7','E9')},inference={'E9':dict(decision='ABSTAIN',deployed_model='B0')},cost={'arm_seconds':1.})
    return dict(case_id=str(i),arms=arms,evaluation_only_family=family,options={},noise_model='independent')


def test_zero_negative_control_activations_do_not_overcome_small_sample_precision():
    c=tomllib.loads(Path('configs/phase2e/study.toml').read_text());r=summarize_worlds([world(i,'matched') for i in range(8)],c)
    for strategy in c['design']['strategies']:
        g=r['strategies'][strategy]['gates']['matched'];assert g['false_K2']==0 and not g['passed'] and g['false_K2_wilson95'][1]>.3
    assert r['qualification']=='PHASE2E_OBSERVABILITY_PARTIALLY_QUALIFIED'


def test_failed_structural_world_is_retained_as_no_benefit():
    c=tomllib.loads(Path('configs/phase2e/study.toml').read_text());rows=[world(0,'S1'),world(1,'S1',True)];r=summarize_worlds(rows,c)
    for strategy in c['design']['strategies']:
        s=r['strategies'][strategy];g=s['gates']['structural'];assert g['independent_worlds']==2 and g['evaluated']==1 and g['failures']==1 and g['practical_improvements']==0
    assert r['independent_worlds']==2


def test_40_world_binomial_precision_is_not_200_design_arms():
    lo,hi=wilson(20,40);assert (hi-lo)/2<.15
    assert (wilson(4,8)[1]-wilson(4,8)[0])/2>.28
