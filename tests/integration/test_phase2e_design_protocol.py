from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from fatigue_dic_rl.training.robust_correction_dataset import generate_case
from fatigue_dic_rl.correction.designed_acquisition import public_prefix,predictors_for,public_designs,blank_batches,combined_batches
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.additional_information_verification import information_probes,temporal_probes


def dataset(tmp_path):
    folder=tmp_path/'case';generate_case(folder,dict(case_id='protocol',seed=310000,split='development',family='matched',sigma_mm=5e-5),config_sha256='test');return folder


def test_acquisition_and_new_prediction_cannot_read_truth_or_N20(tmp_path,monkeypatch):
    folder=dataset(tmp_path);opened=[];original=Path.open
    def guarded(path,*args,**kwargs):
        if path.is_relative_to(folder):
            relative=path.relative_to(folder);opened.append(str(relative))
            assert not relative.name.startswith(('synthetic','generation_spec','observed_reaction'))
            if 'observations' in relative.parts:assert relative.parts[2] in ('0','1')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded);meta,patterns,batches=public_prefix(folder);ps=predictors_for(patterns,'B5',meta,amplitude=.4,R=0.);xy=batches[0].frames[0].case.measurement.observation.coordinates;designs=public_designs(patterns,xy,meta,amplitude=.4);a=MultiTrajectoryAdapter(ps,blank_batches(designs));z=a.parameter_space.encode(dict(d0=.12,c0=.2,c1=.3));y=a.displacement_prediction(z)
    assert y.shape==(160,) and np.isfinite(y).all() and opened
    state=ps[-1].damage_states(z,(10.,20.));assert np.all(state[20.].damage.values>=state[10.].damage.values)
    # Changed material candidate replays from its own initial state.
    zz=a.parameter_space.encode(dict(d0=.14,c0=.2,c1=.3));assert not np.array_equal(ps[-1].damage_states(zz,(20.,))[20.].damage.values,state[20.].damage.values)


def test_new_observation_contract_preserves_covariance_permutation(tmp_path):
    folder=dataset(tmp_path);meta,patterns,batches=public_prefix(folder);ps=predictors_for(patterns,'B2',meta);xy=batches[0].frames[0].case.measurement.observation.coordinates;ds=public_designs(patterns,xy,meta);bs=blank_batches(ds);z=ps[0].parameter_space.encode(dict(d0=.12,A=8e-5,m=1.));a=MultiTrajectoryAdapter(ps,bs);before=a.residual(z);order=np.arange(16)[::-1];channels=np.column_stack((2*order,2*order+1)).ravel()
    from fatigue_dic_rl.observations.covariance import DisplacementCovariance
    shuffled=[replace(d,coordinates_mm=d.coordinates_mm[order],covariance=DisplacementCovariance(d.covariance.matrix[np.ix_(channels,channels)])) for d in ds];after=MultiTrajectoryAdapter(ps,blank_batches(shuffled)).residual(z);assert before@before==pytest.approx(after@after,rel=1e-12)
    future=tuple(replace(d,cycle_count=30.) for d in ds)
    with pytest.raises(ContractError):MultiTrajectoryAdapter(ps,blank_batches(future))
    both=combined_batches(batches,bs);assert all([f.cycle_count for f in b.frames]==[0.,10.,20.] for b in both)


def test_same_cost_force_information_retains_ED_null(tmp_path):
    r=information_probes(dataset(tmp_path),tmp_path/'information');s=r['static_information'];assert s['original_16']['data_rank']==1
    assert s['reaction_budget_matched_15']['data_rank']==2 and s['reaction_budget_matched_15']['cost_units']==s['original_16']['cost_units']==160
    assert s['static_two_modes_8_each']['data_rank']==1 and s['more_64']['data_rank']==1
    assert s['healthy_extra']['data_rank']==3 and s['healthy_extra']['cost_units']==320
    assert r['exact_equivalence']['max_displacement_difference_mm']<1e-12
    assert r['analytic_confounded_model_difference']['projected_norm']<1e-6
    assert r['analytic_confounded_model_difference']['conditional_MC']['abstained_per_model']==2000


def test_real_cycle_query_insertion_and_step_refinement(tmp_path):
    r=temporal_probes(dataset(tmp_path),tmp_path/'temporal');assert all(x['query_insertion_max_difference']==0 for x in r['step_diagnostics']);e=r['RMSE_vs_step_half'];assert e[0]>e[1]>0 and e[0]<1e-3
