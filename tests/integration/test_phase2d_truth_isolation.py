import ast
from pathlib import Path
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.training.robust_correction_dataset import generate_case
from fatigue_dic_rl.correction.robust_estimation import acquisition_for,split_batches,fit_one,rollout
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter


def test_selector_modules_have_no_truth_training_or_evaluation_imports():
    for name in ('model_selection','correction_gate','identifiability_gate','robust_estimation','noise_robustness'):
        text=(Path('src/fatigue_dic_rl/correction')/(name+'.py')).read_text();tree=ast.parse(text)
        modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(m and any(x in m.split('.') for x in ('training','evaluation','networks','rl')) for m in modules)
        assert not any(s in text for s in ('synthetic_truth.json','synthetic_trajectory.npz','generation_spec.json'))


def test_loader_and_bulk_forecast_need_no_future_observations_or_truth(tmp_path):
    folder=tmp_path/'case';generate_case(folder,dict(seed=990,split='test',family='matched',sigma_mm=2e-7,case_id='anonymous'),config_sha256='fixture')
    # Physically remove future/truth assets from the inference copy, retain public load plan.
    for name in ('synthetic_truth.json','synthetic_trajectory.npz','generation_spec.json'):(folder/name).unlink()
    import shutil
    for load in range(5):
        for index in range(3,8):shutil.rmtree(folder/'observations'/str(load)/str(index))
    acquisition=acquisition_for(folder,'B0');fit,adapter=fit_one(acquisition,acquisition.batches);pred=rollout(acquisition,fit)
    assert np.max(abs(pred['parameters']['d0']-.12))>0 # Initialization never comes from truth.
    assert np.isfinite(pred['damage']).all() and pred['cycles'].tolist()==[30.,40.,70.,120.,220.]
    assert all(f.origin_cycle_count==20. for f in pred['forecasts'])
    assert all(max(float(x.cycle_count) for x in b.frames)==20. for b in acquisition.batches)


def test_reserved_history_is_disjoint_from_fit_and_shared_nuisance_is_joint(tmp_path):
    folder=tmp_path/'case';generate_case(folder,dict(seed=991,split='test',family='matched',sigma_mm=5e-5,case_id='anonymous',registration=True),config_sha256='fixture')
    a=acquisition_for(folder,'B5',dict(shared_registration=True));train=split_batches(a,(0.,10.));val=split_batches(a,(20.,))
    ids_train={f.observation_id for b in train for f in b.frames};ids_val={f.observation_id for b in val for f in b.frames}
    assert not ids_train&ids_val
    adapter=MultiTrajectoryAdapter(a.predictors,train)
    assert adapter.parameter_space.names.count('tx')==1
    assert adapter.parameter_space.names.count('theta')==1


def test_public_hidden_channel_rejected(tmp_path):
    import json
    folder=tmp_path/'case';generate_case(folder,dict(seed=992,split='validation',family='matched',sigma_mm=5e-5,case_id='anonymous'),config_sha256='fixture')
    p=folder/'public.json';meta=json.loads(p.read_text());meta['true_family']='matched';p.write_text(json.dumps(meta))
    with pytest.raises(ContractError):acquisition_for(folder,'B0')
