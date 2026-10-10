import tomllib
from pathlib import Path
import numpy as np
from fatigue_dic_rl.training.synthetic_dataset import generate_case,load_labels,load_parameter_labels,load_observations,load_graph_cache
from fatigue_dic_rl.observations.noise import DICMeasurement


def test_physical_dataset_replay_and_observation_only(tmp_path,monkeypatch):
    c=tomllib.loads(Path('configs/phase2b/uniform.toml').read_text());c['dataset']['observation_counts']=[16];c['dataset']['mesh_nx']=[2]
    generate_case((c,'train',31000,str(tmp_path/'one'),'in_distribution'));generate_case((c,'train',31000,str(tmp_path/'two'),'in_distribution'))
    a=load_labels(tmp_path/'one');b=load_labels(tmp_path/'two')
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    with np.load(tmp_path/'one/synthetic_labels.npz') as archive:archive_type=type(archive)
    getitem=archive_type.__getitem__;label_keys=[]
    def parameter_only(archive,key):
        label_keys.append(key);assert key in ('normalized_parameters','physical_parameters');return getitem(archive,key)
    with monkeypatch.context() as patch:
        patch.setattr(archive_type,'__getitem__',parameter_only);labels=load_parameter_labels(tmp_path/'one')
    assert len(label_keys)==2;np.testing.assert_array_equal(labels['normalized_parameters'],a['normalized_parameters'])
    original=DICMeasurement.load;opened=[]
    def checked(p):
        opened.append(str(p));assert Path(p).name in ('0','1','2');return original(p)
    monkeypatch.setattr(DICMeasurement,'load',checked)
    meta,predictor,frames=load_observations(tmp_path/'one');assert len(frames)==3 and len(opened)==3
    z=a['normalized_parameters'];states=predictor.damage_states(z,[0.,10.,20.])
    for i,f in enumerate(frames):
        np.testing.assert_allclose(states[f.cycle_count].damage.values,a['damage'][i],atol=1e-14)
        np.testing.assert_allclose(predictor.observe(z,f.design).displacement_mm,a['exact_displacement_mm'][i],atol=1e-13)
    views=load_graph_cache(tmp_path/'one/training_condition_features.npz');assert len(views)==2 and len(views[0])==3
    assert views[0][0].dic.features.shape[1]==19 and views[0][0].fe.features.shape[1]==9 and views[0][0].global_features.shape==(28,)
