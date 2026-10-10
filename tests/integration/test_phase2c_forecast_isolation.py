from pathlib import Path
import numpy as np
from fatigue_dic_rl.training.correction_dataset import generate_correction_case
import fatigue_dic_rl.evaluation.correction_verification as verification


def test_all_forecasts_complete_before_truth_and_future_observation_IO(tmp_path,monkeypatch):
    p=tmp_path/'case';generate_correction_case(p,seed=193,split='test',scenario='matched',sigma_mm=0,scales=(.7,1.4));count=[0];original_forecast=verification.forecast_future;original_load=np.load;original_read=Path.read_text
    def forecast(*args,**kwargs):
        result=original_forecast(*args,**kwargs);count[0]+=1;return result
    def load(file,*args,**kwargs):
        name=str(file)
        if 'synthetic_trajectory' in name or '/observations/' in name and int(Path(file).parent.name)>2:assert count[0]==2
        return original_load(file,*args,**kwargs)
    def read(self,*args,**kwargs):
        if self.name=='synthetic_truth.json':assert count[0]==2
        return original_read(self,*args,**kwargs)
    monkeypatch.setattr(verification,'forecast_future',forecast);monkeypatch.setattr(np,'load',load);monkeypatch.setattr(Path,'read_text',read)
    r=verification.evaluate_correction_method(p,'B0_physics_prior',tmp_path/'result',full_horizon=False)
    assert r['status']=='CONVERGED' and r['forecast_before_truth_IO'] and r['irreversibility_violations']==0


def test_registered_observation_and_physical_forecast_scores_have_distinct_frames(tmp_path):
    p=tmp_path/'registered'
    generate_correction_case(p,seed=194,split='test',scenario='matched',sigma_mm=0,
        scales=(.7,1.4),registration=True)
    r=verification.evaluate_correction_method(p,'B0_physics_prior',tmp_path/'result',
        full_horizon=False,extra_options={'shared_registration':True})
    assert abs(r['parameters']['tx']-5e-5)<1e-10
    assert max(r['forecast_displacement_rmse_mm'])<1e-10
    assert max(r['forecast_registered_observation_rmse_mm'])<1e-10
    with np.load(tmp_path/'result/fields.npz',allow_pickle=False) as a:
        # Nonzero camera registration remains in observations, never in physical u.
        assert np.max(abs(a['forecast_displacement_mm']-a['forecast_physical_displacement_mm']))>1e-5
