import json,tomllib
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from fatigue_dic_rl.training.synthetic_dataset import generate_case
from fatigue_dic_rl.training.checkpoint import save_checkpoint
from fatigue_dic_rl.networks.temporal_gru import GraphRecurrentEstimator,NetworkConfig
from fatigue_dic_rl.evaluation.neural_belief_verification import initialize_worker,evaluation_record


def test_variable_active_covariance_serialization(tmp_path,monkeypatch):
    torch.set_num_threads(1);c=tomllib.loads(Path('configs/phase2b/uniform.toml').read_text());c['dataset']['observation_counts']=[16];c['dataset']['mesh_nx']=[2]
    r=generate_case((c,'test',31000,str(tmp_path/'case'),'missing'));m=GraphRecurrentEstimator(NetworkConfig(uncertainty=False));p=tmp_path/'checkpoint.pt';save_checkpoint(p,m,{'seed':1});initialize_worker({'test_model':str(p)})
    import fatigue_dic_rl.evaluation.neural_belief_verification as verification
    original_forecast=verification.NeuralBeliefEstimator.forecast_future;original_labels=verification.load_labels;called=[]
    def forecast(self,*args,**kwargs):
        result=original_forecast(self,*args,**kwargs);called.append('forecast');return result
    def labels(path):
        assert called,'future truth opened before forecasting';called.append('labels');return original_labels(path)
    monkeypatch.setattr(verification.NeuralBeliefEstimator,'forecast_future',forecast);monkeypatch.setattr(verification,'load_labels',labels)
    row=evaluation_record(r,'test_model',{},output=str(tmp_path/'evaluation'));assert row['status']=='SUCCESS',row
    assert row['forecast_before_label_and_future_measurement_IO']
    out=tmp_path/'evaluation'/r['case_id']/'test_model';assert (out/'update.npz').exists();assert json.loads((out/'record.json').read_text())['conditioned_through_cycle']==20.
