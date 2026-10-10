"""Local tensor-only checkpoint, versioned source/config provenance."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import torch
from fatigue_dic_rl.networks.temporal_gru import NetworkConfig,GraphRecurrentEstimator


def save_checkpoint(path,model,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch.save({'schema':'phase2b_graph_recurrent_v1','architecture':asdict(model.config),'state_dict':model.state_dict(),'metadata':metadata},path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path,*,device='cpu'):
    data=torch.load(path,map_location=device,weights_only=True)
    if data['schema']!='phase2b_graph_recurrent_v1':raise ValueError('unknown checkpoint schema')
    cfg=dict(data['architecture']);cfg['parameter_names']=tuple(cfg['parameter_names']);model=GraphRecurrentEstimator(NetworkConfig(**cfg)).to(device);model.load_state_dict(data['state_dict']);model.eval();return model,data['metadata']
