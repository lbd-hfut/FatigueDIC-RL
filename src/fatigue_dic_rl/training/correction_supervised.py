"""Privileged residual learning of K global coefficients; FEM features stop-gradient.

Torch differentiates only a bounded parameter vector and alpha*tanh(Phi*a).
No NumPy/SciPy mechanics solve is on this gradient path.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from fatigue_dic_rl.core.validation import ContractError
from .correction_dataset import load_privileged_prefix,source_hash


class ResidualCoefficientModel(torch.nn.Module):
    def __init__(self, dimension, *, alpha=2.):
        super().__init__();self.raw=torch.nn.Parameter(torch.randn(dimension,dtype=torch.float64)*.02);self.alpha=alpha
    @property
    def coefficients(self):return 4*torch.tanh(self.raw)
    def forward(self,phi):return self.alpha*torch.tanh(phi@self.coefficients)


def train_residual_coefficients(train_cases, validation_cases, output, *, seed, dimension=2, alpha=2., epochs=1600, patience=150, basis_kind='energy'):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);t=time.perf_counter()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def tensors(cases, splits):
        values=[load_privileged_prefix(p,allowed_splits=splits,dimension=dimension,basis_kind=basis_kind) for p in cases]
        return tuple(torch.tensor(np.concatenate([v[i] for v in values]),dtype=torch.float64,device=device) for i in range(2))
    tx,ty=tensors(train_cases,('train',));vx,vy=tensors(validation_cases,('validation',))
    if max(float(ty.abs().max()),float(vy.abs().max()))>=alpha:
        raise ContractError('target log hazard ratio exceeds declared bounded correction range; no clipping allowed')
    model=ResidualCoefficientModel(dimension,alpha=alpha).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=0.)
    history=[];best=float('inf');best_raw=None;best_epoch=0;last_best=0
    for epoch in range(epochs):
        optimizer.zero_grad();pred=model(tx);loss=((pred-ty)**2).mean()+1e-7*(model.coefficients**2).sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        with torch.no_grad(): val=float(((model(vx)-vy)**2).mean())
        history.append({'epoch':epoch,'train_loss':float(loss.detach()),'validation_loss':val})
        if val<best-1e-13:
            best=val;best_raw=model.raw.detach().cpu().clone();best_epoch=epoch;last_best=epoch
        if epoch>=300 and epoch-last_best>=patience:break
    model.raw.data.copy_(best_raw.to(device));a=model.coefficients.detach().cpu().numpy()
    metadata={'schema':'phase2c-lowdim-residual-v1','seed':seed,'dimension':dimension,'alpha':alpha,'basis_kind':basis_kind,
        'train_cases':list(map(str,train_cases)),'validation_cases':list(map(str,validation_cases)), 'supervision':'privileged synthetic positive prefix hazard log ratio',
        'fem_gradient':'NONE; fixed physical features; only bounded coefficients have torch gradients',
        'device':device,'gpu_memory_bytes':int(torch.cuda.max_memory_allocated()) if device=='cuda' else None,
        'best_epoch':best_epoch,'validation_mse':best,'coefficients':a.tolist(),'wall_seconds':time.perf_counter()-t,'source_sha256':source_hash()}
    torch.save({'raw':best_raw,'metadata':metadata},out/'checkpoint.pt');metadata['checkpoint_sha256']=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    return metadata


def load_residual_checkpoint(path):
    p=Path(path)
    ledger=p.parent/'metadata.json'
    if ledger.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=json.loads(ledger.read_text())['checkpoint_sha256']:
        raise ContractError('frozen residual checkpoint checksum mismatch')
    payload=torch.load(p,map_location='cpu',weights_only=True);meta=payload['metadata']
    if meta['schema']!='phase2c-lowdim-residual-v1' or meta['dimension'] not in (2,4,8) or not 0<=meta['alpha']<=10 or payload['raw'].shape!=(meta['dimension'],) or not torch.isfinite(payload['raw']).all():
        raise ContractError('invalid bounded residual checkpoint schema or parameters')
    model=ResidualCoefficientModel(meta['dimension'],alpha=meta['alpha']);model.raw.data.copy_(payload['raw']);model.eval()
    return model,meta
