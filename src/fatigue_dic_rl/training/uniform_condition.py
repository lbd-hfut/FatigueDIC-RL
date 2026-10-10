"""Stopped-gradient homogeneous-rectangle training conditions, not an FEM bridge.

Exact for the declared x-traction/free-contraction rectangle, uniform D, R=-1,
reference energy .025, m=1, canonical step 2 and Phase 1B adaptive subdivision.
Production inference and forecasts always use the independent SciPy solvers.
"""
import numpy as np
import torch


def uniform_damage(d0,A,E,N):
    def advance(d,dt,depth=0):
        updated=d+(.95-d)*(-np.expm1(-dt*A*2000./E/(1.-d)**2))
        if updated-d>.03:
            if depth>=8:raise ValueError('uniform training condition exceeds subdivision limit')
            return advance(advance(d,dt/2.,depth+1),dt/2.,depth+1)
        return updated
    d=float(d0);n=0.
    while n+2.<=N+1e-12:d=advance(d,2.);n+=2.
    if N>n+1e-12:d=advance(d,N-n)
    return d


def causal_condition(data,previous):
    """Detach parameter conditions; GRU hidden history retains its torch gradient."""
    z=previous.detach();p=z.cpu().numpy();g=data['global'].detach().cpu().numpy()
    coefficient=np.array([10./(row[22]*2000.*(1.-uniform_damage(.38+.38*v[0],.00205+.00195*v[1],row[22]*2000.,row[18]*40.))) for v,row in zip(p,g)])
    c=torch.as_tensor(coefficient,device=z.device,dtype=z.dtype);nu=data['global'][:,23]
    db,fb=data['db'],data['fb'];dic_basis=torch.stack([data['dx'][:,0],-nu[db]*data['dx'][:,1]],dim=1)
    fe_basis=torch.stack([data['fx'][:,0],-nu[fb]*data['fx'][:,1]],dim=1)
    out=dict(data);df=data['df'].clone();ff=data['ff'].clone();glob=data['global'].clone()
    df[:,4:6]=c[db,None]*dic_basis/.01;df[:,6:8]=df[:,2:4]-df[:,4:6]
    df[:,8:10]=torch.asinh((data['white_measured']-c[db,None]*data['white_basis'])/10.)
    ff[:,-2:]=c[fb,None]*fe_basis/.01;pred=torch.zeros_like(glob[:,6:12]);pred[:,2]=c/.01;pred[:,5]=-nu*c/.01
    glob[:,6:12]=pred;glob[:,12:18]=glob[:,:6]-pred
    out.update(df=df,ff=ff,global_=glob,previous=z);out['global']=out.pop('global_');return out


def forward_causal(model,sequence):
    hidden=None;previous=sequence[0]['previous'];outputs=[]
    for data in sequence:
        if model.config.physics:data=causal_condition(data,previous)
        out=model.step(data,hidden);outputs.append(out);hidden=out['hidden'];previous=out['normalized_mean'].detach()
    return outputs
