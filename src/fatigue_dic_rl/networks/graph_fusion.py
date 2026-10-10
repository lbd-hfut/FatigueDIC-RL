"""Normalized P^T aggregation of embeddings; not statistical precision addition."""
import numpy as np
from fatigue_dic_rl.core.validation import ContractError


def shape_function_aggregate(features, node_ids, weights, node_count):
    f=np.asarray(features);ids=np.asarray(node_ids);w=np.asarray(weights)
    if (f.ndim!=2 or ids.shape!=(len(f),3) or w.shape!=ids.shape or np.any(ids<0)
            or np.any(ids>=node_count) or not np.isfinite(f).all() or not np.isfinite(w).all()
            or np.any(w<0) or not np.allclose(w.sum(axis=1),1.,atol=1e-12)):
        raise ContractError('fusion requires active DIC embeddings and declared [Q,3] Tri3 support')
    total=np.zeros((node_count,f.shape[1]));mass=np.zeros(node_count)
    for i in range(3):
        np.add.at(total,ids[:,i],w[:,i,None]*f);np.add.at(mass,ids[:,i],w[:,i])
    total/=np.maximum(mass[:,None],1e-30)
    return total,mass>0


def torch_shape_aggregate(features, node_ids, weights, node_count):
    import torch
    total=features.new_zeros((node_count,features.shape[1]));mass=features.new_zeros((node_count,1))
    for i in range(3):
        total.index_add_(0,node_ids[:,i],weights[:,i,None]*features)
        mass.index_add_(0,node_ids[:,i],weights[:,i,None])
    return total/mass.clamp_min(1e-12),mass.squeeze(1)>0
