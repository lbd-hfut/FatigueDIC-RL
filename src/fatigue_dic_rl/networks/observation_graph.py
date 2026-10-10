"""Coordinate-defined irregular graphs; NumPy contract usable without torch.

Symmetric covariance whitening is permutation equivariant. The equivalent
Cholesky-whitened innovation remains available for likelihood diagnostics.
"""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array


@dataclass(frozen=True)
class GraphArrays:
    coordinates_mm: np.ndarray
    features: np.ndarray
    edge_index: np.ndarray

    def __post_init__(self):
        for key, kind in [('coordinates_mm','float'),('features','float'),('edge_index','int')]:
            object.__setattr__(self,key,array(getattr(self,key),kind=kind))
        q=len(self.coordinates_mm)
        if (q<1 or self.coordinates_mm.shape!=(q,2) or self.features.ndim!=2 or len(self.features)!=q
                or not np.isfinite(self.coordinates_mm).all() or not np.isfinite(self.features).all()
                or self.edge_index.ndim!=2 or self.edge_index.shape[0]!=2
                or np.any(self.edge_index<0) or np.any(self.edge_index>=q)):
            raise ContractError('graph requires finite physical [Q,2], finite [Q,F], and edges [2,M]')


def coordinate_knn(coordinates_mm, k=6):
    x=np.asarray(coordinates_mm,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or not len(x) or not np.isfinite(x).all() or k<1:
        raise ContractError('kNN needs nonempty finite [Q,2] coordinates and positive k')
    if len(np.unique(x,axis=0))!=len(x):
        raise ContractError('duplicate coordinates need an acquisition-level covariance/merging policy; rejected')
    if len(x)==1:return np.zeros((2,1),dtype=int)  # declared isolated-node self edge
    d=np.sum((x[:,None]-x[None,:])**2,axis=2);np.fill_diagonal(d,np.inf)
    threshold=np.partition(d,min(k,len(x)-1)-1,axis=1)[:,min(k,len(x)-1)-1]
    # Include geometric ties, never break ties using array positions.
    src,dst=np.where(d<=threshold[:,None]+64*np.finfo(float).eps*np.maximum(1.,threshold[:,None]))
    return np.array([dst,src],dtype=int)


def equivariant_whitening(covariance, residual, variance_floor_mm2=0.):
    c=np.asarray(covariance,dtype=float);r=np.asarray(residual,dtype=float).ravel()
    if c.shape!=(len(r),len(r)) or not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ContractError('whitening support mismatch')
    c=c+np.eye(len(r))*variance_floor_mm2
    if np.count_nonzero(c-np.diag(np.diag(c)))==0:
        if np.min(np.diag(c))<=0:raise ContractError('positive covariance or explicit variance floor required')
        w=r/np.sqrt(np.diag(c))
    else:
        eig,v=np.linalg.eigh(c)
        if np.min(eig)<=0:raise ContractError('singular covariance cannot define graph whitened evidence')
        w=v@((v.T@r)/np.sqrt(eig))
    std=np.sqrt(np.diag(c));rho=c/np.outer(std,std)
    effective_components=len(r)**2/float(np.sum(rho**2))
    return w.reshape(-1,2), effective_components/2


def affine_features(coordinates, displacement, *, length_mm=1., displacement_mm=.01):
    """Least-squares displacement gradients/intercepts; no damage/truth decoder."""
    x=np.asarray(coordinates)/length_mm;u=np.asarray(displacement)/displacement_mm
    a=np.column_stack([np.ones(len(x)),x]);return np.linalg.lstsq(a,u,rcond=1e-10)[0].ravel()
