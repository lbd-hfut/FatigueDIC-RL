"""Local model separation after projecting declared nuisance tangent directions.

Whitening is supplied by the existing covariance factor. These distances are
design diagnostics, not p-values, calibrated posterior odds, or population accuracy.
"""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError


@dataclass(frozen=True)
class ProjectedDifference:
    whitened_difference: np.ndarray
    projected_difference: np.ndarray
    absorbed_difference: np.ndarray
    nuisance_basis: np.ndarray
    nuisance_singular_values: np.ndarray
    nuisance_rank: int
    rank_threshold: float

    @property
    def raw_squared_distance(self): return float(self.whitened_difference@self.whitened_difference)
    @property
    def projected_squared_distance(self): return float(self.projected_difference@self.projected_difference)
    @property
    def absorbed_squared_distance(self): return float(self.absorbed_difference@self.absorbed_difference)


def project_whitened_difference(difference, nuisance_jacobian, *, relative_tolerance=1e-8, numerical_resolution=0.):
    d=np.asarray(difference,float);j=np.asarray(nuisance_jacobian,float)
    if d.ndim!=1 or not len(d) or j.ndim!=2 or len(j)!=len(d) or not np.isfinite(d).all() or not np.isfinite(j).all() or not np.isfinite([relative_tolerance,numerical_resolution]).all() or relative_tolerance<=0 or numerical_resolution<0:
        raise ContractError('finite whitened [M] difference and [M,K_n] nuisance sensitivity required')
    if j.shape[1]:
        u,s,_=np.linalg.svd(j,full_matrices=False);threshold=max(numerical_resolution,relative_tolerance*s[0] if len(s) else 0.);rank=int(np.count_nonzero(s>threshold));basis=u[:,:rank]
    else:s=np.empty(0);threshold=numerical_resolution;rank=0;basis=np.empty((len(d),0))
    absorbed=basis@(basis.T@d);perpendicular=d-absorbed
    return ProjectedDifference(d.copy(),perpendicular,absorbed,basis,s,rank,float(threshold))


def whitened_model_difference(first,second,nuisance_jacobian,factor,**kwargs):
    a,b,j=np.asarray(first,float),np.asarray(second,float),np.asarray(nuisance_jacobian,float)
    if a.shape!=b.shape or a.ndim!=1 or j.ndim!=2 or len(j)!=len(a):
        raise ContractError('matched model predictions and physical nuisance Jacobian required')
    return project_whitened_difference(factor.whiten(a-b),factor.whiten(j),**kwargs)


def conditional_noise_monte_carlo(projected,*,seed,count=2000,decision_z=2.3263478740408408):
    """Independent measurement noise only, conditional on fixed predicted means.

    Nuisance tangent variation is projected out. No refitting, FE truth, parameter
    posterior sampling or physical-case independence is implied by these trials.
    """
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0 or isinstance(count,bool) or not isinstance(count,int) or count<1 or decision_z<=0:
        raise ContractError('fixed positive Monte Carlo contract required')
    d=projected.projected_difference;norm=float(np.linalg.norm(d));rng=np.random.default_rng(seed)
    if norm<=max(1e-12,projected.rank_threshold):return dict(trials_per_model=count,abstained_per_model=count,correct_per_model=[0,0],wrong_per_model=[0,0],fixed_means=True,independent_physical_cases=0)
    # A projection of independent unit noise along a unit perpendicular vector
    # remains N(0,1); sampling this scalar is exactly equivalent to full vectors.
    noise=rng.standard_normal((2,count));score=noise+np.array([norm/2,-norm/2])[:,None]
    decisions=np.where(score>decision_z,1,np.where(score<-decision_z,-1,0))
    return dict(trials_per_model=count,correct_per_model=[int(np.sum(decisions[0]==1)),int(np.sum(decisions[1]==-1))],wrong_per_model=[int(np.sum(decisions[0]==-1)),int(np.sum(decisions[1]==1))],abstained_per_model=[int(np.sum(decisions[0]==0)),int(np.sum(decisions[1]==0))],projected_norm=norm,fixed_means=True,independent_physical_cases=0,seed=seed)
