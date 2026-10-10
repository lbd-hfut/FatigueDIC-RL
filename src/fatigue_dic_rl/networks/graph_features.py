"""Observation-only tensors; all physics condition features are stopped gradients."""
from dataclasses import dataclass,replace
import numpy as np
from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.observations.operator import ObservationOperator
from .observation_graph import GraphArrays,coordinate_knn,equivariant_whitening,affine_features
from .fe_graph import fe_node_features


@dataclass(frozen=True)
class GraphFrame:
    dic: GraphArrays
    fe: GraphArrays
    mapping_node_ids: np.ndarray
    mapping_weights: np.ndarray
    global_features: np.ndarray
    previous_normalized: np.ndarray
    training_whitened_measured: np.ndarray | None = None
    training_whitened_affine_basis: np.ndarray | None = None


def build_graph_frame(frame, previous_normalized, *, predictor=None, physics=False, previous_cycle=0.):
    d=frame.design;m=frame.case.measurement;mask=m.active_mask;x=d.coordinates_mm[mask]
    if not len(x):raise ContractError('empty ROI rejected')
    edges=coordinate_knn(x);mapping=ObservationOperator(d.pattern.mesh).prepare(x,coordinate_units='mm',coordinate_system=d.pattern.mesh.coordinate_system)
    if not mapping.valid_mask.all():raise ContractError('active DIC coordinate outside FE mesh')
    u=m.observation.displacement[mask];pred=np.zeros_like(u);node_u=None
    if physics:
        if predictor is None:raise ContractError('physics features require verified predictor')
        z=np.asarray(previous_normalized,dtype=float);state=predictor.damage_states(z,[d.cycle_count])[d.cycle_count]
        predicted=predictor.observe(z,d,state);pred=predicted.displacement_mm[mask]
        mechanics_damage=state.damage if not d.known_healthy else d.pattern.quadrature_layout.field(np.zeros_like(state.damage.values),'1')
        request=replace(predictor._candidate_pattern(d.pattern,predictor.parameter_space.decode(z)),damage=mechanics_damage,
                        cycle_count=d.cycle_count,loading_state=LoadingState(d.pattern.loading_state.load_id,d.loading_phase))
        node_u=predictor.mechanics.solve(request).state.displacement.values
    cov=m.covariance.selected_matrix(mask);residual=u-pred
    white,effective=equivariant_whitening(cov,residual,frame.case.variance_floor_mm2)
    sd=np.sqrt(np.diag(cov)+frame.case.variance_floor_mm2).reshape(-1,2)
    mat=next(iter(d.pattern.materials.values())).parameters;young=mat['young_modulus_n_per_mm2']/2000.;nu=mat['poisson_ratio']
    load=np.zeros(2)
    for b in d.pattern.traction_bcs:load+=np.asarray(b.values)/10.
    time=np.array([d.cycle_count/40.,(d.cycle_count-previous_cycle)/40.])
    if np.min(time)<0:raise ContractError('causal cycles must be monotone')
    # Cholesky whitening is ordered; symmetric whitening above is equivariant.
    # Saturation is a declared feature normalization, never a likelihood modification.
    white=np.arcsinh(white/10.)
    base=np.column_stack([x/1.,u/.01,pred/.01,residual/.01,white,sd/1e-4,
                          np.ones(len(x)),np.ones(len(x)),np.full(len(x),young),np.tile(load,(len(x),1)),np.tile(time,(len(x),1))])
    glob=np.r_[affine_features(x,u),affine_features(x,pred),affine_features(x,residual),time,load,young,nu,
               sd.mean(axis=0)/1e-4,effective/64.,mask.mean()]
    if not physics:
        base[:,4:10]=0.;glob[6:18]=0.  # genuine without-prediction/innovation ablation
    return GraphFrame(GraphArrays(x,base,edges),fe_node_features(d.pattern,node_u),mapping.node_ids,mapping.weights,
                      glob,np.asarray(previous_normalized,dtype=float).copy())
