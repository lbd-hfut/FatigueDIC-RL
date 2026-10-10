import numpy as np
import pytest

def test_stopped_causal_physics_uses_own_previous_mean():
    torch=pytest.importorskip('torch')
    from dataclasses import replace
    from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup,generate_truth,observation_series,fatigue_loading
    from fatigue_dic_rl.networks.graph_features import build_graph_frame
    from fatigue_dic_rl.networks.observation_graph import equivariant_whitening
    from fatigue_dic_rl.networks.temporal_gru import pack_graph_frames
    from fatigue_dic_rl.training.uniform_condition import causal_condition,uniform_damage
    model,space,pattern,predictor,x=uniform_setup({'geometry':{'nx':2,'ny':1,'seed':19},'observation':{'count':16}},fatigue=True)
    truth=generate_truth(pattern,model,{'d0':.12,'A':.0015},[20.],x,load_history=fatigue_loading(pattern));f=observation_series(pattern,truth,x,seed=12)[0]
    z=space.encode({'d0':.25,'A':.0025});g=build_graph_frame(f,z,predictor=predictor,physics=True);cov=f.case.measurement.covariance.selected_matrix(f.case.measurement.active_mask)
    g=replace(g,training_whitened_measured=equivariant_whitening(cov,f.case.measurement.observation.displacement,0.)[0],training_whitened_affine_basis=equivariant_whitening(cov,np.column_stack([x[:,0],-.3*x[:,1]]),0.)[0]);data=pack_graph_frames([g]);previous=data['previous'].clone().requires_grad_();actual=causal_condition(data,previous)
    np.testing.assert_allclose(actual['df'].numpy(),data['df'].numpy(),atol=2e-5);assert not actual['previous'].requires_grad
    np.testing.assert_allclose(uniform_damage(.25,.0025,2000.,20.),predictor.damage_states(z,[20.])[20.].damage.values,atol=1e-11)
    changed=causal_condition(data,torch.zeros_like(previous));assert not torch.allclose(changed['df'],actual['df'])
