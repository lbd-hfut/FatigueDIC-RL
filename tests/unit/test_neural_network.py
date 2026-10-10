from dataclasses import replace
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from fatigue_dic_rl.networks.temporal_gru import NetworkConfig,GraphRecurrentEstimator,pack_graph_frames
from fatigue_dic_rl.networks.graph_features import GraphFrame
from fatigue_dic_rl.networks.observation_graph import GraphArrays,coordinate_knn
from fatigue_dic_rl.training.checkpoint import save_checkpoint,load_checkpoint


def example(q=16,v=9):
    rng=np.random.default_rng(4);dx=rng.random((q,2));fx=rng.random((v,2));return GraphFrame(GraphArrays(dx,rng.normal(size=(q,19)),coordinate_knn(dx)),GraphArrays(fx,rng.normal(size=(v,9)),coordinate_knn(fx)),np.tile([0,1,2],(q,1)),np.tile([.2,.3,.5],(q,1)),rng.normal(size=28),np.zeros(2))


def test_network_dic_and_fe_permutation():
    torch.set_num_threads(1);torch.manual_seed(7);m=GraphRecurrentEstimator();g=example();rng=np.random.default_rng(2);dp=rng.permutation(16);fp=rng.permutation(9);di=np.argsort(dp);fi=np.argsort(fp)
    perm=GraphFrame(GraphArrays(g.dic.coordinates_mm[dp],g.dic.features[dp],di[g.dic.edge_index]),GraphArrays(g.fe.coordinates_mm[fp],g.fe.features[fp],fi[g.fe.edge_index]),fi[g.mapping_node_ids[dp]],g.mapping_weights[dp],g.global_features,g.previous_normalized)
    with torch.no_grad():a=m.step(pack_graph_frames([g]));b=m.step(pack_graph_frames([perm]))
    for k in ['normalized_mean','covariance_normalized','hidden']:torch.testing.assert_close(a[k],b[k],rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('q,v',[(1,3),(16,6),(37,15),(64,45),(256,153)])
def test_arbitrary_shapes_and_psd(q,v):
    torch.set_num_threads(1);m=GraphRecurrentEstimator();r=m.step(pack_graph_frames([example(q,v)]));assert r['normalized_mean'].shape==(1,2);assert torch.all(torch.abs(r['normalized_mean'])<1);assert torch.linalg.eigvalsh(r['covariance_normalized']).min()>0


def test_six_parameter_head_contract():
    torch.set_num_threads(1);cfg=NetworkConfig(parameter_names=('d_base','amplitude','xc','yc','sx','sy'));m=GraphRecurrentEstimator(cfg);g=replace(example(),previous_normalized=np.zeros(6));r=m.step(pack_graph_frames([g]));assert r['covariance_normalized'].shape==(1,6,6);assert torch.linalg.eigvalsh(r['covariance_normalized']).min()>0


def test_causal_prefix_and_checkpoint_replay(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(10);m=GraphRecurrentEstimator();x=pack_graph_frames([example()]);a=m([x,x]);b=m([x,x,x]);torch.testing.assert_close(a[1]['normalized_mean'],b[1]['normalized_mean'],rtol=0,atol=0)
    p=tmp_path/'checkpoint.pt';save_checkpoint(p,m,{'seed':10});reloaded,meta=load_checkpoint(p);torch.testing.assert_close(a[1]['normalized_mean'],reloaded([x,x])[1]['normalized_mean'],rtol=0,atol=0);assert meta['seed']==10
    with pytest.raises(FileExistsError):save_checkpoint(p,m,{})


def test_differentiable_parameter_loss_no_scipy_gradient():
    torch.set_num_threads(1);m=GraphRecurrentEstimator();r=m.step(pack_graph_frames([example()]));loss=r['normalized_mean'].square().mean();loss.backward();assert m.parameter_head.net[-1].weight.grad is not None
    assert not pack_graph_frames([example()])['df'].requires_grad
