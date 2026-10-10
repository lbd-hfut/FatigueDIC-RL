import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.networks.observation_graph import coordinate_knn,equivariant_whitening
from fatigue_dic_rl.networks.graph_fusion import shape_function_aggregate


@pytest.mark.parametrize('q',[1,16,37,64,256])
def test_coordinate_graph_permutation(q):
    x=np.random.default_rng(12).random((q,2));p=np.random.default_rng(9).permutation(q)
    a=coordinate_knn(x);b=coordinate_knn(x[p]);assert set(map(tuple,a.T))==set(map(tuple,p[b].T))


def test_duplicate_coordinate_and_empty_rejected():
    with pytest.raises(ContractError):coordinate_knn(np.zeros((2,2)))
    with pytest.raises(ContractError):coordinate_knn(np.empty((0,2)))


def test_symmetric_whitening_permutation_and_likelihood():
    rng=np.random.default_rng(7);a=rng.normal(size=(12,12));c=a@a.T+np.eye(12);e=rng.normal(size=(6,2));p=rng.permutation(6);idx=np.column_stack([2*p,2*p+1]).ravel()
    w,n=equivariant_whitening(c,e);wp,np_=equivariant_whitening(c[np.ix_(idx,idx)],e[p]);np.testing.assert_allclose(wp,w[p],atol=1e-12)
    np.testing.assert_allclose(np.sum(w*w),np.linalg.norm(np.linalg.solve(np.linalg.cholesky(c),e.ravel()))**2,atol=1e-12);assert np_==pytest.approx(n)


def test_correlated_neighbor_effective_count():
    c=np.kron(np.array([[1.,.99],[.99,1.]]),np.eye(2));_,n=equivariant_whitening(c,np.zeros((2,2)));assert 1.<n<1.02


def test_fusion_interpolation_adjoint_and_normalization():
    f=np.array([[2.,3.],[4.,5.]]);ids=np.array([[0,1,2],[1,2,3]]);w=np.array([[.2,.3,.5],[.4,.4,.2]])
    a,mask=shape_function_aggregate(f,ids,w,5);p=np.array([1,0]);b,bmask=shape_function_aggregate(f[p],ids[p],w[p],5);np.testing.assert_allclose(a,b);assert np.array_equal(mask,bmask)
    np.testing.assert_allclose(a[1],(.3*f[0]+.4*f[1])/.7);assert not mask[4]
    twice,_=shape_function_aggregate(np.tile(f,(2,1)),np.tile(ids,(2,1)),np.tile(w,(2,1)),5);np.testing.assert_allclose(a,twice)


def test_fusion_fe_node_renumbering():
    ids=np.array([[0,1,2],[2,3,4]]);w=np.full((2,3),1/3);f=np.arange(8).reshape(2,4);p=np.array([2,4,1,0,3]);inverse=np.argsort(p)
    a,_=shape_function_aggregate(f,ids,w,5);b,_=shape_function_aggregate(f,inverse[ids],w,5);np.testing.assert_allclose(b,a[p])
