from pathlib import Path
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from fatigue_dic_rl.training.correction_supervised import ResidualCoefficientModel,load_residual_checkpoint
from fatigue_dic_rl.core.validation import ContractError


def test_gradient_path_and_checkpoint_repeatability(tmp_path):
    torch.manual_seed(5);model=ResidualCoefficientModel(4,alpha=1.7);phi=torch.tensor([[.1,.01,.001,.0001],[.5,.25,.125,.0625]],dtype=torch.float64);target=torch.tensor([.2,.7],dtype=torch.float64);loss=((model(phi)-target)**2).mean();loss.backward()
    assert model.raw.grad is not None and torch.isfinite(model.raw.grad).all() and model.raw.grad.norm()>0
    meta={'schema':'phase2c-lowdim-residual-v1','dimension':4,'alpha':1.7,'basis_kind':'energy'};p=tmp_path/'checkpoint.pt';torch.save({'raw':model.raw.detach(),'metadata':meta},p)
    reload,_=load_residual_checkpoint(p);np.testing.assert_array_equal(model(phi).detach().numpy(),reload(phi).detach().numpy())
    assert sum(p.numel() for p in model.parameters())==4


def test_corrupted_residual_checkpoint_is_rejected(tmp_path):
    p=tmp_path/'checkpoint.pt';torch.save({'raw':torch.zeros(2),'metadata':{'schema':'phase2c-lowdim-residual-v1','dimension':2,'alpha':2.}},p);(tmp_path/'metadata.json').write_text(json.dumps({'checkpoint_sha256':'bad'}))
    with pytest.raises(ContractError):load_residual_checkpoint(p)
