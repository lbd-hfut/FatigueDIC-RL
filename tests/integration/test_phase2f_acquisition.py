"""Actual FE producer is separated from a truth-blind, causal acquisition reader."""
from pathlib import Path
import hashlib,tomllib,json
import numpy as np
import pytest
from fatigue_dic_rl.training.robust_correction_dataset import generate_case
from fatigue_dic_rl.evaluation.information_pilot_truth import produce_case
from fatigue_dic_rl.correction.information_sufficiency import read_observations,adapter_for,nominal,history_for
from fatigue_dic_rl.core.validation import ContractError


@pytest.fixture(scope='module')
def acquired(tmp_path_factory):
    root=tmp_path_factory.mktemp('p2f-acquisition');cfg=tomllib.loads(Path('configs/phase2f/pilot.toml').read_text());base=root/'world'
    generate_case(base,dict(case_id='development-fixture',seed=240100,split='phase2f-development',family='matched',sigma_mm=5e-5,Q=16),config_sha256='fixture')
    case=root/'acquisition';produce_case(base,case,dict(name='fixture',loading='fixed',cycles=[0.,10.,20.],options={},healthy=True,reaction=True),cfg,'fixture','fixture')
    return case


def test_acquisition_only_reads_public_prefix_and_independent_auxiliary_data(acquired,monkeypatch):
    saved=Path.open;reads=[]
    def checked(path,*args,**kwargs):
        assert 'synthetic_truth' not in str(path) and 'synthetic_trajectory' not in str(path) and 'evaluation_' not in str(path)
        reads.append(str(path));return saved(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked)
    meta,patterns,batches=read_observations(acquired)
    assert len(batches)==5 and all(len(b.frames)==4 for b in batches)
    assert all(sum(f.case.reaction is not None for f in b.frames)==1 for b in batches)
    assert all(sum(f.case.known_healthy for f in b.frames)==1 for b in batches)
    assert not any('/observations/' in p and Path(p).parent.name=='3' for p in reads)
    ad,ps=adapter_for(meta,patterns,batches,'B0');z=nominal(ad.parameter_space)
    assert np.isfinite(ad.residual(z)).all()
    assert len(ad.residual(z))==5*(4*32+1)


def test_future_rejection_precedes_fem_and_foreign_truth_can_not_be_passed(acquired):
    with pytest.raises(ContractError,match='future'):read_observations(acquired,included_cycles=(30.,))
    meta,patterns,_=read_observations(acquired,included_cycles=(0.,))
    broken=dict(meta,history=[dict(start=2.,end=10.,amplitude=1.,R=-1.)])
    with pytest.raises(ContractError,match='contiguous'):history_for(patterns[0],broken)


def test_same_candidate_material_history_is_query_order_independent(acquired):
    meta,patterns,batches=read_observations(acquired);ad,ps=adapter_for(meta,patterns,batches,'B0');p=ps[-1];z=nominal(p.parameter_space)
    first=p.damage_states(z,(20.,30.));expanded=p.damage_states(z,(0.,11.3,20.,23.7,30.))
    np.testing.assert_array_equal(first[30.].damage.values,expanded[30.].damage.values)
    other=z.copy();other[0]=min(.9,z[0]+.2)
    p.damage_states(other,(30.,))
    again=p.damage_states(z,(30.,));np.testing.assert_array_equal(first[30.].damage.values,again[30.].damage.values)


def test_real_static_fe_force_alone_does_not_resolve_E_damage_healthy_does(acquired):
    from dataclasses import replace
    from fatigue_dic_rl.inverse.parameterizations import ParameterSpace,Parameter,UniformDamage
    from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
    from fatigue_dic_rl.evaluation.information_pilot_diagnostics import local_information
    meta,patterns,batches=read_observations(acquired,included_cycles=(0.,))
    space=ParameterSpace((Parameter('d0',.02,.3,'1'),Parameter('E0',1500.,2600.,'N/mm^2'),Parameter('load_factor',.85,1.15,'1')),{})
    z=space.encode(dict(d0=.12,E0=2050.,load_factor=1.))
    damage=[f.case for b in batches for f in b.frames if not f.case.known_healthy]
    allcases=[f.case for b in batches for f in b.frames]
    dic=StaticForwardAdapter(UniformDamage(),space,[replace(c,reaction=None) for c in damage]);force=StaticForwardAdapter(UniformDamage(),space,damage);health=StaticForwardAdapter(UniformDamage(),space,allcases)
    assert local_information(dic,z)[0]['numerical_rank']==1
    assert local_information(force,z)[0]['numerical_rank']==2
    assert local_information(health,z)[0]['numerical_rank']==3
    E=2250.;equivalent=space.encode(dict(d0=1-2050*(1-.12)/E,E0=E,load_factor=1.))
    np.testing.assert_allclose(force.whitened_prediction(z),force.whitened_prediction(equivalent),rtol=0,atol=1e-7)
    assert np.linalg.norm(health.whitened_prediction(z)-health.whitened_prediction(equivalent))>10
