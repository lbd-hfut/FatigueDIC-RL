"""Acceleration domain, exact ambiguities and observation-only protections."""
from dataclasses import replace
from pathlib import Path
import json,tomllib
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.state_identification_readiness import (
    generate_world,space,verification,health_transfer_diagnostics,ARMS,run_readiness)
from fatigue_dic_rl.inference.state_identification import load_static_cases
from fatigue_dic_rl.inference.uniform_static_scaling import UniformStaticScalingAdapter
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,GaussianDamage


@pytest.fixture(scope='module')
def static_data(tmp_path_factory):
    root=tmp_path_factory.mktemp('readiness-contracts')
    config=tomllib.loads(Path('configs/validation_readiness/development.toml').read_text())
    folder=root/'observations'/'development';generate_world(folder,config['worlds'][0],282010,config)
    return folder,config,space(config)


@pytest.mark.parametrize('name,force,healthy,rank',[(a,b,c,r) for (a,b,c),r in zip(ARMS,(1,2,2,3))])
def test_uniform_static_rank_and_independent_force(static_data,name,force,healthy,rank):
    from fatigue_dic_rl.evaluation.information_pilot_diagnostics import local_information
    folder,config,ps=static_data;ad=UniformStaticScalingAdapter(UniformDamage(),ps,
        load_static_cases(folder,reaction=force,healthy=healthy))
    assert local_information(ad,np.zeros(3))[0]['numerical_rank']==rank


def test_accelerated_fields_and_analytic_jacobian_match_original_tri3(static_data):
    folder,config,ps=static_data
    assert verification(load_static_cases(folder,reaction=True,healthy=True),ps,config)['passed']


def test_nonzero_prescribed_displacement_is_rejected(static_data):
    folder,_,ps=static_data;c=load_static_cases(folder)[0];p=c.pattern
    b=p.displacement_bcs[0];p=replace(p,displacement_bcs=(replace(b,values=np.ones(len(b.values))*.001),)+p.displacement_bcs[1:])
    with pytest.raises(ContractError,match='zero displacement'):
        UniformStaticScalingAdapter(UniformDamage(),ps,[replace(c,pattern=p)])


def test_scaling_rejects_gaussian_decoder(static_data):
    folder,_,ps=static_data
    with pytest.raises(ContractError,match='uniform damage'):
        UniformStaticScalingAdapter(GaussianDamage(),ps,load_static_cases(folder))


def test_unknown_healthy_transfer_restores_damage_null_direction(static_data):
    folder,_,ps=static_data
    r=health_transfer_diagnostics(load_static_cases(folder,reaction=True,healthy=True),ps)
    assert r['local_information']['unknown_healthy_modulus_ratio']['rank']==3
    assert r['local_information']['unknown_healthy_modulus_ratio']['dimension']==4
    assert r['local_information']['unknown_healthy_modulus_ratio']['conditional_damage_information']<1e-7
    assert r['local_information']['unknown_relative_healthy_loading']['conditional_damage_information']<1e-7
    assert r['local_information']['prospective_both_specimen_reactions_unknown_relative_loading']['rank']==4
    assert all(x['max_damage_displacement_difference']<1e-12 and x['max_reaction_difference']<1e-10 for x in r['bounded_actual_FE_equivalent_solutions'])


def test_loader_never_reads_evaluation_or_unrequested_auxiliary_data(static_data,monkeypatch):
    folder,_,_=static_data;original=Path.open;seen=[]
    def checked(path,*args,**kwargs):
        text=str(path);assert 'synthetic_evaluation' not in text
        seen.append(text);return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked);load_static_cases(folder)
    assert not any('/healthy/' in p or p.endswith('reaction.json') for p in seen)


def test_healthy_condition_requires_external_certificate(static_data,tmp_path):
    folder,_,_=static_data;meta=json.loads((folder/'public.json').read_text());meta['healthy_certificate']={'inferred_from_damaged_DIC':True}
    (tmp_path/'public.json').write_text(json.dumps(meta))
    with pytest.raises(ContractError,match='external'):load_static_cases(tmp_path,healthy=True)


def test_extra_truth_channel_is_rejected_before_dic_io(static_data,tmp_path):
    folder,_,_=static_data;meta=json.loads((folder/'public.json').read_text());meta['true_damage']=.12
    (tmp_path/'public.json').write_text(json.dumps(meta))
    with pytest.raises(ContractError,match='truth channel'):load_static_cases(tmp_path)


def test_cost_entry_point_refuses_formal_world_matrix(static_data,tmp_path):
    _,config,_=static_data;config=dict(config,world_seeds=list(range(100)),worlds=[config['worlds'][0]]*100)
    with pytest.raises(ContractError,match='formal'):run_readiness(tmp_path/'must-not-be-created',config,'fixture')
    assert not (tmp_path/'must-not-be-created').exists()
