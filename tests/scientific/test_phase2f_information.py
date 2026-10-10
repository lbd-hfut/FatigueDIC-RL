"""Fast physical identities and causal contracts, without long pilot optimization."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.information_sufficiency import phase_at,history_for,BoundedProfileAdapter,read_observations
from fatigue_dic_rl.evaluation.information_pilot_diagnostics import expressibility_diagnostics,local_information
from fatigue_dic_rl.inverse.parameterizations import ParameterSpace,Parameter,UniformDamage
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference


def test_phase_boundary_uses_completed_loading_and_cycles_not_indices():
    history=[dict(start=0.,end=20.,amplitude=1.,R=-1.),dict(start=20.,end=40.,amplitude=.4,R=0.)]
    assert phase_at(history,0.)['amplitude']==1.
    assert phase_at(history,20.)['amplitude']==1.
    assert phase_at(history,20.01)['amplitude']==.4
    assert phase_at(history,40.)['R']==0.


def test_future_and_truth_keys_rejected_before_measurement_read(tmp_path):
    from fatigue_dic_rl.correction.information_sufficiency import PUBLIC_KEYS
    meta=dict.fromkeys(PUBLIC_KEYS);meta.update(schema='phase2f-development-v1',cycles=[0.,10.,20.],cutoff=20.)
    (tmp_path/'public.json').write_text(json.dumps(meta))
    with pytest.raises(ContractError,match='future or unavailable'):
        read_observations(tmp_path,included_cycles=(30.,))
    meta['synthetic_truth']={};(tmp_path/'public.json').write_text(json.dumps(meta))
    with pytest.raises(ContractError,match='truth channel'):read_observations(tmp_path)


def test_observation_only_modules_have_no_truth_or_evaluation_imports():
    import ast
    for filename in ('src/fatigue_dic_rl/correction/information_sufficiency.py','src/fatigue_dic_rl/inference/information_pilot.py'):
        tree=ast.parse(Path(filename).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any('evaluation' in m or 'training' in m for m in imports if m)
        literals=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        assert not any(s.endswith('synthetic_truth.json') or s.endswith('synthetic_trajectory.npz') for s in literals)


class LinearAdapter:
    def __init__(self):
        self.parameter_space=ParameterSpace((Parameter('d0',.02,.3,'1'),Parameter('A',1e-5,5e-4,'cycle^-1')),dict(m=1.));self.evidence_ids=('observation',);self.evaluation_count=0
    def residual(self,z):return np.array([z[0]+2*z[1],3*z[0]-z[1]])
    whitened_prediction=residual
    displacement_prediction=residual


def test_bounded_profile_reembeds_physical_candidate_exactly():
    base=LinearAdapter();p=BoundedProfileAdapter(base,'A',2e-4);z=p.parameter_space.encode(dict(d0=.15))
    assert p.parameter_space.decode(z)['A']==2e-4
    np.testing.assert_array_equal(p.residual(z),base.residual(base.parameter_space.encode(dict(d0=.15,A=2e-4))))
    with pytest.raises(ContractError):BoundedProfileAdapter(base,'A',9e-4)
    with pytest.raises(ContractError):BoundedProfileAdapter(base,'m',1.)


@pytest.mark.parametrize('family,first,second',[
    ('S3',dict(D=.12,R=-1.,X=0.),dict(D=.12,R=0.,X=0.)),
    ('S4',dict(D=.05,R=-1.,X=0.),dict(D=.3,R=-1.,X=0.)),
    ('S5',dict(D=.12,R=-1.,X=0.),dict(D=.12,R=-1.,X=2.)),
])
def test_missing_feature_is_expression_failure_even_with_perfect_observations(family,first,second):
    from fatigue_dic_rl.correction.correction_basis import CorrectionBasis
    from fatigue_dic_rl.correction.law_family_benchmarks import reference_hazard
    x=np.array([.8]);basis=CorrectionBasis();phis=[];rates=[]
    for p in (first,second):
        d=np.array([p['D']]);xy=np.array([[p['X'],.5]])
        phis.append(basis.evaluate(x,d,xy,p['R']))
        rates.append(reference_hazard(x,d,xy,p['R'],family=family,strength=.5))
    np.testing.assert_array_equal(phis[0],phis[1])
    assert not np.allclose(rates[0],rates[1],rtol=1e-3,atol=0)


def test_shifted_threshold_exact_zero_cannot_be_created_by_bounded_multiplicative_k2():
    from fatigue_dic_rl.correction.law_family_benchmarks import reference_hazard
    x=np.array([.2]);true=reference_hazard(x,np.array([.1]),np.zeros((1,2)),-1.,family='S2',strength=.5)
    assert true[0]==0
    assert (8e-5*x*np.exp(-2))[0]>0


def test_s1_energy_range_is_approximable_but_not_exact():
    r=expressibility_diagnostics()
    assert 0<r['S1']['oracle_log_factor_RMSE']<.1
