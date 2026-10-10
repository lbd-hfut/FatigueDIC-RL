import numpy as np
import pytest
from dataclasses import replace
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.law_family_benchmarks import reference_hazard,ReferenceLawSpecification,FAMILIES
from fatigue_dic_rl.correction.correction_gate import GatedLawSpecification
from fatigue_dic_rl.evaluation.identifiability_verification import pattern_for
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.sequential_fixtures import fatigue_loading
from fatigue_dic_rl.fatigue import MaterialHistoryState,FatigueEvolutionSolver,FatigueStepRequest

@pytest.mark.parametrize('family',FAMILIES)
def test_law_nonnegative_and_zero_drive(family):
    x=np.r_[0.,np.logspace(-5,2,100)];d=np.full(len(x),.12);xy=np.zeros((len(x),2))
    r=reference_hazard(x,d,xy,-1.,family=family)
    assert np.isfinite(r).all() and np.min(r)>=0 and r[0]==0
    np.testing.assert_allclose(r[1:],reference_hazard(x[1:]+1e-10,d[1:],xy[1:],-1.,family=family),atol=1e-10)

@pytest.mark.parametrize('family',FAMILIES)
def test_actual_fem_reference_step_is_irreversible(family):
    p=pattern_for(rectangle_mesh(2,1,width_mm=2,height_mm=1));q=p.quadrature_layout;s=MaterialHistoryState.initialize(q,q.field(np.full(q.count,.12),'1'),damage_max=.95)
    solver=FatigueEvolutionSolver(p,s,ReferenceLawSpecification(family).build({},s.material_ids));cyc=fatigue_loading(p,40.)[0]
    result=solver.advance(FatigueStepRequest(replace(cyc,delta_N=2.))).committed_state
    assert np.all(result.damage.values>=s.damage.values) and np.max(result.damage.values)<.95 and result.cycle_count==2.

@pytest.mark.parametrize('gate',[0.,.25,1.])
def test_gated_hazard_uses_original_transition_and_bounds(gate):
    p=pattern_for(rectangle_mesh(2,1,width_mm=2,height_mm=1));q=p.quadrature_layout;s=MaterialHistoryState.initialize(q,q.field(np.full(q.count,.12),'1'),damage_max=.95)
    phys=dict(A=8e-5,m=1.,c0=.5,c1=-.1);spec=GatedLawSpecification(gate);solver=FatigueEvolutionSolver(p,s,spec.build(phys,s.material_ids));r=solver.advance(FatigueStepRequest(replace(fatigue_loading(p,40.)[0],delta_N=2.)))
    assert np.all(r.committed_state.damage.values>=s.damage.values)
    if gate==0.:
        baseline=FatigueEvolutionSolver(p,s,GatedLawSpecification(0.).build(dict(phys,c0=3.,c1=-3.),s.material_ids));b=baseline.advance(FatigueStepRequest(replace(fatigue_loading(p,40.)[0],delta_N=2.)))
        np.testing.assert_array_equal(r.committed_state.damage.values,b.committed_state.damage.values)

@pytest.mark.parametrize('bad',[-.1,1.1,np.nan,np.inf])
def test_invalid_gate_rejected(bad):
    with pytest.raises(ContractError):GatedLawSpecification(bad)

def test_mean_stress_at_fixed_R_is_exactly_A_equivalent():
    x=np.array([0.,.2,2.,10.]);d=np.full(4,.12);xy=np.zeros((4,2));R=.3;b=.55
    a=reference_hazard(x,d,xy,R,family='S3',strength=b)
    ref=reference_hazard(x,d,xy,R,family='matched',A=8e-5*np.exp(b*(R+1)))
    np.testing.assert_allclose(a,ref,rtol=1e-15)

def test_spatial_law_is_physical_coordinate_permutation_equivariant():
    xy=np.array([[0.,0.],[1.,.5],[2.,1.]]);x=np.ones(3);d=np.full(3,.12);perm=np.array([2,0,1])
    a=reference_hazard(x,d,xy,-1.,family='S5',strength=.3)
    np.testing.assert_array_equal(reference_hazard(x[perm],d[perm],xy[perm],-1.,family='S5',strength=.3),a[perm])

def test_threshold_cannot_be_exactly_represented_by_positive_power_law():
    x=np.array([.1,.5,1.]);r=reference_hazard(x,np.zeros(3),np.zeros((3,2)),-1.,family='S2',strength=.5)
    assert r[0]==r[1]==0 and r[2]>0
