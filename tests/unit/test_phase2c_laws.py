from dataclasses import replace
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.fatigue import MaterialHistoryState,FatigueEvolutionSolver,FatigueStepRequest,ConstantFatigueHazardLaw,EnergyAmplitudeDriver
from fatigue_dic_rl.fatigue.solver import FatigueStepError
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.identifiability_verification import pattern_for
from fatigue_dic_rl.evaluation.sequential_fixtures import fatigue_loading
from fatigue_dic_rl.correction.correction_basis import CorrectionBasis
from fatigue_dic_rl.correction.hybrid_damage_law import HybridDamageLaw,LawSpecification
from fatigue_dic_rl.correction.physics_residual import StructuralReferenceLaw,positive_hazard_log_ratio


def fixture():
    p=pattern_for(rectangle_mesh(2,1,width_mm=2,height_mm=1));s=MaterialHistoryState.initialize(p.quadrature_layout,p.quadrature_layout.field(np.full(p.quadrature_layout.count,.12),'1'),damage_max=.95);c=replace(fatigue_loading(p)[0],delta_N=2.)
    return p,s,c


@pytest.mark.parametrize('k',[2,4,8])
def test_hazard_bounds_zero_and_exact_neutral(k):
    p,s,c=fixture();dr=FatigueEvolutionSolver(p,s,ConstantFatigueHazardLaw(.001)).evaluate_committed(c)[2]
    basis=CorrectionBasis(k);a=np.linspace(-4,4,k)
    for rate in (0.,.001):
        prior=ConstantFatigueHazardLaw(rate);law=HybridDamageLaw(prior,a,basis);raw=prior.hazard(dr,c,s).values;hybrid=law.hazard(dr,c,s).values
        assert np.all(hybrid>=0) and np.all(hybrid>=raw*np.exp(-2)) and np.all(hybrid<=raw*np.exp(2))
        np.testing.assert_array_equal(HybridDamageLaw(prior,np.zeros(k),basis).hazard(dr,c,s).values,raw)
        assert law.last_diagnostics['basis'].shape==(s.layout.count,k)
        assert not law.coefficients.flags.writeable


@pytest.mark.parametrize('bad',[[np.nan,0],[np.inf,0],[4.1,0],[1,2,3]])
def test_illegal_coefficients_rejected(bad):
    with pytest.raises(ContractError):HybridDamageLaw(ConstantFatigueHazardLaw(.001),bad)


@pytest.mark.parametrize('kind',['energy','spatial'])
@pytest.mark.parametrize('k',[2,4,8])
def test_basis_physical_coordinates_permutation_and_zero(kind,k):
    p,s,c=fixture();coords=s.layout.coordinates;xx=np.linspace(0,4,len(coords));basis=CorrectionBasis(k,kind);a=basis.evaluate(xx,s.damage.values,coords,-1);perm=np.arange(len(xx))[::-1]
    np.testing.assert_array_equal(basis.evaluate(xx[perm],s.damage.values[perm],coords[perm],-1),a[perm]);assert np.all(a[0]==0)
    assert a.shape==(len(xx),k) and np.isfinite(a).all()


def test_positive_log_targets_no_fake_zero_supervision():
    y,mask=positive_hazard_log_ratio([0,1,0,2],[0,.5,1,0]);np.testing.assert_array_equal(mask,[False,True,False,False]);assert y[1]==pytest.approx(np.log(2));assert np.isnan(y[~mask]).all()
    with pytest.raises(ContractError):positive_hazard_log_ratio([-1],[1])


def test_reference_structural_nonnegative_and_neutral():
    p,s,c=fixture();base=LawSpecification().build({'A':8e-5,'m':1},s.material_ids);driver=FatigueEvolutionSolver(p,s,base).evaluate_committed(c)[2]
    np.testing.assert_array_equal(StructuralReferenceLaw(base,beta=0).hazard(driver,c,s).values,base.hazard(driver,c,s).values)
    for power in (1,2):assert np.all(StructuralReferenceLaw(base,beta=3,structural_power=power).hazard(driver,c,s).values>=base.hazard(driver,c,s).values)


def test_corrected_history_irreversibility_new_fem_and_single_update():
    p,s,c=fixture();law=HybridDamageLaw(ConstantFatigueHazardLaw(.001),[.2,0]);solver=FatigueEvolutionSolver(p,s,law);row=solver.advance(FatigueStepRequest(c));r=row.substeps[-1]
    assert solver.committed_state.cycle_count==2. and np.all(solver.committed_state.damage.values>=s.damage.values)
    np.testing.assert_array_equal(solver.committed_state.accumulated_damage_increment.values,solver.committed_state.damage.values-s.damage.values)
    expected=.95-(.95-s.damage.values)*np.exp(-c.delta_N*r.hazard_used.values)
    np.testing.assert_allclose(solver.committed_state.damage.values,expected,atol=2e-16)
    direct=Tri3MechanicsSolver().solve(c.request(p,r.state_after.damage,2.,'max'))
    np.testing.assert_array_equal(r.maximum.state.displacement.values,direct.state.displacement.values)
    assert r.maximum.state.cycle_count==2.


def test_fe_failure_atomic_rollback_with_hybrid():
    p,s,c=fixture()
    class Failing(Tri3MechanicsSolver):
        def __init__(self):super().__init__();self.calls=0
        def solve(self,request):
            self.calls+=1
            if self.calls==3:raise RuntimeError('injected new-damage equilibrium failure')
            return super().solve(request)
    solver=FatigueEvolutionSolver(p,s,HybridDamageLaw(ConstantFatigueHazardLaw(.001),[.2,0]),mechanics=Failing())
    with pytest.raises(FatigueStepError):solver.advance(FatigueStepRequest(c))
    assert solver.committed_state is s and solver.failure_log[-1].stage=='end_mechanics'
    assert s.cycle_count==0 and np.all(s.accumulated_damage_increment.values==0)


def test_frozen_constant_corrected_hazard_semigroup():
    p,s,c=fixture()
    class ConstantBasis:
        dimension=2
        def evaluate(self,x,*args):return np.tile([1.,0.],(len(x),1))
    law=HybridDamageLaw(ConstantFatigueHazardLaw(.001),[.2,0],ConstantBasis())
    outputs=[]
    for dt in (10.,5.,2.5):
        sol=FatigueEvolutionSolver(p,s,law)
        for n in np.arange(0,20,dt):sol.advance(FatigueStepRequest(replace(c,N_start=float(n),delta_N=dt)))
        outputs.append(sol.committed_state.damage.values)
    expected=.95-(.95-.12)*np.exp(-20*.001*np.exp(2*np.tanh(.2)))
    np.testing.assert_allclose(outputs,expected,atol=3e-16)


def test_hybrid_uses_quadrature_states_not_element_or_node_indices():
    from fatigue_dic_rl.geometry import MeshData,BoundaryTag,QuadratureLayout
    from fatigue_dic_rl.geometry.triangles import tri3_shape_functions
    from fatigue_dic_rl.mechanics import DisplacementBC,NodalForceBC
    from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverRequest
    p,_,c=fixture()
    mesh=MeshData([[0,0],[2,0],[0,1]],[[0,1,2]],['tri3'],[0],'mm','specimen',
        boundary_tags={'bottom':BoundaryTag([0,1],[[0,1]])},mesh_id='hybrid-multi-q')
    points=np.array([[1/6,1/6],[2/3,1/6],[1/6,2/3]])
    layout=QuadratureLayout(mesh,'three-point',points,np.full(3,1/6),
        tri3_shape_functions(points)@mesh.nodes,np.tile([[2,0],[0,1]],(3,1,1)),
        np.full(3,2),np.zeros(3,dtype=int),(np.arange(3),),np.arange(3))
    damage=layout.field([.1,.2,.3],'1')
    pattern=Tri3SolverRequest.from_mm(mesh,p.materials,p.loading_state,thickness_mm=.8,
        quadrature_layout=layout,damage=damage,
        displacement_bcs=(DisplacementBC([0,0,1],[0,1,1],[0,0,0],'mm'),),
        nodal_force_bcs=(NodalForceBC([2],[[1,0]]),))
    state=MaterialHistoryState.initialize(layout,damage,damage_max=.95,
        history=layout.field([[1],[2],[3]],'1'),history_names=('passive_marker',))
    law=HybridDamageLaw(LawSpecification().build({'A':8e-5,'m':1},state.material_ids),
        [.6,.2],CorrectionBasis(2,'spatial'))
    result=FatigueEvolutionSolver(pattern,state,law).advance(FatigueStepRequest(
        replace(c,boundary_condition_refs=('bottom',)))).substeps[-1]
    assert mesh.n_elements==1 and layout.count==3
    assert law.last_diagnostics['basis'].shape==(3,2)
    assert np.ptp(law.last_diagnostics['factor'])>0
    assert np.all(result.damage_increment.values>0)
    np.testing.assert_array_equal(result.state_after.history.values,[[1],[2],[3]])
