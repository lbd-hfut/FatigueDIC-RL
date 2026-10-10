"""Independent temporal reference, identifiability, state/law separation and meshes."""
from pathlib import Path
from dataclasses import replace,asdict
import json,time
import numpy as np
from scipy.integrate import solve_ivp
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import MeshData,BoundaryTag
from fatigue_dic_rl.fatigue import MaterialHistoryState,FatigueEvolutionSolver,FatigueStepRequest,EnergyAmplitudeDriver,ConstantFatigueHazardLaw,FatigueSolverConfig
from fatigue_dic_rl.correction.hybrid_damage_law import HybridDamageLaw,LawSpecification
from fatigue_dic_rl.correction.correction_basis import CorrectionBasis
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition,public_patterns,public_history
from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter,fit_correction
from fatigue_dic_rl.correction.correction_identifiability import correction_information
from fatigue_dic_rl.inverse.parameterizations import Parameter,ParameterSpace,UniformDamage
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.observations.operator import ObservationOperator
from .tri3_meshes import rectangle_mesh
from .identifiability_verification import pattern_for,clean
from .correction_metrics import damage_metrics


def save_json(folder,name,value):
    (Path(folder)/name).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def constant_verification(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);pattern=pattern_for(rectangle_mesh(2,1,width_mm=2,height_mm=1));layout=pattern.quadrature_layout
    initial=MaterialHistoryState.initialize(layout,layout.field(np.full(layout.count,.12),'1'),damage_max=.95)
    from fatigue_dic_rl.evaluation.sequential_fixtures import fatigue_loading
    segment=fatigue_loading(pattern,horizon=40)[0]
    class ConstantBasis:
        dimension=2
        def evaluate(self,x,*args):return np.tile([1.,0.],(len(x),1))
    rows=[]
    for rate in (0.,.001):
        for dt in (10.,5.,2.5,1.25):
            law=HybridDamageLaw(ConstantFatigueHazardLaw(rate),[.2,0.],ConstantBasis());solver=FatigueEvolutionSolver(pattern,initial,law)
            saved=[initial.damage.values]
            for n in np.arange(0,40,dt):saved.append(solver.advance(FatigueStepRequest(replace(segment,N_start=float(n),delta_N=dt))).committed_state.damage.values)
            expected=.95-(.95-.12)*np.exp(-40*rate*np.exp(2*np.tanh(.2)));err=float(np.max(abs(saved[-1]-expected)))
            assert err<5e-16 and np.min(np.diff(saved,axis=0))>=0 and np.max(saved)<.95
            rows.append({'prior_rate':rate,'step':dt,'analytic_error':err,'final_damage_mean':float(np.mean(saved[-1]))})
    save_json(out,'summary.json',{'rows':rows,'passed':True,'transition':'unchanged Phase1B DamageTransition'})
    return rows


def temporal_verification(case,coefficients,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);acq=load_correction_acquisition(case,mode='hybrid',dimension=len(coefficients),coefficients=coefficients);p=acq.predictors[3];layout=p.pattern.quadrature_layout;z=p.parameter_space.encode({'d0':.12});physical=p.parameter_space.decode(z);law=p.law_spec.build(physical,np.zeros(layout.count,dtype=int));backend=Tri3MechanicsSolver();driver=EnergyAmplitudeDriver();calls=[0]
    initial=np.full(layout.count,.12)
    for seg in p.load_history:
        def rhs(n,d):
            state=MaterialHistoryState.initialize(layout,layout.field(d,'1'),damage_max=.95,cycle_count=n);cycle=replace(seg,N_start=float(n),delta_N=1.)
            lo=backend.solve(cycle.request(p.pattern,state.damage,n,'min'));hi=backend.solve(cycle.request(p.pattern,state.damage,n,'max'));driving=driver.evaluate(lo,hi);calls[0]+=2
            return (.95-d)*law.hazard(driving,cycle,state).values
        ode=solve_ivp(rhs,(seg.N_start,seg.N_end),initial,method='DOP853',rtol=1e-10,atol=1e-12)
        assert ode.success;initial=ode.y[:,-1]
    rows=[];saved={'reference_damage':initial,'quadrature_coordinates_mm':layout.coordinates,'physical_weights_mm2':layout.physical_weights}
    for dt in (4.,2.,1.,.5):
        predictor=CorrectionPredictor(UniformDamage(),p.parameter_space,p.pattern,law_spec=p.law_spec,load_history=p.load_history,cycle_step=dt)
        state=predictor.damage_states(z,[220.])[220.];error=damage_metrics(state.damage.values,initial,layout.physical_weights)['rmse'];rows.append({'step_cycle':dt,'damage_rmse_vs_DOP853':error,'final_damage_mean':float(np.mean(state.damage.values))});saved[f'damage_step_{dt}']=state.damage.values
        solitary=predictor.damage_states(z,[220.])[220.];queries=predictor.damage_states(z,[0.,7.,20.,33.,220.])[220.];np.testing.assert_array_equal(solitary.damage.values,queries.damage.values)
    for a,b in zip(rows,rows[1:]):b['observed_order']=float(np.log2(a['damage_rmse_vs_DOP853']/b['damage_rmse_vs_DOP853']))
    assert all(b['damage_rmse_vs_DOP853']<a['damage_rmse_vs_DOP853'] for a,b in zip(rows,rows[1:]));assert .8<rows[-1]['observed_order']<1.2
    np.savez_compressed(out/'fields.npz',**saved);summary={'rows':rows,'reference_fem_solves':calls[0],'query_state_difference':0.,'coefficients_unchanged_across_steps':list(coefficients),'passed':True,'reference':'independent DOP853 time integration on identical FE discretization; piecewise public loads'};save_json(out,'summary.json',summary);return summary


def information_verification(case,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[]
    settings=[('energy',2,('d0',)),('energy',4,('d0',)),('energy',8,('d0',)),('global_diagnostic',2,('d0','A','m')),('spatial',2,('d0',))]
    for kind,k,unknown in settings:
        acq=load_correction_acquisition(case,mode='hybrid',dimension=k,unknown=unknown,law_spec=LawSpecification(mode='hybrid',dimension=k,basis_kind=kind));adapter=MultiTrajectoryAdapter(acq.predictors,acq.batches);values={'d0':.12,'A':8e-5,'m':1.,**{f'c{j}':0. for j in range(k)}};z=adapter.parameter_space.encode(values);u,reg=correction_information(adapter,z)
        row={'basis_kind':kind,'dimension':len(z),'correction_K':k,'data_rank':u.data_rank,'singular_values':u.singular_values_data,'weak_directions':u.weak_directions,'null_directions':u.data_null_directions,'finite_difference_error':u.finite_difference_error,'status':u.status,'regularization_rank':int(np.count_nonzero(np.diag(reg))),'parameter_names':adapter.parameter_space.names}
        np.savez_compressed(out/f'{kind}_K{k}.npz',J=u.jacobian_whitened,F_data=u.data_fisher,F_regularization=reg,F_penalized_local=u.data_fisher+reg,null_directions=u.data_null_directions,weak_directions=u.weak_directions,parameter_scales=adapter.parameter_space.scales)
        rows.append(row)
    redundant=next(r for r in rows if r['basis_kind']=='global_diagnostic');assert redundant['data_rank']==3 and redundant['dimension']==5
    save_json(out,'summary.json',{'rows':rows,'rank_not_stable_recovery':True,'constant_correction_vs_A':'exact finite equivalence when log-x coefficient zero','log_x_correction_vs_m':'local tangent equivalence at zero correction; bounded tanh differs away from linearization','passed':True})
    return clean(rows)


def rollout_from_diagnostic_state(p,state_values,physical,law_spec,cycles):
    layout=p.pattern.quadrature_layout;state=MaterialHistoryState.initialize(layout,layout.field(state_values,'1'),damage_max=.95,cycle_count=20.);law=law_spec.build(physical,state.material_ids);solver=FatigueEvolutionSolver(p.pattern,state,law);result=[]
    for target in cycles:
        while solver.committed_state.cycle_count<target:
            n=solver.committed_state.cycle_count;seg=p.load_history[1];end=min(target,n+p.cycle_step)
            solver.advance(FatigueStepRequest(replace(seg,N_start=n,delta_N=end-n)))
        result.append(solver.committed_state.damage.values)
    return np.asarray(result)


def state_law_separation(case,test_results,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);acq=load_correction_acquisition(case);meta=acq.metadata
    # All formal results already completed before oracle-only diagnostics.
    with np.load(Path(case)/'synthetic_trajectory.npz',allow_pickle=False) as a:truth={k:a[k].copy() for k in a.files}
    params=json.loads((Path(case)/'synthetic_truth.json').read_text());cycles=[30.,40.,50.,70.,120.,220.];indices=[int(np.flatnonzero(truth['cycles']==n)[0]) for n in cycles];twenty=2
    candidate=Path(test_results)/'raw'/meta['case_id']/'B2_refit_Am';with_state=np.load(candidate/'fields.npz',allow_pickle=False);estimated=with_state['current_damage'].copy();with_state.close()
    correct=LawSpecification(mode='structural_reference',beta=params['beta'],spatial_strength=params['spatial_strength'],structural_power=params.get('structural_power',1));prior=LawSpecification();truephysics={'A':params['A'],'m':params['m']};wrongphysics={'A':8e-5,'m':1.}
    fitted=json.loads((candidate/'record.json').read_text())['parameters']
    refitphysics={name:fitted[name] for name in ('A','m')}
    predictions={}
    for label,states,physical,spec in [('oracle_state_wrong_law',truth['damage'][:,twenty],wrongphysics,prior),('oracle_state_refit_Am_law',truth['damage'][:,twenty],refitphysics,prior),('estimated_state_refit_Am_law',estimated,refitphysics,prior),('estimated_state_correct_law',estimated,truephysics,correct),('oracle_state_correct_law',truth['damage'][:,twenty],truephysics,correct)]:
        predictions[label]=np.asarray([rollout_from_diagnostic_state(p,d,physical,spec,cycles) for p,d in zip(acq.predictors,states)])
    rows={label:[damage_metrics(value[:,j],truth['damage'][:,indices[j]],truth['physical_weights_mm2'])['rmse'] for j in range(len(cycles))] for label,value in predictions.items()};assert max(rows['oracle_state_correct_law'])<1e-12
    np.savez_compressed(out/'fields.npz',**predictions,truth_damage=truth['damage'][:,indices],estimated_current_damage=estimated,cycles=np.asarray(cycles))
    summary={'scenario':params['scenario'],'oracle_only':True,'current_state_restart':'memoryless law diagnostic; not formal DIC-only inference','forecast_damage_rmse':rows};save_json(out,'summary.json',summary);return summary


def incorrect_state_negative_control(case,output):
    """Privileged diagnostic: correct law, deliberately biased initial state.

    Initial-state truth is used to define the perturbation, never formal scoring.
    The correction fit itself sees only the observed prefix. It cannot repair N0.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    params=json.loads((Path(case)/'synthetic_truth.json').read_text())
    wrong_d0=params['d0']-.01
    acq=load_correction_acquisition(case,mode='hybrid')
    space=ParameterSpace((Parameter('c0',-4,4,'1'),Parameter('c1',-4,4,'1')),
        {'d0':wrong_d0,'A':params['A'],'m':params['m']})
    predictors=tuple(CorrectionPredictor(UniformDamage(),space,p.pattern,law_spec=p.law_spec,
        load_history=p.load_history,cycle_step=p.cycle_step) for p in acq.predictors)
    adapter=MultiTrajectoryAdapter(predictors,acq.batches)
    fit=fit_correction(adapter,initializations=[np.zeros(2)])
    cycles=[30.,40.,50.,70.,120.,220.]
    predictions={label:np.asarray([[p.damage_states(z,[n])[n].damage.values for n in cycles] for p in predictors])
        for label,z in [('wrong_state_no_correction',np.zeros(2)),('wrong_state_fitted_correction',fit.normalized)]}
    # No future observation or damage truth was used by fitting or prediction.
    with np.load(Path(case)/'synthetic_trajectory.npz',allow_pickle=False) as a:
        truth={k:a[k].copy() for k in a.files}
    indices=[int(np.flatnonzero(truth['cycles']==n)[0]) for n in cycles]
    errors={name:[damage_metrics(value[:,j],truth['damage'][:,indices[j]],truth['physical_weights_mm2'])['rmse'] for j in range(len(cycles))] for name,value in predictions.items()}
    np.savez_compressed(out/'fields.npz',**predictions,truth_damage=truth['damage'][:,indices],cycles=cycles)
    from dataclasses import asdict
    save_json(out,'optimization.json',asdict(fit.optimization))
    result={'oracle_only':True,'true_d0':params['d0'],'fixed_wrong_d0':wrong_d0,'coefficients':fit.physical,
        'fit_status':fit.status,'data_rank':fit.uncertainty.data_rank,'initial_frame_damage_error':.01,
        'data_cost':fit.data_cost,'forecast_damage_rmse':errors,
        'interpretation':'Correct physics with biased fixed initial state can induce a nonzero correction; this is not evidence of constitutive discrepancy.'}
    save_json(out,'summary.json',result);return result


def spatial_mesh_verification(coefficients,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[];point_rng=np.random.default_rng(89001);x=point_rng.uniform([.03,.03],[1.97,.97],(64,2));coordinates_test=point_rng.uniform([0.,0.],[2.,1.],(100,2));basis=CorrectionBasis(2,'spatial');base_fields=basis.evaluate(np.ones(100),np.full(100,.12),coordinates_test)
    for nx in (2,4,8):
        mesh=rectangle_mesh(nx,max(1,nx//2),width_mm=2,height_mm=1);pattern=pattern_for(mesh);from fatigue_dic_rl.evaluation.sequential_fixtures import fatigue_loading
        space=ParameterSpace((Parameter('d0',.02,.3,'1'),),{'A':8e-5,'m':1.,'c0':.6,'c1':.2});p=CorrectionPredictor(UniformDamage(),space,pattern,law_spec=LawSpecification(mode='hybrid',basis_kind='spatial'),load_history=fatigue_loading(pattern,horizon=40));z=space.encode({'d0':.12});s=p.damage_states(z,[40.])[40.]
        response=Tri3MechanicsSolver().solve(replace(pattern,damage=s.damage,cycle_count=40.));u=ObservationOperator(mesh).prepare(x,coordinate_units='mm',coordinate_system='specimen').apply(response.state.displacement).values
        np.savez_compressed(out/f'mesh_{nx}.npz',coordinates_mm=s.layout.coordinates,damage=s.damage.values,weights_mm2=s.layout.physical_weights,dic_coordinates_mm=x,displacement_mm=u,basis_at_quadrature=basis.evaluate(np.ones(s.layout.count),s.damage.values,s.layout.coordinates))
        rows.append({'nx':nx,'V':mesh.n_nodes,'G':s.layout.count,'area_weighted_mean_damage':float(np.average(s.damage.values,weights=s.layout.physical_weights)),'max_damage':float(s.damage.values.max()),'common_DIC_displacement_mm':u})
        np.testing.assert_array_equal(base_fields,basis.evaluate(np.ones(100),np.full(100,.12),coordinates_test))
    for row in rows:row['displacement_RMSE_vs_finest_mm']=float(np.sqrt(np.mean((row.pop('common_DIC_displacement_mm')-u)**2)))
    # Node re-numbering changes neither physical basis nor coupled observations.
    mesh=rectangle_mesh(2,1,width_mm=2,height_mm=1);perm=np.random.default_rng(89002).permutation(mesh.n_nodes);inverse=np.argsort(perm);tags={n:BoundaryTag(inverse[t.node_ids],inverse[t.facets]) for n,t in mesh.boundary_tags.items()};reordered=MeshData(mesh.nodes[perm],[inverse[c] for c in mesh.connectivity],mesh.element_types,mesh.material_region_ids,'mm','specimen',tags,{},'permuted-physics-correction')
    values=[]
    for mm in (mesh,reordered):
        pp=pattern_for(mm);predictor=CorrectionPredictor(UniformDamage(),space,pp,law_spec=LawSpecification(mode='hybrid',basis_kind='spatial'),load_history=fatigue_loading(pp,horizon=40));ss=predictor.damage_states(z,[40.])[40.];res=Tri3MechanicsSolver().solve(replace(pp,damage=ss.damage,cycle_count=40.));mapped=ObservationOperator(mm).prepare(x,coordinate_units='mm',coordinate_system='specimen').apply(res.state.displacement).values;values.append((ss.damage.values,mapped))
    damage_gap=float(np.max(abs(values[0][0]-values[1][0])));u_gap=float(np.max(abs(values[0][1]-values[1][1])));assert damage_gap<1e-10 and u_gap<1e-12
    result={'meshes':rows,'node_permutation_max_damage_difference':damage_gap,'node_permutation_max_DIC_difference_mm':u_gap,'basis_at_common_coordinates_identical':True,'mesh_objective_localization':'NOT_QUALIFIED; distinct spatial FE discretizations, no raw G comparison','passed':True};save_json(out,'summary.json',result);return result
