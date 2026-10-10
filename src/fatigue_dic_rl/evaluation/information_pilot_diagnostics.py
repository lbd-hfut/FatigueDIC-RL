"""Development diagnostics; truth-free parameter analyses and marked oracle bounds."""
from dataclasses import replace
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import least_squares
from fatigue_dic_rl.correction.information_sufficiency import read_observations,adapter_for,nominal,predictors,BoundedProfileAdapter
from fatigue_dic_rl.correction.robust_estimation import clean,persist_fit,report_fit,CountedMechanics
from fatigue_dic_rl.correction.correction_optimizer import fit_correction
from fatigue_dic_rl.inference.information_pilot import extra_information,differences
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,Parameter,ParameterSpace
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference
from fatigue_dic_rl.correction.correction_basis import CorrectionBasis


def local_information(adapter,z):
    s,kind=checked_local_jacobian(adapter.whitened_prediction,z)
    u=posterior_uncertainty(s,adapter.parameter_space.scales,normalized=z,derivative_method=kind)
    return dict(names=adapter.parameter_space.names,numerical_rank=u.data_rank,singular_values=u.singular_values_data,practical_rank=int(np.sum(u.singular_values_data>=3.)),FD_checked=s.converged,FD_resolution=s.absolute_error_norm,null_directions=u.data_null_directions,weak_directions=u.weak_directions),u


def static_diagnostics(folder,out,config):
    out.mkdir(parents=True,exist_ok=False);meta,patterns,batches=read_observations(folder,included_cycles=(0.,))
    space=ParameterSpace((Parameter('d0',.02,.3,'1'),Parameter('E0',1500.,2600.,'N/mm^2'),Parameter('load_factor',.85,1.15,'1')),{})
    rows={};ads={};fits={}
    for name,force,healthy in [('DIC',False,False),('DIC_force',True,False),('DIC_healthy',False,True),('DIC_force_healthy',True,True)]:
        frames=[f for b in batches for f in b.frames if healthy or not f.case.known_healthy]
        cases=[f.case if force else replace(f.case,reaction=None) for f in frames]
        mech=CountedMechanics();ad=StaticForwardAdapter(UniformDamage(),space,cases,mechanics=mech);ad.evidence_ids=tuple(f.observation_id for f in frames)
        f=fit_correction(ad,initializations=[nominal(space)],max_nfev=40,coefficient_regularization=0.)
        persist_fit(out/name,'static',f,ad);info,u=local_information(ad,f.normalized)
        j=u.jacobian_whitened;others=[1,2];p=project_whitened_difference(j[:,0],j[:,others],numerical_resolution=u.finite_difference_error)
        rows[name]=dict(report_fit(f,ad),**info,conditional_d0_information_norm=float(np.linalg.norm(p.projected_difference)),FEM_solves=mech.calls,DIC_scalars=32*len(cases),independent_reactions=sum(c.reaction is not None for c in cases),healthy_measurements=sum(c.known_healthy for c in cases),cycle_exposure=0,scope='initial state only; no A/m or law inference')
        ads[name]=ad;fits[name]=f
    # Evaluate actual nonlinear equivalent stiffness candidates, not just a local J.
    f=fits['DIC_force'];p=f.physical;Eeff=p['E0']*(1-p['d0']);ridge=[]
    for E in config['bounded_profiles']['state_E_grid']:
        q=dict(p,E0=E,d0=1-Eeff/E)
        if not .02<=q['d0']<=.3:ridge.append(dict(E0=E,status='OUTSIDE_DECLARED_DAMAGE_BOX'));continue
        z=space.encode(q);ref=space.encode(p)
        ridge.append(dict(parameters=q,status='LEGAL_EQUIVALENT',max_u_difference_mm=float(np.max(np.abs(ads['DIC_force'].displacement_prediction(z)-ads['DIC_force'].displacement_prediction(ref)))),healthy_whitened_difference_norm=float(np.linalg.norm(ads['DIC_force_healthy'].whitened_prediction(z)-ads['DIC_force_healthy'].whitened_prediction(ref)))))
    # Bounded actual-FEM likelihood profiles allow simultaneous nonlinear compensation.
    prof=[];ad=ads['DIC_force_healthy'];f=fits['DIC_force_healthy']
    for value in config['bounded_profiles']['state_E_grid']:
        a=BoundedProfileAdapter(ad,'E0',value)
        try:
            g=fit_correction(a,initializations=[a.parameter_space.encode(f.physical)],max_nfev=30,coefficient_regularization=0.)
            persist_fit(out/'healthy_E_profile'/str(value),'profile',g,a);prof.append(dict(E0=value,data_cost=g.data_cost,status=g.status,physical=g.physical))
        except Exception as e:prof.append(dict(E0=value,status='FAILED_RETAINED',reason=str(e)))
    result=dict(variants=rows,nonlinear_ED_equivalence=ridge,healthy_E_bounded_profile=prof,profile_scope='bounded conditional cost, NOT confidence interval or proof of global uniqueness',static_law_identifiability='A/m/hazard absent from static forward operator; frozen diagnostic loads cannot infer the earlier fatigue law',force_scope='load controlled global force balance fixes load_factor, not E0-D without healthy baseline')
    (out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return clean(result)


def information_only(folder,parameters,out):
    out.mkdir(parents=True,exist_ok=False);meta,patterns,batches=read_observations(folder);rows={};ads={};dummyfits={}
    from types import SimpleNamespace
    for method in ('B2','B5'):
        ad,ps=adapter_for(meta,patterns,batches,method);z=ad.parameter_space.encode(parameters[method]);info,u=local_information(ad,z)
        np.savez_compressed(out/(method+'.npz'),J_whitened=u.jacobian_whitened,F_data=u.data_fisher,parameters_normalized=z)
        dummyfits[method]=SimpleNamespace(normalized=z,uncertainty=u);ads[method]=ad
        rows[method]=dict(**info,parameters=parameters[method],FEM_solves=sum(p.mechanics.calls for p in ps),linearization='same primary multi-history fitted point, NO refit on this protocol',parameter_values_selected_without_truth=True)
    result=dict(scope='INFORMATION_ONLY at a fixed observation-estimated point',noise_model=meta['noise_model'],sigma_mm=meta['sigma_mm'],cycles=meta['cycles'],DIC_scalars=5*len(meta['cycles'])*32,candidates=rows,model_difference=differences(ads,dummyfits))
    (out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return clean(result)


def expressibility_diagnostics():
    # Privileged functional bounds ONLY, never used as fit targets or selector input.
    x=np.geomspace(1e-3,15.,200);phi=CorrectionBasis().evaluate(x,np.full(len(x),.12),np.zeros((len(x),2)),-1.)
    target=np.log1p(3*x/(1+x));f=least_squares(lambda a:2*np.tanh(phi@a)-target,[.4,.1],bounds=(-4.,4.))
    s1=dict(oracle_log_factor_RMSE=float(np.sqrt(np.mean(f.fun*f.fun))),max_log_factor_error=float(np.max(np.abs(f.fun))),coefficients=f.x,scope='privileged energy-domain approximation only; not a DIC recovery result')
    return dict(scope='ORACLE FUNCTIONAL EXPRESSION DIAGNOSTIC; not training, inference, or independent validation',S1=s1,
        S2=dict(reason='true shifted threshold x<=0.5 yields zero hazard while unshifted prior x>0 is positive; bounded multiplicative factor cannot create exact zero',minimum_allowed_factor=float(np.exp(-2)),zero_hazard_labels_used_for_training=False),
        S3=dict(reason='energy-only K2 has identical output at equal x and different R; mean-stress multiplier cannot generally be represented',same_x_target_log_spread=.55),
        S4=dict(reason='energy-only K2 lacks D dependence at equal x',same_x_D_range=[.05,.3],target_log_spread=.75),
        S5=dict(reason='energy-only K2 lacks spatial coordinate dependence at equal x',target_log_spread=.6),
        derivative_or_nuisance_information_cannot_fix_an_excluded_functional_dependence=True)


def temporal_diagnostics(folder,physical,out):
    out.mkdir(parents=True,exist_ok=False);meta,patterns,_=read_observations(folder);steps=(2.,1.,.5);arr={};rows=[]
    cycles=(0.,20.,40.,60.,70.,120.,220.)
    for step in steps:
        ps=predictors(meta,patterns,'B5',step=step);vals=[];insertion=[]
        for p in ps:
            z=p.parameter_space.encode(physical);states=p.damage_states(z,cycles);more=p.damage_states(z,(0.,11.3,20.,31.7,40.,60.,70.,120.,220.))
            vals.append([states[n].damage.values for n in cycles]);insertion.append(max(float(np.max(np.abs(states[n].damage.values-more[n].damage.values))) for n in cycles))
        arr[str(step)]=np.asarray(vals);rows.append(dict(step=step,query_insertion_max_difference=max(insertion),FEM_solves=sum(p.mechanics.calls for p in ps)))
    w=patterns[0].quadrature_layout.physical_weights;w=w/w.sum();errs=[float(np.sqrt(np.mean(np.sum((arr[str(s)]-arr['0.5'])**2*w,axis=-1)))) for s in (2.,1.)]
    np.savez_compressed(out/'fields.npz',**{'D_step_'+s:v for s,v in arr.items()})
    result=dict(cycles=cycles,steps=rows,damage_RMSE_vs_half_step=errs,parameters_from_observed_prefix=physical,read_future_measurements=False,same_hazard_function=True)
    (out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return clean(result)


def run_diagnostics(root,config):
    root=Path(root);start=time.perf_counter();out=root/'diagnostics';out.mkdir(exist_ok=False)
    record=json.loads((root/'inference'/'p2f-dev-002-multi_long'/'inference.json').read_text());parameters={m:c['parameters'] for m,c in record['candidates'].items()}
    results=dict(static=static_diagnostics(root/'observations'/'p2f-dev-002-with_reaction_healthy',out/'static',config),expressibility=expressibility_diagnostics())
    results['information_only']={n:information_only(root/'observations'/('p2f-dev-002-'+n),parameters,out/n) for n in ('dense_history','spatial_noise','high_noise')}
    results['temporal']=temporal_diagnostics(root/'observations'/'p2f-dev-002-multi_long',parameters['B5'],out/'temporal')
    results['wall_seconds']=time.perf_counter()-start;(out/'summary.json').write_text(json.dumps(clean(results),indent=2)+'\n');print('Static / nuisance / functional / temporal diagnostics saved',flush=True)
