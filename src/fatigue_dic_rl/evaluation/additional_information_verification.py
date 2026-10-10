"""Static FE information, exact null directions and time-step diagnostics.

These public synthetic probes do not identify a fatigue law from frozen damage.
Healthy-reference acquisition has a separately declared additional cost.
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,Parameter,ParameterSpace
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian
from fatigue_dic_rl.inverse.identifiability import fisher_diagnostics
from fatigue_dic_rl.correction.designed_acquisition import public_prefix,public_designs,predictors_for
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference,conditional_noise_monte_carlo
from fatigue_dic_rl.correction.robust_estimation import clean


def information_probes(folder,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);meta,patterns,batches=public_prefix(folder);xy=np.array([(x,y) for x in np.linspace(.06,1.94,4) for y in np.linspace(.04,.96,4)]);space=ParameterSpace((Parameter('d0',.02,.3,'1'),Parameter('E0',1500.,2600.,'N/mm^2'),Parameter('load_factor',.85,1.15,'1')),{});physical=dict(d0=.12,E0=2040.,load_factor=1.);z=space.encode(physical);rows={};adapters={}
    independent=tuple(replace(p,traction_bcs=tuple(replace(b,values=np.array([0.,b.values[0]])) for b in p.traction_bcs)) for p in patterns)
    designs=dict(original_16=public_designs(patterns,xy,meta),reaction_budget_matched_15=public_designs(patterns,xy[:15],meta,reaction=True),static_two_modes_8_each=public_designs(patterns,xy[:8],meta)+public_designs(independent,xy[:8],meta),more_64=public_designs(patterns,np.array([(x,y) for x in np.linspace(.06,1.94,8) for y in np.linspace(.04,.96,8)]),meta),healthy_extra=public_designs(patterns,xy[:15],meta,reaction=True)+public_designs(patterns,xy,meta,known_healthy=True))
    for name,ds in designs.items():
        adapter=StaticForwardAdapter(UniformDamage(),space,[d.empty_case() for d in ds]);adapters[name]=adapter;s=centered_jacobian(adapter.whitened_prediction,z);f=fisher_diagnostics(s);component_count=len(adapter.whitened_prediction(z));reaction_count=sum(d.reaction_node_ids is not None for d in ds);vector=np.array([1.,physical['E0']*.14/(1-physical['d0'])/550.,0.]);norm=float(np.linalg.norm(s.jacobian@vector)/np.linalg.norm(s.jacobian));rows[name]=dict(data_rank=f.effective_rank,singular_values=f.singular_values.tolist(),finite_difference_error=s.absolute_error_norm,FD_checked=s.converged,scalar_channels=component_count,cost_units=component_count+reaction_count,rank_threshold=f.rank_threshold,normalized_ED_null_residual=norm,cycles_advanced=0,information='frozen state; no fatigue hazard or law labels in observation')
        np.savez_compressed(out/(name+'.npz'),J_whitened=s.jacobian,F_data=s.jacobian.T@s.jacobian,null_vector=vector)
    a=adapters['original_16'];equivalent=dict(d0=1-physical['E0']*(1-physical['d0'])/2200.,E0=2200.,load_factor=1.);ua=a.displacement_prediction(z);ub=a.displacement_prediction(space.encode(equivalent));exact=dict(first=physical,second=equivalent,max_displacement_difference_mm=float(np.max(np.abs(ua-ub))),effective_E_n_mm2=physical['E0']*(1-physical['d0']))
    # Known fixed damage has exactly the same stiffness for all hazard families.
    no_law=dict(static_law_sensitivity_norm=0.,reason='StaticForwardAdapter depends only on D, E0 and current BC; no A/m/hazard input',static_does_not_observe_prior_fatigue_law=True)
    # Counterexample: local model differences entirely tangent to E/D ambiguity.
    s=centered_jacobian(a.whitened_prediction,z);d=s.jacobian[:,0];p=project_whitened_difference(d,s.jacobian[:,1:2],numerical_resolution=s.absolute_error_norm);mc=conditional_noise_monte_carlo(p,seed=242001)
    result=dict(static_information=rows,exact_equivalence=exact,static_vs_fatigue=no_law,analytic_confounded_model_difference=dict(raw_norm=float(np.linalg.norm(d)),projected_norm=float(np.linalg.norm(p.projected_difference)),rank=p.nuisance_rank,conditional_MC=mc),forceinterpretation='load-control force balance constrains load_factor; E0-D remains rank deficient without an additional healthy reference',cost_definition='one DIC scalar=1; one independent global reaction=2; healthy reference is extra budget');(out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return result


def temporal_probes(folder,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);meta,patterns,batches=public_prefix(folder);params=dict(d0=.12,c0=.45,c1=.3);physical={};fields={};difference=[]
    for step in (2.,1.,.5):
        ps=predictors_for(patterns,'B5',meta,amplitude=.7,R=0.,step=step);arr=[];query=[]
        for p in ps:
            z=p.parameter_space.encode(params);s=p.damage_states(z,(20.,30.,40.,70.,120.,220.));expanded=p.damage_states(z,(11.3,20.,23.7,30.,40.,70.,120.,220.));arr.append([s[n].damage.values for n in (20.,30.,40.,70.,120.,220.)]);query.append(max(float(np.max(np.abs(s[n].damage.values-expanded[n].damage.values))) for n in s))
        physical[str(step)]=np.asarray(arr);fields['D_step_'+str(step)]=physical[str(step)];difference.append(dict(step=step,query_insertion_max_difference=max(query)))
    weights=patterns[0].quadrature_layout.physical_weights;weights=weights/weights.sum();errors=[float(np.sqrt(np.mean(np.sum((physical[str(s)]-physical['0.5'])**2*weights,axis=-1)))) for s in (2.,1.)];result=dict(cycles=[20.,30.,40.,70.,120.,220.],step_diagnostics=difference,RMSE_vs_step_half=errors,relative_order_ratio=errors[0]/errors[1] if errors[1] else None,parameters_chosen_without_truth=params,frozen_future_measurements_loaded=0,law_invariance='same continuous hazard parameters at every numerical step; refinement changes integration error only');np.savez_compressed(out/'temporal_fields.npz',**fields);(out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return result
