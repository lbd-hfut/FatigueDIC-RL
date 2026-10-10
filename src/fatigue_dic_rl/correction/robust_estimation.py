"""Observation-only TRF candidates, causal history validation, and frozen rollouts."""
from dataclasses import asdict, replace
from pathlib import Path
import json,time
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.inference.belief_state import ObservationBatch, MechanicsBelief, EstimateRole, StatePrediction
from fatigue_dic_rl.inference.forecast import ForecastResult
from fatigue_dic_rl.inverse.forward_adapter import ReactionObservation
from .acquisition import load_correction_acquisition
from .correction_optimizer import MultiTrajectoryAdapter, fit_correction
from .correction_gate import GatedLawSpecification
from .identifiability_gate import observability, conditional_singular_values
from .model_selection import HistoricalCandidate, SelectionRule, select_model, choose_shrinkage

METHODS={'B0':dict(mode='prior',unknown=('d0',)), 'B1':dict(mode='prior',unknown=('d0','A')),
         'B2':dict(mode='prior',unknown=('d0','A','m')), 'B5':dict(mode='hybrid',dimension=2)}


def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


class CountedMechanics(Tri3MechanicsSolver):
    def __init__(self):super().__init__();self.calls=0
    def solve(self,request):self.calls+=1;return super().solve(request)


def acquisition_for(folder, name, options=None):
    options=options or {}; opts=dict(METHODS[name]);opts.update({k:v for k,v in options.items() if k in ('shared_registration','load_unknown','step','dimension')})
    acquisition=load_correction_acquisition(folder,**opts)
    for p in acquisition.predictors:p.mechanics=CountedMechanics()
    if options.get('reaction'):
        # One independent reaction measurement at N0 per specimen, not duplicated per frame.
        r=json.loads((Path(folder)/'observed_reaction.json').read_text());batches=[]
        for p,b,row in zip(acquisition.predictors,acquisition.batches,r):
            first=b.frames[0];obs=ReactionObservation(p.pattern.mesh.boundary_tags['left'].node_ids,0,row['value_n'],row['std_n'])
            first=replace(first,case=replace(first.case,reaction=obs));batches.append(replace(b,frames=(first,)+b.frames[1:]))
        acquisition=replace(acquisition,batches=tuple(batches))
    return acquisition


def split_batches(acquisition, cycles):
    return tuple(ObservationBatch(tuple(f for f in b.frames if f.cycle_count in cycles),noise_contract=b.noise_contract) for b in acquisition.batches)


def report_fit(fit, adapter):
    o=observability(fit.uncertainty,fit.normalized)
    targets=[i for i,n in enumerate(adapter.parameter_space.names) if n in ('A','m') or n.startswith('c') and n[1:].isdigit()]
    return dict(status=fit.status,parameters=fit.physical,normalized=fit.normalized,data_cost=fit.data_cost,regularization_cost=fit.regularization_cost,
                wall_seconds=fit.wall_seconds,evaluation_count=adapter.evaluation_count,
                optimizer_nfev=[r.nfev for r in fit.optimization.starts],uncertainty_status=fit.uncertainty.status,warnings=fit.uncertainty.warnings,
                **o,singular_values=fit.uncertainty.singular_values_data,conditional_evolution_singular_values=conditional_singular_values(fit.uncertainty.jacobian_whitened,targets),evidence_ids=fit.evidence_ids)


def persist_fit(out, prefix, fit, adapter):
    out.mkdir(parents=True,exist_ok=True)
    (out/(prefix+'_fit.json')).write_text(json.dumps(clean(report_fit(fit,adapter)),indent=2)+'\n')
    (out/(prefix+'_optimization.json')).write_text(json.dumps(clean(asdict(fit.optimization)),indent=2)+'\n')
    u=fit.uncertainty;np.savez_compressed(out/(prefix+'_information.npz'),J=u.jacobian_whitened,Fdata=u.data_fisher,Fprior=u.prior_fisher,singular_values=u.singular_values_data,null_directions=u.data_null_directions,weak_directions=u.weak_directions)


def fit_one(acquisition,batches, *, initial=None):
    adapter=MultiTrajectoryAdapter(acquisition.predictors,batches)
    if initial is None:
        nominal=dict(d0=.12,A=8e-5,m=1.,tx=0.,ty=0.,theta=0.,load_factor=1.,**{f'c{i}':0. for i in range(8)})
        initial=adapter.parameter_space.encode(nominal)
    fit=fit_correction(adapter,initializations=[initial],max_nfev=80,coefficient_regularization=.001)
    return fit,adapter


def infer_models(folder, output, *, options=None, rule=None):
    """Fits/selection/future forecasts complete before returning; never opens truth."""
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();options=options or {};bundles={};history=[];failures=[];reports={}
    for name in METHODS:
        acq=acquisition_for(folder,name,options);train=split_batches(acq,(0.,10.));val=split_batches(acq,(20.,))
        try:
            fit,adapter=fit_one(acq,train);persist_fit(out/name,'history',fit,adapter)
            va=MultiTrajectoryAdapter(acq.predictors,val);res=va.residual(fit.normalized);o=observability(fit.uncertainty,fit.normalized)
            history.append(HistoricalCandidate(name,float(res@res),len(res),o['practical_rank'],o['dimension'],fit.status=='CONVERGED',o['boundary'],o['derivative_checked']))
            bundles[name]=dict(acq=acq,history_fit=fit,validation_adapter=va)
            reports[name]=dict(history=report_fit(fit,adapter),validation_square=float(res@res),validation_whitened_rms=float(np.sqrt(np.mean(res**2))))
        except Exception as exc:
            row=dict(method=name,stage='history',reason=type(exc).__name__+': '+str(exc));failures.append(row)
            if hasattr(exc,'optimization'):(out/(name+'_failed_optimization.json')).write_text(json.dumps(clean(asdict(exc.optimization)),indent=2)+'\n')
    if 'B0' not in bundles:
        (out/'failures.json').write_text(json.dumps(failures,indent=2));raise ContractError('physics fallback failed; candidate histories retained')
    selection=select_model(history,rule or SelectionRule());grid=[];gate_acquisitions={}
    if selection['selected']=='B5':
        a=bundles['B5']['history_fit'].physical
        for g in (0.,.25,.5,.75,1.):
            if g==0.:grid.append(next(c.validation_square for c in history if c.name=='B0'));continue
            acq=load_correction_acquisition(folder,mode='hybrid',coefficients=[a['c0'],a['c1']],unknown=('d0',),law_spec=GatedLawSpecification(gate=g),**{k:v for k,v in options.items() if k in ('shared_registration','load_unknown','step')})
            # Registration/load estimates were obtained from the training prefix only.
            
            for p in acq.predictors:p.mechanics=CountedMechanics()
            z=acq.predictors[0].parameter_space.encode(a);va=MultiTrajectoryAdapter(acq.predictors,split_batches(acq,(20.,)));r=va.residual(z);grid.append(float(r@r));gate_acquisitions[g]=acq
        shrink=choose_shrinkage(selection,grid,coefficient_norm_square=a['c0']**2+a['c1']**2,components=history[0].validation_components)
    else:shrink=choose_shrinkage(selection,[],coefficient_norm_square=0.,components=history[0].validation_components)
    # Choice is now frozen. Full-prefix refits cannot change selected model or gate.
    selection_seconds=time.perf_counter()-start
    for name,b in bundles.items():
        acq=b['acq']
        try:
            fit,adapter=fit_one(acq,acq.batches,initial=b['history_fit'].normalized);persist_fit(out/name,'filtered',fit,adapter)
            b.update(fit=fit,adapter=adapter);reports[name]['filtered']=report_fit(fit,adapter)
            from .noise_robustness import observation_diagnostics
            reports[name]['observation_diagnostics']=observation_diagnostics(acq,fit,adapter)
        except Exception as exc:
            failures.append(dict(method=name,stage='filtered',reason=type(exc).__name__+': '+str(exc)));b.update(fit=b['history_fit'],adapter=MultiTrajectoryAdapter(acq.predictors,split_batches(acq,(0.,10.))))
            reports[name]['filtered_fallback']='history-only tentative prediction, refit failed'
    outputs={};forecast_start=time.perf_counter()
    for name,b in bundles.items():
        try: outputs[name]=rollout(b['acq'],b['fit'])
        except Exception as exc: failures.append(dict(method=name,stage='forecast',reason=type(exc).__name__+': '+str(exc)))
    if 'B0' not in outputs: raise ContractError('physics forecast failed; no deployable fallback')
    chosen=selection['selected']
    if chosen not in outputs or not observability(bundles[chosen]['fit'].uncertainty,bundles[chosen]['fit'].normalized)['eligible'] and chosen!='B0':
        selection=dict(selection,selected_before_deployment_check=chosen,selected='B0',deployment_rejection='full-prefix information or physics forecast failure')
        chosen='B0';shrink=dict(gate=0.,fallback='B0',reason='deployment stability rejection')
    outputs['B7']=outputs[chosen]
    if shrink['gate']==0.:outputs['B8']=outputs[shrink['fallback']]
    elif shrink['gate']==1.:outputs['B8']=outputs['B5']
    else:
        # Apply the frozen gate to the full-prefix coefficients; re-estimate only d0/nuisance.
        a=bundles['B5']['fit'].physical;g=shrink['gate'];acq=load_correction_acquisition(folder,mode='hybrid',coefficients=[a['c0'],a['c1']],unknown=('d0',),law_spec=GatedLawSpecification(gate=g),**{k:v for k,v in options.items() if k in ('shared_registration','load_unknown','step')})
        for p in acq.predictors:p.mechanics=CountedMechanics()
        fit,adapter=fit_one(acq,acq.batches,initial=acq.predictors[0].parameter_space.encode(a));persist_fit(out/'B8','filtered',fit,adapter);outputs['B8']=rollout(acq,fit)
    total_seconds=time.perf_counter()-start
    cost=dict(selection_seconds=selection_seconds,forecast_seconds=time.perf_counter()-forecast_start,total_online_seconds=total_seconds,
              candidate_fem_solves={n:sum(p.mechanics.calls for p in b['acq'].predictors) for n,b in bundles.items()},
              gate_validation_fem_solves=sum(p.mechanics.calls for a in gate_acquisitions.values() for p in a.predictors),
              candidate_failures=failures,information='all candidate fitting, two-scale Jacobians, historical validation, full-prefix refits and forecasts included')
    record=dict(selection=selection,shrinkage=shrink,candidates=reports,cost=cost,options=options,truth_read=False,future_observations_read=False,
                selection_fit_cycles=[0.,10.],selection_validation_cycles=[20.],filtered_cycles=[0.,10.,20.],forecast_cycles=[30.,40.,70.,120.,220.],uncertainty_qualification='NOT_QUALIFIED; data Fisher only')
    (out/'inference.json').write_text(json.dumps(clean(record),indent=2)+'\n')
    return outputs,clean(record)


def rollout(acq,fit):
    origin=max(f.cycle_count for b in acq.batches for f in b.frames if f.observation_id in fit.evidence_ids)
    times=(30.,40.,70.,120.,220.);designs=acq.future_designs(cycles=times);result={k:[] for k in ('damage','physical_u','registered_u','heldout_u','hazard','factor')};current=[];beliefs=[];forecasts=[]
    meta=acq.metadata;nodes=np.asarray(meta['mesh']['nodes']);w,h=np.ptp(nodes[:,0]),np.ptp(nodes[:,1]);rng=np.random.default_rng(meta['seed']+202);held=rng.uniform([.03,.03],[1.97,.97],(64,2))*[w/2.,h]
    from fatigue_dic_rl.observations.operator import ObservationOperator
    from fatigue_dic_rl.observations.registration import RegistrationError
    from fatigue_dic_rl.fatigue import EnergyAmplitudeDriver
    for p,b,ds in zip(acq.predictors,acq.batches,designs):
        # One canonical replay of this exact candidate for every target, no old-state reuse.
        states=p.damage_states(fit.normalized,tuple(sorted({origin,20.,*times})));state=states[origin];current.append(states[20.].damage.values)
        belief=MechanicsBelief(p.parameter_space.names,fit.normalized,[fit.physical[n] for n in p.parameter_space.names],fit.uncertainty,state.damage,origin,EstimateRole.FILTERED,fit.evidence_ids,None,json.dumps(dict(physics_model_id=p.model_id,algorithm='prefix_MAP_with_history_validated_model_selection')));beliefs.append(belief)
        predictions=[];dd=[];pu=[];ru=[];hu=[];haz=[];factor=[];pattern=p._candidate_pattern(p.pattern,fit.physical);hm=ObservationOperator(pattern.mesh).prepare(held,coordinate_units='mm',coordinate_system='specimen');law=p.law_spec.build(fit.physical,state.material_ids)
        for design in ds:
            s=states[design.cycle_count];obs=p.observe(fit.normalized,design,s,role=EstimateRole.FORECAST,conditioned_observation_ids=fit.evidence_ids);predictions.append(StatePrediction(design.cycle_count,s.damage,(obs,),EstimateRole.FORECAST,origin,fit.evidence_ids,None,(),'not_requested'))
            registration=RegistrationError([fit.physical.get('tx',0.),fit.physical.get('ty',0.)],fit.physical.get('theta',0.)).field(design.coordinates_mm)
            dd.append(s.damage.values);ru.append(obs.displacement_mm);pu.append(obs.displacement_mm-registration)
            cyc=replace(p.load_history[1],N_start=s.cycle_count,delta_N=p.cycle_step);hi=p.mechanics.solve(cyc.request(pattern,s.damage,s.cycle_count,'max'));lo=p.mechanics.solve(cyc.request(pattern,s.damage,s.cycle_count,'min'));hu.append(hm.apply(hi.state.displacement).values);rate=law.hazard(EnergyAmplitudeDriver().evaluate(lo,hi),cyc,s);haz.append(rate.values)
            factor.append(law.last_diagnostics['factor'] if hasattr(law,'last_diagnostics') else np.ones(s.layout.count))
        forecasts.append(ForecastResult(origin,fit.evidence_ids,tuple(predictions)))
        for k,v in zip(result,(dd,pu,ru,hu,haz,factor)):result[k].append(v)
    return dict(**{k:np.asarray(v) for k,v in result.items()},current_damage=np.asarray(current),cycles=np.asarray(times),parameters=fit.physical,current_state_role='filtered' if origin==20. else 'prediction_from_N10',origin_cycle=origin,beliefs=beliefs,forecasts=forecasts)
