"""Observation-only Phase2E prefix fitting, experiment choice and causal selection.

E9 is a conditional plug-in evidence screen with ABSTAIN, not a calibrated
posterior model probability. The Phase2D B7 score/rank/threshold functions are
called unchanged on the exact same observations and physical experiment.
"""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import time,json
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.designed_acquisition import METHODS,public_prefix,predictors_for,public_designs,combined_batches
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter,fit_correction
from fatigue_dic_rl.correction.robust_estimation import clean,persist_fit
from fatigue_dic_rl.correction.identifiability_gate import observability
from fatigue_dic_rl.correction.model_selection import HistoricalCandidate,select_model
from fatigue_dic_rl.correction.experimental_design import DesignScore,choose_experiment
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian
from fatigue_dic_rl.inference.belief_state import MechanicsBelief,EstimateRole,StatePrediction
from fatigue_dic_rl.inference.forecast import ForecastResult


@dataclass
class PrefixExperimentBelief:
    metadata: dict
    patterns: tuple
    batches: tuple
    options: dict
    fits: dict
    history_seconds: float
    history_fem_solves: int


def fit_prefix(folder,output,*,options=None,max_nfev=80):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);options=options or {};meta,patterns,batches=public_prefix(folder,options);fits={};t=time.perf_counter();calls=0
    for name in METHODS:
        ps=predictors_for(patterns,name,meta,options);adapter=MultiTrajectoryAdapter(ps,batches);nom=dict(d0=.12,A=8e-5,m=1.,c0=0.,c1=0.,tx=0.,ty=0.,theta=0.,load_factor=1.,E0=2050.);z=adapter.parameter_space.encode(nom);fit=fit_correction(adapter,initializations=[z],max_nfev=max_nfev);fits[name]=fit;persist_fit(out/name,'N0_N10',fit,adapter);calls+=sum(p.mechanics.calls for p in ps)
    return PrefixExperimentBelief(meta,patterns,batches,dict(options),fits,time.perf_counter()-t,calls)


def choose_public_design(library,*,strategy,seed):
    if strategy in ('fixed','uniform'):
        name='original' if strategy=='fixed' else 'uniform';row=next(r for r in library if r['amplitude']==1. and r['R']==-1. and r['point_set']==name)
        return dict(row,status='FIXED_PUBLIC_DESIGN' if row['safe'] else 'UNSAFE_FIXED_DESIGN')
    scores=[DesignScore(r['design_id'],r['fisher_score'],r['projected_separation'],160,10.,r['safe'],r['derivative_checked']) for r in library];decision=choose_experiment(scores,strategy=strategy,seed=seed)
    if decision['design_id'] is None:
        row=next(r for r in library if r['amplitude']==1. and r['R']==-1. and r['point_set']=='original');return dict(row,status='ABSTAIN_DESIGN',design_reason=decision['reason'])
    return dict(next(r for r in library if r['design_id']==decision['design_id']),status=decision['status'])


def projected_validation_choice(rows, *, decision_z=2.3263478740408408,minimum_separation=3.,minimum_singular=3.):
    """Fixed engineering/conditional-normal screen; independently tested later."""
    baseline=rows['B0'];m=len(baseline['residual']);adequacy=m+3*np.sqrt(2*m)
    if baseline['validation_square']<=adequacy:
        weak=rows['B2']['combined_minimum_singular']<minimum_singular and rows['B5']['combined_minimum_singular']<minimum_singular
        return dict(decision='ABSTAIN' if weak else 'KEEP_PHYSICS',deployed_model='B0',reason='physics predictive residual adequate; weak family information' if weak else 'physics predictive residual adequate; not unique law identification',threshold=adequacy)
    eligible={n:r['converged'] and not r['boundary'] and r['derivative_checked'] and r['combined_minimum_singular']>=minimum_singular for n,r in rows.items()}
    params=[n for n in ('B1','B2') if eligible[n]]
    param=min(params,key=lambda n:rows[n]['validation_square']+2*rows[n]['dimension']) if params else 'B0'
    corrected=rows['B5']
    if not eligible['B5'] or param=='B0':return dict(decision='ABSTAIN',deployed_model=param,reason='insufficient nuisance/parameter information',eligible=eligible)
    d=corrected['prediction']-rows[param]['prediction'];jn=rows[param]['nuisance_jacobian'];proj=project_whitened_difference(d,jn,numerical_resolution=rows[param].get('numerical_resolution',0.));norm=float(np.linalg.norm(proj.projected_difference))
    innovation=-rows[param]['residual'];mid_innovation=innovation-d/2.;score=float(proj.projected_difference@mid_innovation/norm) if norm>1e-12 else 0.
    if norm>=minimum_separation and score>decision_z and corrected['validation_square']<rows[param]['validation_square']:
        return dict(decision='CORRECTION_EVIDENCE',deployed_model='B5',projected_separation=norm,conditional_score=score,eligible=eligible,reason='independent N20, projected plug-in evidence screen; no model posterior claim')
    return dict(decision='ABSTAIN',deployed_model=param,projected_separation=norm,conditional_score=score,eligible=eligible,reason='material family not separated; deploy accepted simple calibration')


def assimilate_design(belief,design,new_batches,coordinates,output,*,future_R=-1.,max_nfev=80,decision_z=2.3263478740408408,minimum_separation=3.):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();rows={};history_candidates=[];psets={};before_calls=0;full_fits={};failures=[]
    for name in METHODS:
        ps=predictors_for(belief.patterns,name,belief.metadata,belief.options,amplitude=design['amplitude'],R=design['R'],future_R=future_R);psets[name]=ps;adapter=MultiTrajectoryAdapter(ps,new_batches);old=belief.fits[name];z=adapter.parameter_space.encode(old.physical);prediction=adapter.whitened_prediction(z);residual=adapter.residual(z)
        try:s=centered_jacobian(adapter.whitened_prediction,z);j=s.jacobian;checked=s.converged;resolution=s.absolute_error_norm
        except Exception as exc:j=np.zeros((len(residual),len(z)));checked=False;resolution=0.;failures.append(dict(method=name,stage='prospective_sensitivity',reason=str(exc)))
        combined=np.vstack([old.uncertainty.jacobian_whitened,j]);sv=np.linalg.svd(combined,compute_uv=False);o=observability(old.uncertainty,z);names=adapter.parameter_space.names;nuis=[i for i,n in enumerate(names) if n in ('tx','ty','theta','E0','load_factor')]
        rows[name]=dict(prediction=prediction,residual=residual,validation_square=float(residual@residual),validation_components=len(residual),nuisance_jacobian=j[:,nuis],combined_minimum_singular=float(sv[-1]),dimension=len(z),derivative_checked=checked,converged=old.status=='CONVERGED',boundary=o['boundary'],numerical_resolution=resolution)
        history_candidates.append(HistoricalCandidate(name,float(residual@residual),len(residual),o['practical_rank'],o['dimension'],old.status=='CONVERGED',o['boundary'],o['derivative_checked']))
        np.savez_compressed(out/(name+'_prospective_information.npz'),prediction_white=prediction,innovation_white=-residual,J_future=j,J_combined=combined,F_combined=combined.T@combined)
    b7=select_model(history_candidates);e9=projected_validation_choice(rows,decision_z=decision_z,minimum_separation=minimum_separation);selection_seconds=time.perf_counter()-t
    requested=set(['B2','B5',b7['selected'],e9['deployed_model']]);requested.add('B0') # retained public physics fallback, all budgets use the same fitting cap
    for name in sorted(requested):
        adapter=MultiTrajectoryAdapter(psets[name],combined_batches(belief.batches,new_batches,belief.options))
        try:
            fit=fit_correction(adapter,initializations=[adapter.parameter_space.encode(belief.fits[name].physical)],max_nfev=max_nfev);persist_fit(out/name,'full_N0_N10_N20',fit,adapter);full_fits[name]=fit
        except Exception as exc:
            failures.append(dict(method=name,stage='full_fit',reason=type(exc).__name__+': '+str(exc)))
            if hasattr(exc,'optimization'):(out/(name+'_failed_optimization.json')).write_text(json.dumps(clean(asdict(exc.optimization)),indent=2)+'\n')
    if 'B0' not in full_fits:raise ContractError('no consistent physical fallback; failed full histories retained')
    for label,selection,key in [('B7',b7,'selected'),('E9',e9,'deployed_model')]:
        chosen=selection[key]
        if chosen not in full_fits or chosen!='B0' and not observability(full_fits[chosen].uncertainty,full_fits[chosen].normalized)['eligible']:
            selection.update(deployment_rejected=chosen);selection[key]='B0'
            if label=='E9':selection['decision']='ABSTAIN';selection['reason']='full-prefix information/convergence/boundary rejection'
    forecast_start=time.perf_counter();forecasts={}
    for name,fit in full_fits.items():
        try:forecasts[name]=forecast_candidate(psets[name],fit,belief.metadata,coordinates)
        except Exception as exc:failures.append(dict(method=name,stage='forecast',reason=str(exc)))
    if 'B0' not in forecasts:raise ContractError('public physics forecast failed')
    for label,selection,key in [('B7',b7,'selected'),('E9',e9,'deployed_model')]:
        if selection[key] not in forecasts:selection.update(deployment_rejected=selection[key]);selection[key]='B0';e9['decision']='ABSTAIN'
        forecasts[label]=forecasts[selection[key]]
    cost=dict(selection_seconds=selection_seconds,forecast_seconds=time.perf_counter()-forecast_start,comparison_seconds=time.perf_counter()-t,FEM_solves=sum(p.mechanics.calls for ps in psets.values() for p in ps),history_shared_seconds=belief.history_seconds,history_shared_fem_solves=belief.history_fem_solves,failures=failures)
    record=dict(B7=b7,E9=e9,candidates={n:{k:v for k,v in row.items() if k not in ('prediction','residual','nuisance_jacobian')} for n,row in rows.items()},cost=cost,design={k:v for k,v in design.items() if k!='indices'},parameter_initialization='prefix MAP only',design_conditioned_through_N=10.,model_selection_through_N=20.,future_cycles=[30.,40.,70.,120.,220.],future_truth_read=False,uncertainty='data-only local information; NOT CALIBRATED')
    (out/'inference.json').write_text(json.dumps(clean(record),indent=2)+'\n');return forecasts,clean(record)


def forecast_candidate(predictors,fit,meta,coordinates):
    cycles=(30.,40.,70.,120.,220.);damage=[];current=[];u=[];forecasts=[];beliefs=[];physical_u=[]
    from fatigue_dic_rl.observations.registration import RegistrationError
    for i,p in enumerate(predictors):
        states=p.damage_states(fit.normalized,(20.,*cycles));state=states[20.];current.append(state.damage.values);beliefs.append(MechanicsBelief(p.parameter_space.names,fit.normalized,[fit.physical[n] for n in p.parameter_space.names],fit.uncertainty,state.damage,20.,EstimateRole.FILTERED,fit.evidence_ids,None,'Phase2E full-prefix plug-in MAP'))
        designs=public_designs((p.pattern,),coordinates,meta,cycle=30.,amplitude=meta['future_load_factor']);preds=[];dd=[];uu=[];pu=[]
        for n in cycles:
            d=replace(designs[0],observation_id=f'forecast-{i}-{n:g}',cycle_count=n);obs=p.observe(fit.normalized,d,states[n],role=EstimateRole.FORECAST,conditioned_observation_ids=fit.evidence_ids);preds.append(StatePrediction(n,states[n].damage,(obs,),EstimateRole.FORECAST,20.,fit.evidence_ids,None,(),'not_calibrated'));dd.append(states[n].damage.values);uu.append(obs.displacement_mm);pu.append(obs.displacement_mm-RegistrationError([fit.physical.get('tx',0.),fit.physical.get('ty',0.)],fit.physical.get('theta',0.)).field(coordinates))
        forecasts.append(ForecastResult(20.,fit.evidence_ids,tuple(preds)));damage.append(dd);u.append(uu);physical_u.append(pu)
    return dict(damage=np.asarray(damage),current_damage=np.asarray(current),registered_u=np.asarray(u),physical_u=np.asarray(physical_u),cycles=np.asarray(cycles),parameters=fit.physical,beliefs=beliefs,forecasts=forecasts,fit_status=fit.status)
