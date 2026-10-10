"""Observation-only Development fits and unmodified model-rule replay."""
from dataclasses import asdict
from pathlib import Path
import json,time
import numpy as np
from fatigue_dic_rl.correction.information_sufficiency import read_observations,adapter_for,nominal,forecast_fields,BoundedProfileAdapter
from fatigue_dic_rl.correction.correction_optimizer import fit_correction
from fatigue_dic_rl.correction.robust_estimation import clean,persist_fit,report_fit
from fatigue_dic_rl.correction.identifiability_gate import observability,conditional_singular_values
from fatigue_dic_rl.correction.model_selection import HistoricalCandidate,select_model
from fatigue_dic_rl.inference.designed_model_estimator import projected_validation_choice
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian

METHODS=('B0','B1','B2','B5')


def extra_information(fit,adapter):
    u=fit.uncertainty;j=u.jacobian_whitened;names=adapter.parameter_space.names
    cov=u.identifiable_covariance_normalized;sd=np.sqrt(np.maximum(np.diag(cov),0.))
    denom=sd[:,None]*sd[None,:]
    corr=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
    targets=[i for i,n in enumerate(names) if n in ('A','m','c0','c1')]
    initial=[i for i,n in enumerate(names) if n=='d0']
    _,s,v=np.linalg.svd(j,full_matrices=False)
    return dict(parameter_names=names,correlation_identifiable_subspace=corr,
        correlation_scope='full local covariance' if u.data_rank==len(names) else 'identifiable-subspace only; null variance unbounded, NOT full posterior correlation',
        weak_singular_directions=v[s<3.],conditional_initial_singular=conditional_singular_values(j,initial),
        conditional_evolution_singular=conditional_singular_values(j,targets),data_only=True,
        full_rank_is_not_reliable_identification=True)


def differences(adapters,fits):
    if not {'B2','B5'}<=set(fits):return dict(status='UNAVAILABLE candidates retained as failed')
    a,b=adapters['B2'],adapters['B5'];d=b.whitened_prediction(fits['B5'].normalized)-a.whitened_prediction(fits['B2'].normalized)
    result=[]
    for name in ('B2','B5'):
        j=fits[name].uncertainty.jacobian_whitened;names=adapters[name].parameter_space.names
        for scope in ('initial_and_nuisance','all_candidate_parameters'):
            idx=[i for i,n in enumerate(names) if n in ('d0','E0','load_factor')] if scope=='initial_and_nuisance' else list(range(len(names)))
            p=project_whitened_difference(d,j[:,idx],numerical_resolution=fits[name].uncertainty.finite_difference_error)
            result.append(dict(reference=name,scope=scope,raw_norm=float(np.linalg.norm(d)),projected_norm=float(np.linalg.norm(p.projected_difference)),absorbed_norm=float(np.linalg.norm(p.absorbed_difference)),nuisance_rank=p.nuisance_rank,
                components=len(d),raw_per_component_RMS=float(np.linalg.norm(d)/np.sqrt(len(d))),projected_per_component_RMS=float(np.linalg.norm(p.projected_difference)/np.sqrt(len(d))),interpretation='local conditional difference; not significance or model posterior'))
    return result


def infer_prefix(folder,output,*,max_nfev=40,profile_grids=None,initial_controls=()):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    meta,patterns,batches=read_observations(folder);training=meta['cycles'][:-1];last=meta['cycles'][-1]
    # Preserve the exact public MeshData objects: the existing physics protocol
    # deliberately rejects transferring material history to a different mesh.
    from fatigue_dic_rl.inference.belief_state import ObservationBatch
    train=tuple(ObservationBatch(tuple(f for f in b.frames if f.cycle_count in training)) for b in batches)
    val=tuple(ObservationBatch(tuple(f for f in b.frames if f.cycle_count==last)) for b in batches)
    fits={};fullad={};psets={};history={};rows={};candidates=[];failures=[];allocated=[];hist_adapters={}
    for method in METHODS:
        ps=None
        try:
            ad,ps=adapter_for(meta,patterns,train,method);allocated.append(ps);z=nominal(ad.parameter_space)
            h=fit_correction(ad,initializations=[z],max_nfev=max_nfev);history[method]=h;hist_adapters[method]=ad;persist_fit(out/method,'history',h,ad)
            va,vps=adapter_for(meta,patterns,val,method);allocated.append(vps);zh=va.parameter_space.encode(h.physical);r=va.residual(zh);prediction=va.whitened_prediction(zh)
            s,kind=checked_local_jacobian(va.whitened_prediction,zh);jn=s.jacobian;combined=np.vstack([h.uncertainty.jacobian_whitened,jn]);o=observability(h.uncertainty,h.normalized)
            indices=[i for i,n in enumerate(va.parameter_space.names) if n in ('E0','load_factor')]
            rows[method]=dict(prediction=prediction,residual=r,validation_square=float(r@r),dimension=len(zh),derivative_checked=s.converged,converged=h.status=='CONVERGED',boundary=o['boundary'],combined_minimum_singular=float(np.linalg.svd(combined,compute_uv=False)[-1]),nuisance_jacobian=jn[:,indices],numerical_resolution=s.absolute_error_norm)
            candidates.append(HistoricalCandidate(method,float(r@r),len(r),o['practical_rank'],o['dimension'],h.status=='CONVERGED',o['boundary'],o['derivative_checked']))
            full,fullps=adapter_for(meta,patterns,batches,method)
            allocated.append(fullps)
            fit=fit_correction(full,initializations=[full.parameter_space.encode(h.physical)],max_nfev=max_nfev)
            persist_fit(out/method,'full',fit,full);fits[method]=fit;fullad[method]=full;psets[method]=fullps
        except Exception as exc:
            failures.append(dict(method=method,reason=type(exc).__name__+': '+str(exc)))
            if hasattr(exc,'optimization'):(out/(method+'_failed_optimization.json')).write_text(json.dumps(clean(asdict(exc.optimization)),indent=2)+'\n')
    if 'B0' not in fits:
        (out/'candidate_failures.json').write_text(json.dumps(clean(failures),indent=2)+'\n')
        raise ValueError('physics fallback failed; raw candidate failures preserved')
    b7=select_model(candidates)
    e9=projected_validation_choice(rows) if set(rows)==set(METHODS) else dict(decision='ABSTAIN',deployed_model='B0',reason='candidate failure; frozen rule unavailable')
    for label,selection,key in [('B7',b7,'selected'),('E9',e9,'deployed_model')]:
        chosen=selection[key]
        if chosen not in fits or chosen!='B0' and not observability(fits[chosen].uncertainty,fits[chosen].normalized)['eligible']:
            selection.update(deployment_rejected=chosen);selection[key]='B0'
            if label=='E9':selection['decision']='ABSTAIN'
    held=np.array([(x,y) for x in np.linspace(.09,1.91,8) for y in np.linspace(.07,.93,8)])
    forecasts={}
    for method,fit in fits.items():
        try:forecasts[method]=forecast_fields(meta,psets[method],fit,held)
        except Exception as exc:failures.append(dict(method=method,stage='forecast',reason=str(exc)))
    for label,selection,key in [('B7',b7,'selected'),('E9',e9,'deployed_model')]:
        chosen=selection[key]
        if chosen not in forecasts:chosen='B0';selection[key]=chosen
        forecasts[label]=forecasts[chosen]
    for label,f in forecasts.items():np.savez_compressed(out/(label+'_forecast_before_truth.npz'),**f,coordinates_mm=held)
    profiles={};controls=[]
    for method,name,grid in profile_grids or ():
        if method not in fits:continue
        base=fullad[method];opt=fits[method];rr=[]
        for value in grid:
            restricted=BoundedProfileAdapter(base,name,value)
            try:
                z=restricted.parameter_space.encode(opt.physical);f=fit_correction(restricted,initializations=[z],max_nfev=30)
                persist_fit(out/'profiles'/method/f'{name}-{value:g}','profile',f,restricted)
                rr.append(dict(value=value,data_cost=f.data_cost,regularization_cost=f.regularization_cost,status=f.status,physical=f.physical,finite_difference_checked='finite_difference_check_failed' not in f.uncertainty.warnings))
            except Exception as exc:rr.append(dict(value=value,status='FAILED',reason=str(exc)))
        profiles[method+':'+name]=rr
    if 'B2' in fits:
        for d0 in initial_controls:
            base=fullad['B2']
            try:
                f=fit_correction(base,initializations=[nominal(base.parameter_space,d0=d0)],max_nfev=max_nfev)
                persist_fit(out/'initial_controls'/str(d0),'cold',f,base)
                controls.append(dict(initial_d0=d0,data_cost=f.data_cost,status=f.status,physical=f.physical))
                cf=forecast_fields(meta,psets['B2'],f,held)
                np.savez_compressed(out/'initial_controls'/str(d0)/'forecast_before_truth.npz',**cf)
            except Exception as exc:controls.append(dict(initial_d0=d0,status='FAILED',reason=str(exc)))
    record=dict(scope='DEVELOPMENT_ONLY',B7=b7,E9=e9,rule_modification=False,history_training_cycles=training,validation_cycle=last,cutoff=meta['cutoff'],future_cycles=meta['future_cycles'],
        candidates={m:dict(**report_fit(f,fullad[m]),**extra_information(f,fullad[m])) for m,f in fits.items()},model_differences=differences(fullad,fits),profiles=profiles,initial_controls=controls,failures=failures,
        FEM_solves=sum(p.mechanics.calls for ps in allocated for p in ps),wall_seconds=time.perf_counter()-start,truth_read=False,future_measurements_read=False,
        uncertainty='local approximation NOT CALIBRATED',protocol_transfer='Frozen B7/E9 functions on a newly declared history; not their original time-protocol qualification')
    (out/'inference.json').write_text(json.dumps(clean(record),indent=2)+'\n')
    return clean(record),forecasts,meta,patterns,held
