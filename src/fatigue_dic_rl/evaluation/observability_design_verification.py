"""Paired physical-world experiment evaluation; privileged scoring happens last.

The designer and estimator never receive the generator specification, truth
trajectory, family label, or future measurement. New synthetic observations are
acquired only after a public experiment is committed to disk. Forecast arrays
are committed before the separate producer generates any held-out truth.
"""
from pathlib import Path
import json,time,traceback
import numpy as np
from fatigue_dic_rl.correction.robust_estimation import clean
from fatigue_dic_rl.correction.design_library import virtual_library
from fatigue_dic_rl.inference.designed_model_estimator import fit_prefix,choose_public_design,assimilate_design
from fatigue_dic_rl.evaluation.designed_experiment_truth import acquisition_at_twenty,actual_rollout


def damage_metrics(estimate,truth,weights):
    error=np.asarray(estimate)-np.asarray(truth);w=np.asarray(weights);w=w/w.sum()
    mse=np.sum(error**2*w,axis=-1)
    return dict(rmse=np.sqrt(np.mean(mse,axis=0)).tolist(),mae=np.mean(np.sum(np.abs(error)*w,axis=-1),axis=0).tolist(),maximum=np.max(np.abs(error),axis=(0,-1)).tolist())


def evaluate_world(task):
    folder,output,config=task;folder=Path(folder);out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();meta=json.loads((folder/'public.json').read_text());conditions=json.loads((folder/'public_conditions.json').read_text());arms={}
    if set(conditions)!={'estimation_options','future_R'}:raise ValueError('public conditions contract failed')
    try:
        belief=fit_prefix(folder,out/'prefix',options=conditions['estimation_options'],max_nfev=config['design']['max_nfev']);t=time.perf_counter()
        library,pool,fields,calls=virtual_library(belief.patterns,belief.batches,belief.metadata,belief.options,belief.fits,amplitudes=config['design']['amplitudes'],ratios=config['design']['stress_ratios'],future_R=conditions['future_R'],damage_limit=config['design']['max_damage'],stress_limit=config['design']['max_nominal_stress_n_mm2'])
        library_seconds=time.perf_counter()-t;(out/'public_design_library.json').write_text(json.dumps(clean(library),indent=2)+'\n');np.savez_compressed(out/'public_virtual_fields.npz',coordinates_mm=pool,**fields)
        strategies=list(config['design']['strategies'])
        if meta['case_id'] in config['design'].get('secondary_reaction_cases',[]):strategies.append('discrimination_reaction')
        for strategy in strategies:
            arm=out/strategy;arm.mkdir(exist_ok=False);a_start=time.perf_counter();reaction=strategy=='discrimination_reaction';design=choose_public_design(library,strategy='discrimination' if reaction else strategy,seed=meta['seed']+240001)
            if reaction:design=dict(design,indices=design['indices'][:15],point_count=15,reaction_count=5,measurement_cost=160,secondary='paired load-control force; replaces one DIC point per specimen')
            (arm/'committed_design.json').write_text(json.dumps(clean(design),indent=2)+'\n')
            try:
                if not design['safe']:raise ValueError('no safe planned acquisition; ABSTAIN without experiment')
                idx=design['indices'];new,current_truth=acquisition_at_twenty(folder,belief.patterns,design,pool,idx,belief.options,future_R=conditions['future_R'],reaction=reaction)
                for i,b in enumerate(new):
                    b.frames[0].case.measurement.save(arm/'acquired_observation'/str(i))
                    if reaction:(arm/'acquired_observation'/str(i)/'reaction.json').write_text(json.dumps(dict(value_n=b.frames[0].case.reaction.value_n,standard_deviation_n=.01,component=0))+'\n')
                # Scoring coordinates are a predeclared spatial holdout, never measured.
                held=np.array([(x,y) for x in np.linspace(.09,1.91,8) for y in np.linspace(.07,.93,8)])
                forecasts,inference=assimilate_design(belief,design,new,held,arm/'inference',future_R=conditions['future_R'],max_nfev=config['design']['max_nfev'],decision_z=config['selection']['decision_z'],minimum_separation=config['selection']['minimum_separation'])
                for name,f in forecasts.items():np.savez_compressed(arm/(name+'_forecast_before_truth.npz'),damage=f['damage'],current_damage=f['current_damage'],physical_u=f['physical_u'],registered_u=f['registered_u'],cycles=f['cycles'],coordinates_mm=held)
                # Held-out truth is first generated here, after every candidate forecast.
                truth=actual_rollout(folder,belief.patterns,design,held,cycles=(30.,40.,70.,120.,220.),future_R=conditions['future_R']);np.savez_compressed(arm/'evaluation_truth.npz',**{k:v for k,v in truth.items() if k!='parameters'})
                weights=truth['weights'];methods={};signal=float(np.sqrt(np.mean(current_truth['u'][:,0,idx]**2)));noise=float(np.sqrt(np.mean(np.diag(new[0].frames[0].case.measurement.covariance.matrix))));damage_increment=float(np.mean(np.sum((current_truth['damage'][:,0]-current_truth['initial_experiment_damage'])*weights/weights.sum(),axis=-1)))
                for name,f in forecasts.items():
                    dm=damage_metrics(f['damage'],truth['damage'],weights);cm=damage_metrics(f['current_damage'],current_truth['damage'][:,0],weights);error=f['physical_u']-truth['u'];methods[name]=dict(current_damage_rmse=cm['rmse'],forecast_damage_rmse=dm['rmse'],forecast_damage_mae=dm['mae'],forecast_damage_max_error=dm['maximum'],heldout_displacement_rmse_mm=np.sqrt(np.mean(error**2,axis=(0,2,3))).tolist(),irreversibility_violations=int(np.sum(np.diff(np.concatenate([f['current_damage'][:,None],f['damage']],axis=1),axis=1)<-1e-13)),parameters=f['parameters'],fit_status=f['fit_status'])
                ref='B2';ratio=methods['E9']['forecast_damage_rmse'][-1]/max(methods[ref]['forecast_damage_rmse'][-1],1e-15) if ref in methods else None
                stress=float(max(meta['load_scales'])*10*design['amplitude']);arms[strategy]=dict(status='COMPLETED',design=clean(design),inference=inference,methods=methods,ratio_E9_vs_refit=ratio,signal_diagnostics_evaluation_only=dict(displacement_rms_mm=signal,noise_rms_mm=noise,amplitude_SNR=signal/noise,model_separation_before_measurement=design['projected_separation'],actual_N10_N20_mean_damage_increment=damage_increment,actual_N20_max_damage=float(current_truth['damage'].max())),cost=dict(measurement_scalar_components=155 if reaction else 160,measurement_cost_units=160,reaction_components=5 if reaction else 0,fatigue_cycles_per_specimen=10.,specimens_per_world=5,world_specimen_cycles=50.,peak_nominal_stress_n_mm2=stress,R=design['R'],squared_stress_range_cycle_proxy=sum((10*s*design['amplitude']*(1-design['R']))**2*10 for s in meta['load_scales']),library_seconds=library_seconds,library_FEM_solves=calls,arm_seconds=time.perf_counter()-a_start),leakage_audit=dict(design_before_measurement=True,forecast_before_heldout_truth=True,selected_through_cycle=20.,future_measurements_loaded=0))
            except Exception as exc:
                (arm/'failure_traceback.txt').write_text(traceback.format_exc());arms[strategy]=dict(status='FAILED',reason=type(exc).__name__+': '+str(exc),design=clean(design),wall_seconds=time.perf_counter()-a_start)
            (arm/'metrics.json').write_text(json.dumps(clean(arms[strategy]),indent=2)+'\n')
        result=dict(case_id=meta['case_id'],split=meta['split'],status='COMPLETED' if all(a['status']=='COMPLETED' for a in arms.values()) else 'PARTIAL',sigma_mm=meta['sigma_mm'],noise_model=meta['noise_model'],options=conditions['estimation_options'],future_R=conditions['future_R'],arms=arms,wall_seconds=time.perf_counter()-started,history_seconds=belief.history_seconds,history_FEM_solves=belief.history_fem_solves,library_seconds=library_seconds,library_FEM_solves=calls)
    except Exception as exc:
        (out/'failure_traceback.txt').write_text(traceback.format_exc());result=dict(case_id=meta['case_id'],split=meta['split'],status='FAILED',reason=type(exc).__name__+': '+str(exc),arms=arms,wall_seconds=time.perf_counter()-started)
    # Labels are attached by the scorer only, once the complete causal run ends.
    privileged=json.loads((folder/'synthetic_truth.json').read_text());result['evaluation_only_family']=privileged['family'];(out/'metrics.json').write_text(json.dumps(clean(result),indent=2)+'\n');return result
