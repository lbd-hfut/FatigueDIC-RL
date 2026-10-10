"""Synthetic scoring occurs only after all truth-blind selections and forecasts."""
from pathlib import Path
import json,time
import numpy as np
from fatigue_dic_rl.correction.robust_estimation import infer_models,clean
from fatigue_dic_rl.evaluation.correction_metrics import damage_metrics


def evaluate_case(folder,output,*,options=None):
    outputs,inference=infer_models(folder,output,options=options)
    # No label, law-family or future observation IO has occurred before this line.
    folder=Path(folder);out=Path(output)
    with np.load(folder/'synthetic_trajectory.npz',allow_pickle=False) as data:truth={k:data[k].copy() for k in data.files}
    private=json.loads((folder/'synthetic_truth.json').read_text());spec=json.loads((folder/'generation_spec.json').read_text());current_index=int(np.flatnonzero(truth['cycles']==20.)[0]);indices=[int(np.flatnonzero(truth['cycles']==n)[0]) for n in outputs['B0']['cycles']];rows={}
    from fatigue_dic_rl.observations.registration import RegistrationError
    reg=RegistrationError([5e-5,-3e-5],4e-5).field(truth['coordinates_mm']) if spec.get('registration') else np.zeros_like(truth['coordinates_mm'])
    from fatigue_dic_rl.observations.noise import DICMeasurement
    observed=np.array([[DICMeasurement.load(folder/'observations'/str(i)/str(j)).observation.displacement for j in indices] for i in range(len(outputs['B0']['damage']))])
    for name,pred in outputs.items():
        weights=truth['physical_weights_mm2'];fore=pred['damage'];actual=truth['damage'][:,indices]
        metrics=[damage_metrics(fore[:,j],actual[:,j],weights) for j in range(len(indices))]
        rows[name]=dict(current_state_role=pred['current_state_role'],forecast_origin=pred['origin_cycle'],current_damage=damage_metrics(pred['current_damage'],truth['damage'][:,current_index],weights),forecast_damage_rmse=[r['rmse'] for r in metrics],forecast_damage_mae=[r['mae'] for r in metrics],forecast_damage_max_error=[float(np.max(np.abs(fore[:,j]-actual[:,j]))) for j in range(len(indices))],damage_trajectory_bias=np.mean(fore-actual,axis=(0,2)),forecast_physical_displacement_rmse_mm=np.sqrt(np.mean((pred['physical_u']-truth['displacement_mm'][:,indices])**2,axis=(0,2,3))),forecast_registered_observation_rmse_mm=np.sqrt(np.mean((pred['registered_u']-truth['displacement_mm'][:,indices]-reg)**2,axis=(0,2,3))),spatial_heldout_displacement_rmse_mm=np.sqrt(np.mean((pred['heldout_u']-truth['heldout_displacement_mm'][:,indices])**2,axis=(0,2,3))),heldout_noisy_observation_rmse_mm=float(np.nanmean((pred['registered_u']-observed)**2)**.5),hazard_rmse_synthetic_only=np.sqrt(np.mean((pred['hazard']-truth['hazard'][:,indices])**2,axis=(0,2))),factor_min=float(pred['factor'].min()),factor_max=float(pred['factor'].max()),irreversibility_violations=int(np.sum(np.diff(np.concatenate([pred['current_damage'][:,None,:],fore],axis=1),axis=1)<-1e-12)),parameters=pred['parameters'],parameter_error={n:pred['parameters'][n]-private[n] for n in ('d0','A','m')},damage_min=float(fore.min()),damage_max=float(fore.max()))
        np.savez_compressed(out/(name+'_fields.npz'),**{k:v for k,v in pred.items() if isinstance(v,np.ndarray)})
    record=dict(case_id=spec['case_id'],seed=spec['seed'],split=spec['split'],family=spec['family'],sigma_mm=spec['sigma_mm'],spec=spec,status='COMPLETED_WITH_CANDIDATE_FAILURES' if inference['cost']['candidate_failures'] else 'COMPLETED',inference=inference,methods=rows,oracle_information_used_for_selection=False,forecasts_completed_before_truth_scoring=True)
    (out/'record.json').write_text(json.dumps(clean(record),indent=2)+'\n');return clean(record)


def evaluate_task(task):
    folder,out,options=task
    try:return evaluate_case(folder,out,options=options)
    except Exception as exc:
        import traceback
        path=Path(out);path.mkdir(parents=True,exist_ok=True);(path/'failure_traceback.txt').write_text(traceback.format_exc());record=dict(path=folder,status='FAILED',reason=type(exc).__name__+': '+str(exc));(path/'failure.json').write_text(json.dumps(record,indent=2)+'\n');return record
