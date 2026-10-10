"""Case/seed-level metrics; never counts FE/DIC points as independent trials."""
import numpy as np
from .belief_metrics import coverage_summary


def parameter_statistics(errors,standard_deviations=None):
    e=np.asarray(errors);out={'bias':e.mean(axis=0).tolist(),'rmse':np.sqrt(np.mean(e*e,axis=0)).tolist()}
    if standard_deviations is not None:
        s=np.asarray(standard_deviations);out.update(coverage=[coverage_summary(e[:,i],s[:,i]) for i in range(e.shape[1])],mean_interval_width=(2*1.95996398454*s.mean(axis=0)).tolist(),standardized_error_rmse=np.sqrt(np.mean((e/s)**2,axis=0)).tolist())
    return out


def summarize_records(records):
    good=[r for r in records if r['status']=='SUCCESS'];total=len(records)
    out={'count':total,'success_count':len(good),'failure_frequency':1-len(good)/total,'failures':[r for r in records if r['status']!='SUCCESS']}
    out['unconverged_optimization_count']=sum(r.get('optimization_converged') is False for r in records)
    if not good:return out
    errors=[r['parameter_error'] for r in good if r.get('parameter_error') is not None]
    sd=[r['parameter_sd'] for r in good if r.get('parameter_sd') is not None]
    if errors:out['parameters']=parameter_statistics(errors,sd if len(sd)==len(errors) else None)
    raw_sd=[r['raw_parameter_sd'] for r in good if r.get('raw_parameter_sd') is not None]
    if errors and len(raw_sd)==len(errors):out['raw_parameters']=parameter_statistics(errors,raw_sd)
    for key in ['current_damage_rmse','current_damage_coverage','current_damage_interval_width','current_displacement_rmse_mm','heldout_spatial_rmse_mm','online_seconds','network_seconds','fem_solves','forecast_seconds','uncertainty_seconds']:
        a=[r[key] for r in good if r.get(key) is not None]
        out[key+'_mean']=float(np.mean(a)) if a else None
    for key in ['forecast_damage_rmse','forecast_displacement_rmse_mm','damage_coverage','mean_observation_coverage','predictive_observation_coverage','normalized_predictive_nis','forecast_damage_interval_width','forecast_mean_displacement_interval_width_mm','forecast_predictive_displacement_interval_width_mm']:
        a=[r[key] for r in good if r.get(key) is not None]
        out[key+'_mean']=np.asarray(a).mean(axis=0).tolist() if a else None
    return out
