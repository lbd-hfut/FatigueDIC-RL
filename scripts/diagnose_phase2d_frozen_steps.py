"""Post-evaluation integration audit; coefficients/selection never change.

Oracle reference is only a synthetic numerical diagnostic, unavailable to inference.
The three cases are the first declared matched case and first two S1 cases at
the legacy noise level, independent of their forecast errors.
"""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from fatigue_dic_rl.correction.robust_estimation import acquisition_for,clean
from fatigue_dic_rl.correction.law_family_benchmarks import ReferenceLawSpecification
from fatigue_dic_rl.evaluation.correction_metrics import damage_metrics


def check(task):
    case,evaluation,outroot=task;t=time.perf_counter();folder=Path(case['path']);out=Path(outroot)/case['case_id'];out.mkdir(parents=True,exist_ok=False)
    formal=json.loads((Path(evaluation)/'raw'/case['case_id']/'record.json').read_text());times=(20.,30.,40.,70.,120.,220.)
    parameters=json.loads((folder/'synthetic_truth.json').read_text());reference=ReferenceLawSpecification(parameters['family'],parameters['A'],parameters['m'],parameters['strength']);rows=[];fields={}
    weights=np.load(folder/'synthetic_trajectory.npz',allow_pickle=False)['physical_weights_mm2']
    # Fresh consistent N0 rollout for the oracle, at a finer reference resolution.
    acq=acquisition_for(folder,'B0',dict(step=.25));truth=[]
    for p in acq.predictors:
        p.law_spec=reference;z=p.parameter_space.encode(parameters);states=p.damage_states(z,times);truth.append([states[n].damage.values for n in times])
    truth=np.asarray(truth);fields['fine_oracle_damage']=truth
    for name in ('B0','B2','B5'):
        physical=formal['methods'][name]['parameters'];paths=[]
        for step in (2.,1.,.5):
            acq=acquisition_for(folder,name,dict(formal['inference']['options'],step=step));values=[]
            for p in acq.predictors:
                z=p.parameter_space.encode(physical);states=p.damage_states(z,times);values.append([states[n].damage.values for n in times])
            damage=np.asarray(values);fields[f'{name}_step_{step}']=damage
            error=[damage_metrics(damage[:,j],truth[:,j],weights)['rmse'] for j in range(len(times))];paths.append(damage)
            rows.append(dict(method=name,step_cycles=step,damage_rmse_vs_fine_oracle=error,coefficients_unchanged=True))
        for i in (0,1):
            rows[-3+i]['N220_difference_to_next_finer']=damage_metrics(paths[i][:,-1],paths[i+1][:,-1],weights)['rmse']
    np.savez_compressed(out/'fields.npz',cycles=times,**fields)
    result=dict(case_id=case['case_id'],family=case['family'],diagnostic_only=True,no_refit=True,no_selection_change=True,reference_step_cycles=.25,rows=rows,wall_seconds=time.perf_counter()-t)
    (out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--evaluation',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];cases=[next(r for r in records if r['split']=='test' and r['family']=='matched' and r['sigma_mm']==2e-7)]+[r for r in records if r['split']=='test' and r['family']=='S1' and r['sigma_mm']==2e-7][:2]
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(check,[(r,a.evaluation,str(out)) for r in cases]))
    (out/'summary.json').write_text(json.dumps(clean(dict(rows=rows,wall_seconds=time.perf_counter()-t,scope='frozen parameters; step refinement is not model calibration')),indent=2)+'\n');print('frozen coefficient step audit complete',flush=True)
