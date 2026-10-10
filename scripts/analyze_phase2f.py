"""Post-pilot descriptive audit; no selectors, thresholds or new worlds are fit."""
import argparse,json,hashlib,time,csv
from pathlib import Path
import numpy as np
from fatigue_dic_rl.correction.information_sufficiency import read_observations,adapter_for,forecast_fields
from fatigue_dic_rl.correction.robust_estimation import clean
from types import SimpleNamespace


def run(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=False);summary=json.loads((root/'summary.json').read_text());rows=[];freq=dict(unconverged=0,boundary=0,weak_practical_rank=0,failed_candidates=0);rules={}
    for p,expected in json.loads((root/'plan.json').read_text())['frozen_rules'].items():
        actual=hashlib.sha256(Path(p).read_bytes()).hexdigest();rules[p]=dict(expected=expected,actual=actual,unchanged=actual==expected)
    for result in summary['results']:
        if result['status']!='COMPLETED':rows.append(dict(case_id=result['case_id'],method='ALL',status=result['status']));continue
        inf=result['inference'];freq['failed_candidates']+=len(inf['failures'])
        casefolder=root/'observations'/result['case_id'];public=json.loads((casefolder/'public.json').read_text())
        with np.load(casefolder/'evaluation_prefix_truth.npz') as z:
            w=z['weights']/z['weights'].sum();prefix_growth=float(np.mean(np.sum((z['damage'][:,-1]-z['damage'][:,0])*w,axis=-1)))
        prefix_exposure=sum((seg['end']-seg['start'])*(seg['amplitude']*(1-seg['R']))**2 for seg in public['history'] if seg['end']<=public['cutoff'])
        prefix_cycles=5*public['cutoff']
        for name,c in inf['candidates'].items():
            freq['unconverged']+=c['status']!='CONVERGED';freq['boundary']+=c['boundary'];freq['weak_practical_rank']+=c['practical_rank']<c['dimension']
        for name,s in result['scores'].items():
            c=inf['candidates'][inf['B7']['selected'] if name=='B7' else inf['E9']['deployed_model'] if name=='E9' else name];budget=result['measurement_budget'];component_count=budget['DIC_scalars']+budget['independent_reactions']
            rows.append(dict(case_id=result['case_id'],world_id=result['world_id'],family=result['family_FOR_EVALUATION_ONLY'],method=name,status=c['status'],current_damage_RMSE=s['current_damage_RMSE'],forecast_damage_RMSE=s['damage_RMSE'],last_forecast_damage_RMSE=s['damage_RMSE_by_horizon'][-1],heldout_u_RMSE_mm=s['heldout_displacement_RMSE_mm'],whitened_prefix_RMS=float(np.sqrt(2*c['data_cost']/component_count)),signal_SNR=result['signal_SNR'],rank=c['numerical_rank'],practical_rank=c['practical_rank'],dimension=c['dimension'],minimum_singular=c['minimum_singular_value'],boundary=c['boundary'],deployed=inf['B7']['selected'] if name=='B7' else inf['E9']['deployed_model'] if name=='E9' else name,decision=inf['E9']['decision'] if name=='E9' else '',measurement_cost=budget['total_cost_units'],FEM_solves_workflow=inf['FEM_solves'],wall_seconds_workflow=inf['wall_seconds'],prefix_damage_growth=prefix_growth,prefix_specimen_cycles=prefix_cycles,amplitude_range_squared_cycle_proxy_per_specimen=prefix_exposure))
    fields=sorted(set(k for r in rows for k in r))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    # Every predeclared bounded profile forecast is built from prefix fit logs.
    # No truth or law label is opened in this forecasting section.
    folder=root/'observations'/'p2f-dev-002-multi_long';fitfolder=root/'inference'/'p2f-dev-002-multi_long'
    meta,patterns,batches=read_observations(folder);main=json.loads((fitfolder/'inference.json').read_text());held=np.array([(x,y) for x in np.linspace(.09,1.91,8) for y in np.linspace(.07,.93,8)])
    forecasts=[]
    for label,prof in main['profiles'].items():
        method,param=label.split(':');ad,ps=adapter_for(meta,patterns,batches,method)
        for item in prof:
            if 'physical' not in item:forecasts.append(dict(label=label,value=item['value'],status=item['status']));continue
            f=SimpleNamespace(normalized=ad.parameter_space.encode(item['physical']),evidence_ids=ad.evidence_ids);t=time.perf_counter();before=sum(p.mechanics.calls for p in ps);pred=forecast_fields(meta,ps,f,held);path=out/'profiles'/method/f'{param}-{item["value"]:g}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**pred)
            forecasts.append(dict(label=label,value=item['value'],status=item['status'],data_cost=item['data_cost'],physical=item['physical'],path=str(path),prediction=pred,forecast_wall_seconds=time.perf_counter()-t,forecast_FEM_solves=sum(p.mechanics.calls for p in ps)-before))
    # Scoring is isolated and happens after all profile forecasts are committed.
    with np.load(fitfolder/'evaluation_truth_after_forecast.npz') as z:truth=dict(z)
    for p in forecasts:
        if 'prediction' not in p:continue
        d=p.pop('prediction')['damage']-truth['damage'][:,1:];w=truth['weights']/truth['weights'].sum();p['damage_RMSE_by_horizon']=np.sqrt(np.mean(np.sum(d*d*w,axis=-1),axis=0))
    summary2=dict(scope='DEVELOPMENT_ONLY descriptive analysis; no independent test',independent_worlds=3,full_fit_cases=10,rows=rows,full_candidate_diagnostics=freq,profile_forecasts=forecasts,frozen_rule_integrity=rules,uncertainty_calibration=False,independent_confirmation=False)
    (out/'summary.json').write_text(json.dumps(clean(summary2),indent=2)+'\n');print('Analysis complete; all predeclared profiles retained',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',required=True);a=p.parse_args();run(a.development)
