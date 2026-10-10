"""Actual B7 deployment: all history candidates, one selected refit and rollout."""
import argparse,json,time
from pathlib import Path
from fatigue_dic_rl.correction.robust_estimation import acquisition_for,split_batches,fit_one,rollout,clean
from fatigue_dic_rl.correction.identifiability_gate import observability
from fatigue_dic_rl.correction.model_selection import HistoricalCandidate,select_model
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter


def benchmark(folder):
    t=time.perf_counter();bundles={};candidates=[]
    for name in ('B0','B1','B2','B5'):
        acq=acquisition_for(folder,name);fit,adapter=fit_one(acq,split_batches(acq,(0.,10.)));validation=MultiTrajectoryAdapter(acq.predictors,split_batches(acq,(20.,)));res=validation.residual(fit.normalized);o=observability(fit.uncertainty,fit.normalized);candidates.append(HistoricalCandidate(name,float(res@res),len(res),o['practical_rank'],o['dimension'],fit.status=='CONVERGED',o['boundary'],o['derivative_checked']));bundles[name]=(acq,fit)
    selection=select_model(candidates);selection_wall=time.perf_counter()-t;chosen=selection['selected'];acq,previous=bundles[chosen];fit,adapter=fit_one(acq,acq.batches,initial=previous.normalized)
    if chosen!='B0' and not observability(fit.uncertainty,fit.normalized)['eligible']:
        chosen='B0';acq,previous=bundles[chosen];fit,adapter=fit_one(acq,acq.batches,initial=previous.normalized)
    before=time.perf_counter();pred=rollout(acq,fit);forecast_wall=time.perf_counter()-before
    return dict(selected=chosen,history_selection_seconds=selection_wall,final_selected_fit_seconds=fit.wall_seconds,selected_forecast_seconds=forecast_wall,total_online_seconds=time.perf_counter()-t,fem_solves=sum(p.mechanics.calls for a,_ in bundles.values() for p in a.predictors),parameters=fit.physical,scope='B7 actual serial deployment; no full refit/forecast of unselected baselines; no truth access')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--evaluation',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];chosen=[next(c for c in cases if c['split']=='test' and c['family']==f and c['sigma_mm']==s) for s in (2e-7,5e-5) for f in ('matched','parameter','S1')];rows=[]
    for c in chosen:
        row=benchmark(c['path']);old=json.loads((Path(a.evaluation)/'raw'/c['case_id']/'inference.json').read_text());row.update(case_id=c['case_id'],same_selection_as_frozen_matrix=row['selected']==old['selection']['selected']);rows.append(row);(out/(c['case_id']+'.json')).write_text(json.dumps(clean(row),indent=2)+'\n');print(c['case_id'],row['selected'],row['total_online_seconds'],flush=True)
    (out/'summary.json').write_text(json.dumps(clean(rows),indent=2)+'\n')
