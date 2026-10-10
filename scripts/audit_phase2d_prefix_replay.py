"""Independent replay of fitted-prefix and held-history residuals, without truth IO."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from fatigue_dic_rl.correction.robust_estimation import acquisition_for,split_batches
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter


def replay(task):
    case,outroot,oldroot=task;old=Path(oldroot)/'raw'/case['case_id'];out=Path(outroot)/case['case_id'];out.mkdir(parents=True,exist_ok=False);inference=json.loads((old/'inference.json').read_text());rows=[]
    for method in ('B0','B1','B2','B5'):
        acq=acquisition_for(case['path'],method,inference['options']);saved={}
        for stage,cycles in [('history',(0.,10.)),('filtered',(0.,10.,20.))]:
            path=old/method/(stage+'_fit.json')
            if not path.exists():continue
            fit=json.loads(path.read_text());adapter=MultiTrajectoryAdapter(acq.predictors,split_batches(acq,cycles));z=acq.predictors[0].parameter_space.encode(fit['parameters']);residual=adapter.residual(z);data_cost=float(.5*residual@residual);difference=abs(data_cost-fit['data_cost']);assert difference<=1e-8*max(1.,abs(data_cost));saved[stage+'_residual']=residual;saved[stage+'_prediction_mm']=adapter.displacement_prediction(z)
            row=dict(method=method,stage=stage,data_cost_difference=difference,parameter_initialization_unchanged=True)
            if stage=='history':
                validation=MultiTrajectoryAdapter(acq.predictors,split_batches(acq,(20.,)));v=validation.residual(z);saved['held_history_innovation_whitened']=-v;saved['held_history_prediction_mm']=validation.displacement_prediction(z);expected=inference['candidates'][method]['validation_square'];delta=abs(float(v@v)-expected);assert delta<=1e-8*max(1.,expected);row['validation_square_difference']=delta
            rows.append(row)
        np.savez_compressed(out/(method+'_prefix.npz'),**saved)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--evaluation',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases=[r for r in json.loads((Path(a.dataset)/'manifest.json').read_text())['records'] if r['split'] in ('test','ood','diagnostic')];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(replay,[(r,str(out),a.evaluation) for r in cases]))
    (out/'summary.json').write_text(json.dumps(dict(physical_datasets=len(rows),fit_stage_rows=sum(len(r) for r in rows),truth_io=False,passed=True,wall_seconds=time.perf_counter()-t,rows=rows),indent=2)+'\n')
