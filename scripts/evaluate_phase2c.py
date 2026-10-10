"""Same-acquisition classical calibration, learned correction and held-out rollouts."""
import argparse,json,tomllib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.evaluation.correction_verification import evaluate_correction_method
from fatigue_dic_rl.evaluation.identifiability_verification import clean
from fatigue_dic_rl.training.correction_dataset import source_hash


def evaluate_task(task):
    folder,method,out,kwargs=task
    try:return evaluate_correction_method(folder,method,out,**kwargs)
    except Exception as exc:
        dest=Path(out);dest.mkdir(parents=True,exist_ok=True);row={'status':'FAILED','folder':folder,'method':method,'reason':type(exc).__name__+': '+str(exc)}
        public=Path(folder)/'public.json'
        if public.exists():
            meta=json.loads(public.read_text());row.update(case_id=meta['case_id'],seed=meta['seed'],scenario=meta['case_id'][len(meta['split'])+1:].rsplit('-',1)[0])
        if hasattr(exc,'optimization'):
            from dataclasses import asdict
            (dest/'failed_optimization.json').write_text(json.dumps(clean(asdict(exc.optimization)),indent=2)+'\n')
        (dest/'failure.json').write_text(json.dumps(row,indent=2)+'\n');return row


def summarize(rows):
    import numpy as np
    result={}
    for scenario in sorted({r['scenario'] for r in rows if 'scenario' in r}):
        result[scenario]={}
        for method in sorted({r['method'] for r in rows if r.get('scenario')==scenario}):
            group=[r for r in rows if r.get('scenario')==scenario and r['method']==method];good=[r for r in group if r['status']!='FAILED']
            result[scenario][method]={'count':len(group),'failed':len(group)-len(good),'converged':sum(r['status']=='CONVERGED' for r in good),'rank_distribution':[r['data_rank'] for r in good],
                'current_damage_rmse_mean':float(np.mean([r['current_damage']['rmse'] for r in good])) if good else None,
                'forecast_damage_rmse_mean':np.mean([r['forecast_damage_rmse'] for r in good],axis=0).tolist() if good else None,
                'online_seconds_mean':float(np.mean([r['online_seconds'] for r in good])) if good else None,'fem_solves_mean':float(np.mean([r['fem_solves'] for r in good])) if good else None}
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2c/study.toml');ap.add_argument('--dataset',required=True);ap.add_argument('--training',required=True);ap.add_argument('--output',required=True);ap.add_argument('--partition',choices=['test','ood'],default='test');a=ap.parse_args();c=tomllib.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());cases=[r for r in manifest['records'] if r['split']==a.partition];tasks=[];t=time.perf_counter()
    if c['training']['maximum_nfev']!=80 or c['training']['coefficient_regularization']!=.001:
        raise ValueError('This frozen evaluator declares TRF budget80 and coefficient ridge .001; unsupported overrides must not be ignored')
    for case in cases:
        methods=['B0_physics_prior','B1_refit_A','B2_refit_Am','B5_multistep_K2']
        if case['scenario']=='structural' and a.partition=='test':methods+=['B3_single_step_K2','B5_multistep_K4','B5_multistep_K8','ablation_no_physics_K2','ablation_no_history_K2','ablation_spatial_K2']
        if a.partition=='ood':methods=['B0_physics_prior','B2_refit_Am','B5_multistep_K2']
        for method in methods:tasks.append((case['path'],method,str(out/'raw'/case['case_id']/method),{}))
        regime=case['scenario'] if a.partition=='test' else 'structural'
        for seed in c['training']['seeds']:
            dims=c['training']['dimensions'] if case['scenario']=='structural' and a.partition=='test' else [c['training']['primary_supervised_dimension']]
            for k in dims:
                checkpoint=Path(a.training)/regime/f'K{k}'/f'seed_{seed}'/'checkpoint.pt';method=f'B4_supervised_K{k}_seed_{seed}'
                tasks.append((case['path'],method,str(out/'raw'/case['case_id']/method),{'checkpoint':str(checkpoint)}))
    with ProcessPoolExecutor(max_workers=c['dataset']['workers']) as pool:rows=list(pool.map(evaluate_task,tasks))
    (out/'records.json').write_text(json.dumps(clean(rows),indent=2)+'\n');(out/'summary.json').write_text(json.dumps({'scenarios':summarize(rows),'failures':[r for r in rows if r['status']=='FAILED'],'wall_seconds':time.perf_counter()-t,'source_sha256':source_hash(),'same_acquisitions_and_public_loads':True,'uncertainty_qualified':False},indent=2)+'\n');print('evaluated',len(rows),'failed',sum(r['status']=='FAILED' for r in rows),flush=True)
