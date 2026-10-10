"""Small DEVELOPMENT-only information study. No independent test entry point."""
import argparse,hashlib,json,time,tomllib,traceback,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from fatigue_dic_rl.training.robust_correction_dataset import generate_case
from fatigue_dic_rl.evaluation.information_pilot_truth import produce_case,actual_fields
from fatigue_dic_rl.inference.information_pilot import infer_prefix
from fatigue_dic_rl.correction.robust_estimation import clean


def source_hash():
    paths=sorted(p for root in ('src','scripts','configs') for p in Path(root).rglob('*') if p.is_file() and p.suffix in ('.py','.toml','.json') and '__pycache__' not in p.parts)
    return hashlib.sha256('\n'.join(str(p)+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest()


def definitions(config):
    short=config['protocols']['short_cycles'];long=config['protocols']['long_cycles'];rows=[]
    for w in range(3):
        for name,loading,cycles in [('fixed_short','fixed',short),('multi_long','multi',long)]:
            rows.append(dict(world=w,name=name,loading=loading,cycles=cycles,options={},fit=True))
    rows.append(dict(world=2,name='fixed_long',loading='fixed',cycles=long,options={},fit=True))
    for name,force,healthy in [('unknown_material_load',False,False),('with_reaction',True,False),('with_reaction_healthy',True,True)]:
        rows.append(dict(world=2,name=name,loading='multi',cycles=long,options=dict(material_unknown=True,load_unknown=True),reaction=force,healthy=healthy,fit=True))
    for name,extra in [('dense_history',dict(cycles=config['protocols']['dense_cycles'])),('spatial_noise',dict(noise_model='spatial')),('high_noise',dict(sigma_mm=config['protocols']['high_noise_sigma_mm']))]:
        rows.append(dict(world=2,name=name,loading='multi',cycles=extra.pop('cycles',long),options={},fit=False,**extra))
    return rows


def score_fields(pred,true,weights):
    w=weights/weights.sum();d=pred['damage']-true['damage'];dc=pred['current_damage']-true['current_damage']
    return dict(damage_RMSE=float(np.sqrt(np.mean(np.sum(d*d*w,axis=-1)))),current_damage_RMSE=float(np.sqrt(np.mean(np.sum(dc*dc*w,axis=-1)))),damage_RMSE_by_horizon=np.sqrt(np.mean(np.sum(d*d*w,axis=-1),axis=0)),heldout_displacement_RMSE_mm=float(np.sqrt(np.mean((pred['u']-true['u'])**2))),max_damage_error=float(np.max(np.abs(d))),irreversibility_violations=int(np.sum(np.diff(np.concatenate([pred['current_damage'][:,None],pred['damage']],axis=1),axis=1)<-1e-12)))


def execute(task):
    case,base,out,config=task;path=case['path'];t=time.perf_counter()
    try:
        profiles=None;controls=()
        if case['world_id']=='p2f-development-002' and case['protocol']['name']=='multi_long':
            p=config['bounded_profiles'];profiles=[('B2','m',p['m_grid']),('B5','c1',p['c1_grid'])];controls=p['initial_d0_controls']
        record,forecasts,meta,patterns,held=infer_prefix(path,out,max_nfev=config['study']['max_nfev'],profile_grids=profiles,initial_controls=controls)
        # Held-out truth is constructed AFTER observation-only inference has saved forecasts.
        truth=actual_fields(base,meta,held,[meta['cutoff'],*meta['future_cycles']]);np.savez_compressed(Path(out)/'evaluation_truth_after_forecast.npz',**truth,coordinates_mm=held)
        true=dict(damage=truth['damage'][:,1:],current_damage=truth['damage'][:,0],u=truth['u'][:,1:]);weights=truth['weights']
        scores={m:score_fields(f,true,weights) for m,f in forecasts.items()}
        control_scores={}
        for d0 in controls:
            file=Path(out)/'initial_controls'/str(d0)/'forecast_before_truth.npz'
            if file.exists():
                with np.load(file) as p:control_scores[str(d0)]=score_fields(dict(p),true,weights)
        label=json.loads((Path(base)/'synthetic_truth.json').read_text())
        cost_DIC=5*len(meta['cycles'])*2*config['study']['Q']+(5*2*config['study']['Q'] if meta['healthy'] else 0)
        exposure=[dict(**s,cycles=s['end']-s['start']) for s in meta['history']]
        result=dict(scope='DEVELOPMENT_ONLY',case_id=meta['case_id'],world_id=meta['world_id'],protocol=case['protocol'],family_FOR_EVALUATION_ONLY=label['family'],status='COMPLETED',scores=scores,initial_control_scores=control_scores,inference=record,
            signal_RMS_mm=float(np.sqrt(np.mean(truth['u'][:,0]**2))),noise_sigma_mm=meta['sigma_mm'],signal_SNR=float(np.sqrt(np.mean(truth['u'][:,0]**2))/meta['sigma_mm']),damage_growth_by_specimen=np.mean(truth['damage'][:,-1]-truth['damage'][:,0],axis=-1),max_true_damage=float(truth['damage'].max()),exposure=exposure,measurement_budget=dict(DIC_scalars=cost_DIC,independent_reactions=5 if meta['reaction'] else 0,total_cost_units=cost_DIC+10*int(meta['reaction']),independent_worlds=1,paired_specimens=5),future_truth_FEM_solves=truth['actual_FEM_solves'],wall_seconds=time.perf_counter()-t,truth_used_by_inference=False)
    except Exception as exc:
        result=dict(scope='DEVELOPMENT_ONLY',case_id=case['case_id'],world_id=case['world_id'],protocol=case['protocol'],status='FAILED_RETAINED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-t)
        Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'evaluation.json').write_text(json.dumps(clean(result),indent=2)+'\n')
    return clean(result)


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);config=tomllib.loads(Path(a.config).read_text());h=hashlib.sha256(Path(a.config).read_bytes()).hexdigest();sh=source_hash();specs=definitions(config);t=time.perf_counter()
    if config['study']['seeds']!=[240000,240001,240002] or len(config['study']['families'])!=3:raise ValueError('frozen small Development plan changed')
    frozen_rules={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('src/fatigue_dic_rl/correction/model_selection.py','src/fatigue_dic_rl/inference/designed_model_estimator.py')}
    plan=dict(scope='DEVELOPMENT_ONLY',config_sha256=h,source_sha256=sh,independent_worlds=3,paired_specimens_per_world=5,definitions=specs,seeds=config['study']['seeds'],families_GENERATION_ONLY=config['study']['families'],created_before_measurements=True,frozen_rules=frozen_rules,formal_independent_test_executed=False,stopping=config['stopping'],information_comparison='Budgets and fatigue exposures differ; report both; no unconditional equal-cost design claim')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');bases=[]
    for w,(seed,family) in enumerate(zip(config['study']['seeds'],config['study']['families'])):
        base=out/'worlds'/f'p2f-development-{w:03d}'
        generate_case(base,dict(case_id=base.name,seed=seed,split='phase2f-development',family=family,sigma_mm=config['study']['sigma_mm'],Q=config['study']['Q']),config_sha256=h);bases.append(str(base.resolve()))
    cases=[]
    for spec in specs:
        case=produce_case(bases[spec['world']],out/'observations'/f'p2f-dev-{spec["world"]:03d}-{spec["name"]}',spec,config,sh,h);cases.append(case)
    (out/'acquisition_manifest.json').write_text(json.dumps(dict(records=cases,acquisition_wall_seconds=time.perf_counter()-t),indent=2)+'\n')
    tasks=[(c,bases[c['protocol']['world']],str(out/'inference'/c['case_id']),config) for c in cases if c['protocol']['fit']];rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(execute,task):task[0]['case_id'] for task in tasks}
        for f in as_completed(futures):
            r=f.result();rows.append(r);print(r['case_id'],r['status'],round(r['wall_seconds'],1),flush=True)
            (out/'progress.json').write_text(json.dumps(clean(rows),indent=2)+'\n')
    rows.sort(key=lambda r:r['case_id']);(out/'summary.json').write_text(json.dumps(clean(dict(scope='DEVELOPMENT_ONLY',independent_worlds=3,paired_specimens_per_world=5,fit_cases=len(rows),information_only_cases=3,results=rows,wall_seconds=time.perf_counter()-t,source_sha256=sh,config_sha256=h,formal_independent_test_executed=False)),indent=2)+'\n')
    from fatigue_dic_rl.evaluation.information_pilot_diagnostics import run_diagnostics
    run_diagnostics(out,config)
    print('DEVELOPMENT study complete; no independent qualification test',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/phase2f/pilot.toml');p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a)
