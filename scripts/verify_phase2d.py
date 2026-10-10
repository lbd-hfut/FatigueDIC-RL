"""Pre-register scenarios and generate independent physics-consistent trajectories."""
import argparse,json,hashlib,tomllib,time,datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.robust_correction_dataset import generate_case,source_hash


def frozen_plan(config):
    c=tomllib.loads(Path(config).read_text());cases=[]
    def add(split,seed,family,sigma,**opts):cases.append(dict(case_id=f'p2d-{len(cases):04d}',split=split,seed=seed,family=family,sigma_mm=sigma,**opts))
    for split,start,count in [('train',97000,c['study']['train_cases']),('validation',98000,c['study']['validation_cases']),('calibration',99000,c['study']['calibration_cases'])]:
        for i in range(count):add(split,start+i,('matched','parameter','S1')[i%3],5e-5)
    for level,sigma in enumerate(c['study']['noise_mm']):
        for i in range(c['study']['cases_per_noise']):add('test',100000+1000*level+i,'matched' if i<8 else 'parameter' if i<14 else 'S1',sigma)
    for f,family in enumerate(c['study']['structural_ood_families']):
        for i in range(c['study']['ood_cases_per_family']):add('ood',110000+100*f+i,family,5e-5,R=(-1.,0.,.3)[i%3] if family=='S3' else -1.)
    for kind in ['spatial','registration','loading','material','clustered','missing','Q64']:
        for i in range(4):
            options={};flags={};family='matched' if i%2==0 else 'S1'
            if kind=='spatial':flags['spatial']=True
            if kind=='registration':flags['registration']=True;options['shared_registration']=i>=2
            if kind=='loading':flags['load_bias']=1.07;options.update(load_unknown=i>=2,reaction=i>=2)
            if kind=='material':flags['E_bias']=1.1
            if kind=='clustered':flags['clustered']=True
            if kind=='missing':flags['missing']=.35
            if kind=='Q64':flags['Q']=64
            add('diagnostic',120000+1000*len(cases)+i,family,5e-5,diagnostic=kind,estimation_options=options,**flags)
    for seed in (150000,150001):
        for nx,ny in ((2,1),(4,2),(8,4)):add('diagnostic',seed,'S5',5e-5,nx=nx,ny=ny,diagnostic='paired_spatial_mesh')
    for i in range(2):add('diagnostic',151000+i,'S1',5e-5,width=3.,height=.7,diagnostic='geometry')
    return dict(schema='phase2d-frozen-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),source_sha256=source_hash(),cases=cases,selection_inputs='anonymous observation-only acquisition, N0/10 fit and N20 history validation',structural_ood_families=['S2','S3','S4','S5'],test_generated_before_selection=False)


def generate_task(task):
    path,spec,sha=task;return generate_case(path,spec,config_sha256=sha)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','generate'],required=True);ap.add_argument('--config',default='configs/phase2d/study.toml');ap.add_argument('--plan');ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if a.stage=='freeze':
        plan=frozen_plan(a.config);(out/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(len(plan['cases']),'cases pre-registered',flush=True)
    else:
        plan=json.loads(Path(a.plan).read_text());t=time.perf_counter();jobs=[(str(out/'cases'/s['case_id']),s,plan['config_sha256']) for s in plan['cases']]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(generate_task,jobs))
        (out/'manifest.json').write_text(json.dumps(dict(records=rows,plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),source_sha256=source_hash(),wall_seconds=time.perf_counter()-t),indent=2)+'\n');print(len(rows),'physics cases generated',flush=True)
