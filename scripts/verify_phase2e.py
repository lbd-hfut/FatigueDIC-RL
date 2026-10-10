"""New physical worlds; development precedes separately frozen independent tests."""
import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.robust_correction_dataset import generate_case,source_hash
from fatigue_dic_rl.correction.design_provenance import verify_freeze


def cases(split):
    rows=[]
    if split=='development':defs=[('matched',5e-5),('matched',2.5e-4),('parameter',5e-5),('parameter',2.5e-4),('S1',5e-5),('S1',2.5e-4),('S1',5e-5),('matched',5e-5)];start=210000
    elif split=='validation':defs=[('matched',7.5e-4),('S1',7.5e-4),('S1',5e-5),('matched',5e-5)];start=220000
    elif split=='test':defs=[(f,s) for f in ('matched','parameter','S1') for s in ([5e-5]*4+[2.5e-4]*2+[7.5e-4]*2)]+[(f,5e-5) for f in ('S2','S3','S4','S5') for _ in range(2)]+[(f,5e-5) for _ in range(4) for f in ('matched','S1')];start=230000
    else:raise ValueError(split)
    for i,(f,s) in enumerate(defs):
        opts={};extra={};condition='independent'
        if split=='development' and i==6 or split=='test' and i in (32,33):extra['registration']=True;opts['shared_registration']=True;condition='registration'
        if split=='development' and i==7 or split=='test' and i in (34,35):extra['load_bias']=1.07;opts['load_unknown']=True;condition='loading'
        if split=='validation' and i>=2 or split=='test' and i in (38,39):extra['spatial']=True;condition='spatial'
        if split=='test' and i in (36,37):extra['E_bias']=1.1;opts['material_unknown']=True;condition='material'
        rows.append(dict(case_id=f'p2e-{split}-{i:03d}',seed=start+i,split=split,family=f,sigma_mm=s,condition=condition,estimation_options=opts,future_R=0. if (start+i)%2 else -1.,**extra))
    return rows


def generate(task):
    path,spec,h=task
    # The old producer normally generates held-out trajectories for split=test.
    # Here experiments have not yet been chosen: generate the common prefix only.
    record=generate_case(path,dict(spec,split='phase2e-common-prefix'),config_sha256=h)
    record.update(spec)
    public=Path(path)/'public.json';meta=json.loads(public.read_text());meta['split']=spec['split'];public.write_text(json.dumps(meta,indent=2)+'\n')
    (Path(path)/'public_conditions.json').write_text(json.dumps(dict(estimation_options=spec['estimation_options'],future_R=spec['future_R']),indent=2)+'\n')
    record['public_sha256']=hashlib.sha256(public.read_bytes()).hexdigest()
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['development','validation','test'],required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--config',default='configs/phase2e/study.toml');ap.add_argument('--freeze-record');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);specs=cases(a.split);h=hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    if a.split=='test':
        if not a.freeze_record:raise ValueError('independent test generation requires a frozen protocol record')
        record=json.loads(Path(a.freeze_record).read_text());verify_freeze(record,h)
        if record['test_cases']!=specs:raise ValueError('frozen independent world definitions have changed')
    plan=dict(config_sha256=h,source_sha256=source_hash(),cases=specs,created_before_generation=True);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(generate,[(str(out/'cases'/s['case_id']),s,h) for s in specs]))
    (out/'manifest.json').write_text(json.dumps(dict(records=rows,config_sha256=h,wall_seconds=time.perf_counter()-t),indent=2)+'\n');print(a.split,len(rows),'independent worlds')
