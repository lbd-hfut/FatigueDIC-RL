"""Run the frozen non-RL design comparison; every failure is retained."""
import argparse,json,tomllib,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from fatigue_dic_rl.evaluation.observability_design_verification import evaluate_world
from fatigue_dic_rl.correction.design_provenance import verify_freeze,runtime_manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--config',default='configs/phase2e/study.toml');ap.add_argument('--workers',type=int,default=3);ap.add_argument('--case-id');ap.add_argument('--freeze-record');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);config=tomllib.loads(Path(a.config).read_text());records=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];records=[r for r in records if a.case_id is None or r['case_id']==a.case_id];t=time.perf_counter();rows=[]
    if any(r['split']=='test' for r in records):
        if not a.freeze_record:raise ValueError('independent evaluation requires the frozen protocol record')
        verify_freeze(json.loads(Path(a.freeze_record).read_text()),hashlib.sha256(Path(a.config).read_bytes()).hexdigest())
    (out/'execution_plan.json').write_text(json.dumps(dict(case_ids=[r['case_id'] for r in records],config=config,runtime_manifest=runtime_manifest(),independent_unit='physical world; five paired counterfactual experimental arms'),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(evaluate_world,(r['path'],str(out/'cases'/r['case_id']),config)):r['case_id'] for r in records}
        for f in as_completed(jobs):
            r=f.result();rows.append(r);print(r['case_id'],r['status'],round(r['wall_seconds'],2),flush=True)
            (out/'completed_so_far.json').write_text(json.dumps(rows,indent=2)+'\n')
    rows.sort(key=lambda r:r['case_id']);(out/'results.json').write_text(json.dumps(dict(worlds=rows,wall_seconds=time.perf_counter()-t),indent=2)+'\n');print('worlds',len(rows),'failed',sum(r['status']=='FAILED' for r in rows),flush=True)
