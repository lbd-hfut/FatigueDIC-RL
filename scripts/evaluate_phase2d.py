"""Run frozen independent splits; outputs never replace historic results."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.evaluation.robust_correction_verification import evaluate_task
from fatigue_dic_rl.training.robust_correction_dataset import source_hash

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--splits',nargs='+',default=['test','ood','diagnostic']);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());cases=[r for r in manifest['records'] if r['split'] in a.splits];t=time.perf_counter();jobs=[(r['path'],str(out/'raw'/r['case_id']),r.get('estimation_options',{})) for r in cases]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(evaluate_task,jobs):rows.append(row);print(row.get('case_id',row.get('path')),row['status'],flush=True)
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');summary=dict(cases=len(rows),failed=sum(r['status']=='FAILED' for r in rows),wall_seconds=time.perf_counter()-t,source_sha256=source_hash());(out/'execution.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
