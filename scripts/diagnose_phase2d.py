import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.evaluation.robust_correction_scientific import temporal_law_families,snr_probe_task

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];t=time.perf_counter()
    tasks=[(r['path'],str(out/'snr'/r['case_id'])) for r in rows if r['split']=='test']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:info=list(pool.map(snr_probe_task,tasks))
    (out/'snr_records.json').write_text(json.dumps(info,indent=2)+'\n');reference=next(r for r in rows if r['split']=='validation');temporal=temporal_law_families(reference['path'],out/'temporal');(out/'execution.json').write_text(json.dumps(dict(physical_cases=len(info),temporal_laws=len(temporal),temporal_passed=all(r['passed'] for r in temporal),wall_seconds=time.perf_counter()-t),indent=2)+'\n');print('diagnostics complete',flush=True)
