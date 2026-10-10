"""Run ALL 37 historic scientific cases into fresh Phase 2B evidence folders."""
import argparse,json,subprocess,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def run_case(task):
    phase,config,script,out=task;out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);t=time.perf_counter();command=['python','-W','error',script,'--config',config,'--output',str(out)]
    with (out.parent/(out.name+'_command.txt')).open('x') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    return {'phase':phase,'config':config,'command':command,'returncode':r.returncode,'wall_seconds':time.perf_counter()-t}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);jobs=[]
    for phase,folder,script in [('1a','configs/phase1','scripts/verify_phase1a.py'),('1b','configs/phase1b','scripts/verify_phase1b.py'),('1c','configs/phase1c','scripts/verify_phase1c.py'),('2a','configs/phase2a','scripts/verify_phase2a.py')]:
        for p in sorted(Path(folder).glob('*.toml')):jobs.append((phase,str(p),script,str(out/phase/p.stem)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(run_case,jobs))
    t=time.perf_counter();command=['python','scripts/phase0_smoke.py','--config','configs/phase0.toml'];r=subprocess.run(command,capture_output=True,text=True);(out/'phase0_smoke.txt').write_text(r.stdout+r.stderr);rows.append({'phase':'0','command':command,'returncode':r.returncode,'wall_seconds':time.perf_counter()-t});(out/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n');print('historic cases',len(rows),'failed',sum(r['returncode']!=0 for r in rows));raise SystemExit(int(any(r['returncode']!=0 for r in rows)))
