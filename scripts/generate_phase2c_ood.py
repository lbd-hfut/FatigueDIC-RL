import argparse,json,tomllib,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.correction_dataset import generate_correction_case,source_hash


def task(args):folder,kwargs=args;return generate_correction_case(folder,**kwargs)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2c/study.toml');p.add_argument('--ood-config',default='configs/phase2c/ood_contract.toml');p.add_argument('--output',required=True);a=p.parse_args();c=tomllib.loads(Path(a.config).read_text());ot=Path(a.ood_config).read_text();o=tomllib.loads(ot);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'frozen_ood.toml').write_text(ot);t=time.perf_counter();jobs=[];first=c['ood']['first_seed'];n=c['ood']['cases_per_scenario']
 for i,s in enumerate(c['ood']['scenarios']):
  for j in range(n):
   seed=first+i*n+j;kw=dict(seed=seed,split='ood',scenario='spatial_discrepancy' if s=='spatial_discrepancy' else 'structural',case_label=s,sigma_mm=c['dataset']['sigma_mm'],config_sha256=hashlib.sha256(ot.encode()).hexdigest());kw.update(o[s]);jobs.append((str(out/s/f'case_{seed}'),kw))
   if s=='cross_mesh':
    reference=dict(kw);reference.update(nx=2,ny=1,case_label='paired_mesh_reference');jobs.append((str(out/'paired_mesh_reference'/f'case_{seed}'),reference))
 with ProcessPoolExecutor(max_workers=c['dataset']['workers']) as pool:rows=list(pool.map(task,jobs))
 (out/'manifest.json').write_text(json.dumps({'records':rows,'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'paired_mesh_reference_not_independent':True},indent=2)+'\n');print('generated OOD',len(rows),flush=True)
