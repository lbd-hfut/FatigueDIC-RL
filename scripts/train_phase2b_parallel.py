"""Independent CPU seed workers, each with identical frozen training settings."""
import argparse,json,hashlib,tarfile
from pathlib import Path
import tomllib
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.synthetic_dataset import source_hash
from fatigue_dic_rl.training.training_loop import train_one


def seed_worker(task):
    c,manifest,out,seed=task;out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[];previous=None
    for variant in ['graph_gru','physics_global','physics_shape','physics_shape_uncertainty']:
        model,r=train_one(c,manifest,out/f'{variant}_seed_{seed}',variant,seed,initial_model=previous if variant=='physics_shape_uncertainty' else None);rows.append(r)
        if variant=='physics_shape':previous=model
    (out/'training_matrix.json').write_text(json.dumps(rows,indent=2)+'\n');return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2b/uniform.toml');ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',default='41001,41002,41003,41004,41005');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();c=tomllib.loads(Path(a.config).read_text());c['_training_source_sha256']=source_hash();manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seeds=[int(s) for s in a.seeds.split(',')]
    if any(s not in c['training']['seeds'] for s in seeds):raise ValueError('seeds must belong to predeclared config')
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'));(out/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    with tarfile.open(out/'training_source.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p),recursive=False)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=[r for group in pool.map(seed_worker,[(c,manifest,str(out/f'seed_{s}'),s) for s in seeds]) for r in group]
    (out/'training_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
