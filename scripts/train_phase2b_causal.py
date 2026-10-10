"""Validation-diagnosed causal physics conditioning; same five frozen seeds."""
import argparse,json,tarfile,hashlib
from pathlib import Path
import tomllib
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.synthetic_dataset import source_hash
from fatigue_dic_rl.training.training_loop import train_one

def worker(task):
    c,manifest,root,seed=task;root=Path(root);root.mkdir();rows=[];mean=None
    for variant in ['physics_global','physics_shape','physics_shape_uncertainty']:
        model,row=train_one(c,manifest,root/f'{variant}_seed_{seed}',variant,seed,initial_model=mean if variant.endswith('uncertainty') else None);rows.append(row)
        if variant=='physics_shape':mean=model
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();c=tomllib.loads(Path('configs/phase2b/uniform_causal.toml').read_text());c['_training_source_sha256']=source_hash();manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'));(out/'source_manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    with tarfile.open(out/'training_source.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f),recursive=False)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=[r for group in pool.map(worker,[(c,manifest,str(out/f'seed_{s}'),s) for s in c['training']['seeds']]) for r in group]
    (out/'training_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
