"""Exploratory six-parameter initial Gaussian, known fatigue A; no shape reset."""
import argparse,json,hashlib,tarfile
from pathlib import Path
import tomllib
from fatigue_dic_rl.training.training_loop import train_one
from fatigue_dic_rl.training.synthetic_dataset import source_hash
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2b/gaussian.toml');ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();c=tomllib.loads(Path(a.config).read_text());manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);c['_training_source_sha256']=source_hash();files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'))
    (out/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    with tarfile.open(out/'training_source.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p),recursive=False)
    rows=[]
    for seed in c['training']['seeds']:
        for variant in ['graph_initial','physics_shape']:
            model,r=train_one(c,manifest,out/f'{variant}_seed_{seed}',variant,seed);rows.append(r)
    (out/'training_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
