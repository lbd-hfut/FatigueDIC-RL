import argparse,json,hashlib,tarfile
from pathlib import Path
import tomllib
from fatigue_dic_rl.training.training_loop import train_one
from fatigue_dic_rl.training.checkpoint import load_checkpoint
from fatigue_dic_rl.training.synthetic_dataset import source_hash

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2b/uniform.toml');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--stage',choices=['initial','sequential'],required=True);p.add_argument('--device',default='cpu');p.add_argument('--initial-directory');a=p.parse_args();c=tomllib.loads(Path(a.config).read_text());manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[]
    c['_training_source_sha256']=source_hash()
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'))
    (out/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    with tarfile.open(out/'training_source.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p),recursive=False)
    variants=['graph_initial','graph_observation_only'] if a.stage=='initial' else ['graph_gru','physics_global','physics_shape','physics_shape_uncertainty']
    for seed in c['training']['seeds']:
        previous=None
        for v in variants:
            model,m=train_one(c,manifest,out/f'{v}_seed_{seed}',v,seed,device=a.device,initial_model=previous if v=='physics_shape_uncertainty' else None);results.append(m)
            if v=='physics_shape':previous=model
    (out/'training_matrix.json').write_text(json.dumps(results,indent=2)+'\n')
