import argparse,json
from pathlib import Path
import tomllib
from fatigue_dic_rl.training.training_loop import train_one
from fatigue_dic_rl.training.synthetic_dataset import source_hash
from fatigue_dic_rl.training.checkpoint import load_checkpoint
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2b/gaussian.toml');ap.add_argument('--dataset',required=True);ap.add_argument('--mean-training',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();c=tomllib.loads(Path(a.config).read_text());c['_training_source_sha256']=source_hash();manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in c['training']['seeds']:
        mean,meta=load_checkpoint(Path(a.mean_training)/f'physics_shape_seed_{seed}/checkpoint.pt');model,r=train_one(c,manifest,out/f'physics_shape_uncertainty_seed_{seed}','physics_shape_uncertainty',seed,initial_model=mean);rows.append(r)
    (out/'training_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
