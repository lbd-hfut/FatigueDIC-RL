"""Train physics innovation from a validated direct-GRU initialization."""
import argparse,json,tarfile,hashlib
from pathlib import Path
import tomllib,torch
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.synthetic_dataset import source_hash
from fatigue_dic_rl.training.checkpoint import load_checkpoint
from fatigue_dic_rl.training.training_loop import train_one
from fatigue_dic_rl.networks.temporal_gru import GraphRecurrentEstimator,NetworkConfig
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index

def worker(task):
    c,manifest,root,seed,direct_path=task;root=Path(root);root.mkdir();direct,_=load_checkpoint(direct_path);model=GraphRecurrentEstimator(NetworkConfig(physics=True,uncertainty=False));model.load_state_dict(direct.state_dict())
    with torch.no_grad():
        model.dic.input[0].weight[:,4:10]=0.;model.fe.input[0].weight[:,-2:]=0.;model.encode[0].weight[:,128+6:128+18]=0.;model.encode[0].weight[:,-2:]=0.
    mean,r=train_one(c,manifest,root/f'physics_shape_seed_{seed}','physics_shape',seed,initial_model=model)
    _,u=train_one(c,manifest,root/f'physics_shape_uncertainty_seed_{seed}','physics_shape_uncertainty',seed,initial_model=mean);r['initialization']='direct GRU, physics-only feature weights initially zero; trainable; selected by validation including initial checkpoint';r['initialization_checkpoint_sha256']=hashlib.sha256(Path(direct_path).read_bytes()).hexdigest();(root/'warm_start_audit.json').write_text(json.dumps(r,indent=2)+'\n');return [r,u]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--direct',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();c=tomllib.loads(Path('configs/phase2b/uniform_warm_causal.toml').read_text());c['_training_source_sha256']=source_hash();m=json.loads((Path(a.dataset)/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cp=checkpoint_index('unused',a.direct)
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'));(out/'source_manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    with tarfile.open(out/'training_source.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f),recursive=False)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=[r for group in pool.map(worker,[(c,m,str(out/f'seed_{s}'),s,cp[f'graph_gru_seed_{s}']) for s in c['training']['seeds']]) for r in group]
    (out/'training_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
