"""M2 validation gate, before any final test labels are read."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fatigue_dic_rl.training.synthetic_dataset import load_graph_cache,load_labels
from fatigue_dic_rl.training.checkpoint import load_checkpoint
from fatigue_dic_rl.networks.temporal_gru import pack_graph_frames
from fatigue_dic_rl.evaluation.neural_belief_metrics import parameter_statistics
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--training',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(1)
    records=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];validation=[r for r in records if r['split']=='validation'];x=[load_graph_cache(Path(r['path'])/'training_condition_features.npz')[0][0] for r in validation];target=np.array([load_labels(r['path'])['physical_parameters'] for r in validation]);rows=[]
    for checkpoint in sorted(Path(a.training).glob('graph_initial_seed_*/checkpoint.pt')):
        model,meta=load_checkpoint(checkpoint)
        with torch.no_grad():z=model.step(pack_graph_frames(x,physics=False))['normalized_mean'].numpy()
        physical=np.column_stack([.38+z[:,0]*.38,.00205+z[:,1]*.00195]);errors=physical-target;stats=parameter_statistics(errors);rows.append({'seed':meta['seed'],'validation':stats,'fatigue_A_interpretation':'single N0 has zero A sensitivity; estimate is training-prior dependent','d0_gate_passed':stats['rmse'][0]<.03,'physical_prediction':physical.tolist()})
    p=Path(a.output);p.write_text(json.dumps({'partition':'validation','test_access':False,'rows':rows,'status':'PASSED' if rows and all(r['d0_gate_passed'] for r in rows) else 'FAILED'},indent=2)+'\n');print([(r['seed'],r['validation']['rmse']) for r in rows])
