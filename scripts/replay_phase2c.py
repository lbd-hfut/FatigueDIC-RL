"""Five-seed retraining and physics dataset replay; compare, never select models."""
import argparse,json,time
from pathlib import Path
import numpy as np
from fatigue_dic_rl.training.correction_dataset import generate_correction_case,source_hash
from fatigue_dic_rl.training.correction_supervised import train_residual_coefficients

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
    rows=[]
    for seed in range(85000,85005):
        old=json.loads((root/f'training/structural/K4/seed_{seed}/metadata.json').read_text())
        new=train_residual_coefficients(old['train_cases'],old['validation_cases'],out/f'seed_{seed}',seed=seed,dimension=4)
        np.testing.assert_array_equal(old['coefficients'],new['coefficients'])
        assert old['best_epoch']==new['best_epoch'] and old['validation_mse']==new['validation_mse']
        rows.append({'seed':seed,'maximum_coefficient_difference':0.,'best_epoch':new['best_epoch'],'validation_mse':new['validation_mse']})
    regenerated=generate_correction_case(out/'regenerated_train',seed=81000,split='train',scenario='structural')
    original=root/'dataset/structural/train/case_81000'
    with np.load(original/'synthetic_trajectory.npz',allow_pickle=False) as old,np.load(Path(regenerated['path'])/'synthetic_trajectory.npz',allow_pickle=False) as fresh:
        assert np.max(fresh['cycles'])==20
        for key in old.files:np.testing.assert_array_equal(old[key],fresh[key])
    for load in range(5):
        for frame in range(3):
            for p in (original/'observations'/str(load)/str(frame)).glob('*.npz'):
                other=Path(regenerated['path'])/'observations'/str(load)/str(frame)/p.name
                with np.load(p,allow_pickle=False) as old,np.load(other,allow_pickle=False) as fresh:
                    for key in old.files:np.testing.assert_array_equal(old[key],fresh[key])
    (out/'execution.json').write_text(json.dumps({'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'training_replay':rows,
        'dataset_prefix_replay_max_difference':0.,'future_training_fields_generated':False,'passed':True},indent=2)+'\n')
    print('five-seed training and dataset deterministic replay passed',flush=True)
