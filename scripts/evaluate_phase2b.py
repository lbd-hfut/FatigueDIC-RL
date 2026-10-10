import argparse,json
from pathlib import Path
import tomllib
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index,fit_calibration,evaluate_partition
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2b/uniform.toml');p.add_argument('--dataset',required=True);p.add_argument('--initial',required=True);p.add_argument('--sequential',required=True);p.add_argument('--output',required=True);p.add_argument('--stage',choices=['validation','calibration','test'],required=True);p.add_argument('--calibration');a=p.parse_args();c=tomllib.loads(Path(a.config).read_text());manifest=json.loads((Path(a.dataset)/'manifest.json').read_text());checkpoints=checkpoint_index(a.initial,a.sequential)
    if a.stage=='calibration':result=fit_calibration(manifest,checkpoints,a.output,workers=c['dataset']['workers'])
    else:
        coeff=json.loads(Path(a.calibration).read_text())['coefficients'] if a.calibration else {}
        result=evaluate_partition(manifest,checkpoints,coeff,a.output,partition=a.stage,baseline_count=c['evaluation']['baseline_test_cases'] if a.stage=='test' else 0,workers=c['dataset']['workers'])
    print('completed',a.stage,'wall',result['wall_seconds'])
