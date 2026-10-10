"""Re-evaluate frozen Phase2B assets; never retrain or rewrite historic outputs."""
import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from fatigue_dic_rl.evaluation.neural_belief_verification import (
    checkpoint_index, evaluate_partition, initialize_worker, test_case,
)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--historic', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--suites', nargs='+', choices=['uniform', 'generalization', 'fixed_noise', 'gaussian', 'architecture'], default=['uniform', 'generalization', 'fixed_noise', 'gaussian', 'architecture'])
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    old, out = Path(a.historic), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    t = time.perf_counter()
    cp = checkpoint_index(old/'training_initial', old/'selected')
    cal = json.loads((old/'calibration_final/calibration.json').read_text())['coefficients']
    executed = {}
    if 'uniform' in a.suites:
        result = evaluate_partition(json.loads((old/'dataset/manifest.json').read_text()), cp, cal, out/'uniform', baseline_count=16, workers=a.workers)
        executed['uniform'] = result['case_count']
    if 'generalization' in a.suites or 'fixed_noise' in a.suites:
        for label in [s for s in a.suites if s in ('generalization', 'fixed_noise')]:
            if label == 'generalization':
                cases = json.loads((old/'generalization_final/dataset_manifest.json').read_text())
                keys = [k for k in cp if not k.startswith(('graph_initial', 'graph_observation'))]
                baseline = lambda c: (c['seed']-51000) % 12 < 3
            else:
                cases = []
                for p in sorted((old/'fixed_noise_final/dataset').iterdir()):
                    m = json.loads((p/'public.json').read_text())
                    cases.append(dict(case_id=m['case_id'], seed=m['seed'], scenario=m['scenario'], path=str(p)))
                keys = [k for k in cp if k.startswith('physics_shape_uncertainty')]
                baseline = lambda c: c['seed'] < 53004
            dest = out/label
            dest.mkdir()
            jobs = [(c, keys, cal, baseline(c), str(dest/'raw')) for c in cases]
            with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize_worker, initargs=({k:cp[k] for k in keys},)) as pool:
                rows = [r for group in pool.map(test_case, jobs) for r in group]
            (dest/'records.json').write_text(json.dumps(rows, indent=2)+'\n')
            executed[label] = len(rows)
    if 'gaussian' in a.suites:
        gcp = {}
        for name in ('graph_initial', 'physics_shape', 'physics_shape_uncertainty'):
            for seed in range(71001, 71006):
                key = f'{name}_seed_{seed}'
                matches = list((old/'gaussian_training').rglob(key+'/checkpoint.pt'))
                if len(matches) != 1:
                    raise RuntimeError((key, len(matches)))
                gcp[key] = str(matches[0])
        result = evaluate_partition(json.loads((old/'gaussian_dataset/manifest.json').read_text()), gcp,
            json.loads((old/'gaussian_calibration_final/calibration.json').read_text())['coefficients'],
            out/'gaussian', baseline_count=4, workers=a.workers)
        executed['gaussian'] = result['case_count']
    if 'architecture' in a.suites:
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location('phase2b_checks', Path(__file__).with_name('verify_phase2b.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [module.check_architecture(path, old/'dataset/validation/case_0000') for key, path in cp.items() if key.startswith('physics_shape_uncertainty')]
        (out/'architecture.json').write_text(json.dumps(rows, indent=2)+'\n')
        executed['architecture'] = len(rows)
    (out/'execution.json').write_text(json.dumps({'executed':executed, 'wall_seconds':time.perf_counter()-t,
        'all_checkpoints_frozen':True, 'calibration_retrained':False}, indent=2)+'\n')
    print('frozen Phase2B regression complete', executed, flush=True)
