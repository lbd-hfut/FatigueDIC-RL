"""Single-worker CPU pipeline benchmark after all training/evaluation workers exit."""
import argparse,json,time
from pathlib import Path
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index,initialize_worker,evaluation_record
from fatigue_dic_rl.evaluation.neural_belief_metrics import summarize_records
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);out=root/'benchmark_final';out.mkdir(exist_ok=False);cp=checkpoint_index(root/'training_initial',root/'selected');initialize_worker(cp);manifest=json.loads((root/'dataset/manifest.json').read_text());cal=json.loads((root/'calibration_final/calibration.json').read_text())['coefficients'];cases=[r for r in manifest['records'] if r['split']=='validation'][:3];rows=[];t=time.perf_counter()
    for case in cases:
        for key in cp:rows.append(evaluation_record(case,key,cal))
        for key in ['physics_prior','independent_frame_inverse','prefix_cold_map','sequential_warm_map']:rows.append(evaluation_record(case,key,{},classical=True))
    summary={k:summarize_records([r for r in rows if r['method']==k]) for k in sorted({r['method'] for r in rows})};(out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps({'methods':summary,'cases':[r['case_id'] for r in cases],'conditions':'CPU torch/numerical1 thread; sequential single process after other tasks exit; fresh estimator/history per case; no diagnostic output within timed inference; source and public observation IO included; forecast/uncertainty cost separate','wall_seconds':time.perf_counter()-t},indent=2)+'\n');print('benchmark complete',len(rows))
