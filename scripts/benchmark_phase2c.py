"""Serial end-to-end cost on the frozen first structural test case."""
import argparse,json,time
from pathlib import Path
from fatigue_dic_rl.evaluation.correction_verification import evaluate_correction_method

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];t=time.perf_counter()
    case=root/'dataset/structural/test/case_84000'
    for method in ('B2_refit_Am','B5_multistep_K2','B4_supervised_K4_seed_85000'):
        options={'checkpoint':root/'training/structural/K4/seed_85000/checkpoint.pt'} if method.startswith('B4') else {}
        rows.append(evaluate_correction_method(case,method,out/method,**options))
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'execution.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-t,'case_selection':'first frozen structural test case; no accuracy tuning',
        'serial_with_single_BLAS_thread':True,'background_contention_possible':True,
        'scope':'prefix fitting + data Fisher + all five specimen forecasts through N220 + physical field diagnostics; scoring IO excluded from online_seconds',
        'coefficients_inference':'fixed K-vector and matrix basis; no amortized neural state estimator'},indent=2)+'\n')
    print('serial cost benchmark complete',flush=True)
