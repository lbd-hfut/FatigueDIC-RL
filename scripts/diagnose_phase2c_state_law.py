"""Separate state error from law error AFTER frozen formal evaluation."""
import argparse,json,time
from pathlib import Path
from fatigue_dic_rl.evaluation.correction_scientific import state_law_separation,incorrect_state_negative_control
from fatigue_dic_rl.training.correction_dataset import source_hash

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
    results={scenario:state_law_separation(root/'dataset'/scenario/'test/case_84000',root/'fair_calibration',out/scenario) for scenario in ('matched','structural')}
    results['biased_state_negative_control']=incorrect_state_negative_control(root/'dataset/matched/test/case_84000',out/'biased_state_negative_control')
    (out/'execution.json').write_text(json.dumps({'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'oracle_diagnostic_only':True,'results':results},indent=2)+'\n')
    print('state/law diagnostic completed',flush=True)
