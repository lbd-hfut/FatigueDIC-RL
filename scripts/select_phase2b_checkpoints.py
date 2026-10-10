"""Freeze a whole model cohort before calibration/test, never select a best seed."""
import argparse,json,hashlib
from pathlib import Path
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);out=root/'selected';out.mkdir(exist_ok=False)
    old=checkpoint_index(root/'training_initial',root/'training_sequential');cold=checkpoint_index('unused',root/'training_causal');warm=checkpoint_index('unused',root/'training_warm_causal');selected={}
    for s in [41001,41002,41003,41004,41005]:
        for variant,cohort in [('graph_gru',old),('physics_global',cold),('physics_shape',warm),('physics_shape_uncertainty',warm)]:
            key=f'{variant}_seed_{s}';p=Path(cohort[key]);selected[key]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (out/'checkpoint_selection.json').write_text(json.dumps({'overrides':selected,'selection_rule':'whole five-seed causal cohort, not best seed; global cold causal; shape/cov direct-GRU initialized causal; F seed41004 validation-only early-stop diagnosis override','information_used':'training/validation only; independent calibration/test not opened for uniform selection','fusion_caveat':'global and final shape differ in initialization; isolated fusion comparison exists in symmetric cold-causal and teacher-conditioned validation cohorts; no fusion-only benefit claimed'},indent=2)+'\n');print(len(selected),'frozen checkpoints')
