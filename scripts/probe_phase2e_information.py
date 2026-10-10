import argparse,json
from pathlib import Path
from fatigue_dic_rl.evaluation.additional_information_verification import information_probes,temporal_probes

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);result=dict(information=information_probes(a.case,out/'static'),temporal=temporal_probes(a.case,out/'temporal'));(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('static and temporal probes complete')
