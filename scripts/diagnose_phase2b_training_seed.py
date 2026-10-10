"""Validation-only delayed stopping diagnostic; final test remains unopened."""
import argparse,json
from pathlib import Path
import tomllib
from fatigue_dic_rl.training.training_loop import train_one
from fatigue_dic_rl.training.synthetic_dataset import source_hash
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();c=tomllib.loads(Path('configs/phase2b/uniform.toml').read_text());c['_training_source_sha256']=source_hash();m=json.loads((Path(a.dataset)/'manifest.json').read_text());train_one(c,m,Path(a.output)/'graph_gru_seed_41004','graph_gru',41004)
