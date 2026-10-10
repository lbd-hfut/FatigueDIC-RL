"""Six-world Development cost probe; deliberately refuses formal test expansion."""
import argparse,hashlib,tomllib
from pathlib import Path
from fatigue_dic_rl.evaluation.state_identification_readiness import run_readiness

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/validation_readiness/development.toml')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();path=Path(args.config)
    run_readiness(args.output,tomllib.loads(path.read_text()),hashlib.sha256(path.read_bytes()).hexdigest())
