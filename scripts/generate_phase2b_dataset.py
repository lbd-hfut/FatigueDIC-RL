import argparse
from pathlib import Path
import tomllib
from fatigue_dic_rl.training.synthetic_dataset import generate_dataset
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2b/uniform.toml');p.add_argument('--output',required=True);a=p.parse_args();b=Path(a.config).read_bytes();r=generate_dataset(tomllib.loads(b.decode()),a.output,config_bytes=b);print('trajectories',len(r['records']),'wall_seconds',r['wall_seconds'])
