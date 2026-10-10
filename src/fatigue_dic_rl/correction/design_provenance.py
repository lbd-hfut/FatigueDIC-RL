"""Runtime file fingerprint used to enforce a pre-test protocol freeze."""
from pathlib import Path
from hashlib import sha256
import json


def runtime_manifest():
    files=sorted(p for root in ('src','scripts','configs/phase2e') for p in Path(root).rglob('*') if p.is_file() and p.suffix in ('.py','.toml') and '__pycache__' not in p.parts)
    return {str(p):sha256(p.read_bytes()).hexdigest() for p in files}


def verify_freeze(record,config_hash):
    if record['config_sha256']!=config_hash:raise ValueError('frozen config has changed')
    if record['runtime_manifest']!=runtime_manifest():raise ValueError('runtime source has changed after the independent-test freeze')
    return True
