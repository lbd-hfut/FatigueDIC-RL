import os
from pathlib import Path
import subprocess
import sys
import json
import numpy as np


def test_cli_provenance_observation_isolation_and_no_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root/"src"), MPLCONFIGDIR=str(tmp_path/"mpl"), OPENBLAS_NUM_THREADS="1")
    out = tmp_path/"experiment"
    command = [sys.executable, str(root/"scripts"/"verify_phase1c.py"), "--config",
               str(root/"configs"/"phase1c"/"observation_consistency.toml"), "--output", str(out)]
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out/"summary.json").read_text())
    assert summary["status"] == "PASSED"
    assert len(summary["metadata"]["source_sha256"]) == 64
    assert summary["metadata"]["units"]["covariance"] == "mm^2"
    with np.load(out/"noiseless_observations"/"observation.npz") as a:
        assert set(a.files) == {"coordinates_mm", "displacement_mm", "valid_mask", "roi_mask", "confidence", "covariance_mm2"}
    original = (out/"summary.json").read_bytes()
    again = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    assert again.returncode != 0 and original == (out/"summary.json").read_bytes()
