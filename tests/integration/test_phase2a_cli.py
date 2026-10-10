import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def test_phase2a_cli_provenance_histories_jacobian_and_no_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root/"src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MPLCONFIGDIR=str(tmp_path/"mpl"))
    out = tmp_path/"uniform"
    command = [sys.executable, "-W", "error", str(root/"scripts"/"verify_phase2a.py"),
               "--config", str(root/"configs"/"phase2a"/"uniform.toml"), "--output", str(out)]
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr+result.stdout
    summary = json.loads((out/"summary.json").read_text())
    assert summary["status"] == "PASSED"
    assert len(summary["metadata"]["source_sha256"]) == 64
    assert summary["metadata"]["noise_contract"].startswith("frame noise independent")
    prefix = json.loads((out/"uniform_prefix_2.json").read_text())
    assert prefix["starts"][0]["history"]
    assert len(prefix["provenance"]["conditioning_ids"]) == 3
    assert prefix["provenance"]["fixed_prior_terms"] == 0
    with np.load(out/"uniform_prefix_2.npz", allow_pickle=False) as data:
        assert data["jacobian_whitened"].shape == (96, 1)
        assert data["data_fisher"].shape == data["posterior_fisher"].shape == (1, 1)
    before = (out/"summary.json").read_bytes()
    again = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    assert again.returncode != 0 and before == (out/"summary.json").read_bytes()


def test_all_configs_have_runners_and_large_statistics_stay_in_cli():
    import tomllib
    from fatigue_dic_rl.evaluation.sequential_verification import RUNNERS
    root = Path(__file__).resolve().parents[2]
    configs = {p.stem: tomllib.loads(p.read_text()) for p in (root/"configs"/"phase2a").glob("*.toml")}
    assert set(configs) == set(RUNNERS)
    for name in ("coverage_uniform", "coverage_fatigue"):
        assert configs[name]["calibration"]["seeds"] >= 100
        assert set(configs[name]["calibration"]["scenarios"]) >= {"independent", "spatial", "registration", "loading"}
