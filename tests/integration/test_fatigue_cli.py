import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_scientific_cli_real_trajectory_and_preserved_outputs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path/"synthetic"
    command = [sys.executable, "scripts/verify_phase1b.py", "--config", "configs/phase1b/synthetic_trajectory.toml",
               "--output", str(output)]
    env = dict(os.environ, PYTHONPATH=str(root/"src"), MPLCONFIGDIR=str(tmp_path/"matplotlib"))
    proc = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout)["status"] == "PASS"
    summary = json.loads((output/"summary.json").read_text())
    assert summary["bitwise_reproducible_numerical_arrays"]
    assert "worktree-source-sha256:" in summary["metadata"]["code_version"]
    with np.load(output/"trajectory/truth/trajectory.npz") as data:
        assert data["N"][-1] == 40 and data["damage"].shape[1] == 16
    with np.load(output/"trajectory/observations/displacement.npz") as data:
        assert "damage" not in data.files
        assert not data["valid_mask"][:, -1].any()
    assert (output/"damage_fields.png").is_file() and (output/"solver.log").is_file()
    before = (output/"summary.json").read_bytes()
    again = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert again.returncode != 0 and (output/"summary.json").read_bytes() == before
    script = "import fatigue_dic_rl.fatigue; import sys; assert not any(x in sys.modules for x in ['torch', 'fatigue_dic_rl.rl', 'fatigue_dic_rl.networks'])"
    subprocess.run([sys.executable, "-c", script], cwd=root, env=env, check=True, capture_output=True)


def test_failed_cli_preserves_failure_evidence(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = tmp_path/"bad.toml"
    config.write_text((root/"configs/phase1b/synthetic_trajectory.toml").read_text().replace(
        "young_modulus_n_per_mm2 = 2000.0", "young_modulus_n_per_mm2 = -1.0"))
    out = tmp_path/"failed"
    env = dict(os.environ, PYTHONPATH=str(root/"src"))
    proc = subprocess.run([sys.executable, "scripts/verify_phase1b.py", "--config", str(config), "--output", str(out)],
        cwd=root, env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    result = json.loads((out/"summary.json").read_text())
    assert result["status"] == "FAILED" and result["message"] and result["traceback"]
    assert not (out/"trajectory").exists()
