import json
import os
from pathlib import Path
import subprocess
import sys


def test_real_fe_to_dic_cli_and_preserved_outputs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path/"uniaxial"
    command = [sys.executable, "scripts/verify_phase1a.py", "--config", "configs/phase1/uniaxial_tension.toml", "--output", str(output)]
    env = dict(os.environ, PYTHONPATH=str(root/"src"), MPLCONFIGDIR=str(tmp_path/"matplotlib"))
    subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["fe_to_dic"]["image_correlation_performed"] is False
    assert (output/"uniaxial_fields.npz").is_file()
    assert (output/"uniaxial_stiffness.npz").is_file()
    assert (output/"solver.log").is_file()
    before = (output/"summary.json").read_bytes()
    second = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert (output/"summary.json").read_bytes() == before
