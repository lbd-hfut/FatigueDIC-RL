import json
import os
from pathlib import Path
import subprocess
import sys

from fatigue_dic_rl.smoke import run_smoke


def test_analytic_module_smoke():
    first = run_smoke()
    assert first == run_smoke()
    assert first["fe_solve_performed"] is False
    assert first["valid_observation_points"] == 6
    assert first["outside_points"] == 2
    assert first["affine_max_abs_error_mm"] < 1e-14
    assert first["damage_bounds_passed"] and first["irreversibility_passed"]


def test_physics_only_cli_imports_no_rl_or_torch():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root/"src"))
    proc = subprocess.run([sys.executable, "scripts/phase0_smoke.py", "--config", "configs/phase0.toml"],
                          cwd=root, env=env, capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout)["total_cycles"] == 1111
    script = "from fatigue_dic_rl.smoke import run_smoke; run_smoke(); import sys; assert not any(x in sys.modules for x in ['torch', 'fatigue_dic_rl.rl', 'scipy'])"
    subprocess.run([sys.executable, "-c", script], cwd=root, env=env, check=True, capture_output=True, text=True)
