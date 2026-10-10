from pathlib import Path
import tomllib

import pytest

from fatigue_dic_rl.evaluation.fatigue_verification import RUNNERS

pytestmark = pytest.mark.scientific


@pytest.mark.parametrize("case", list(RUNNERS))
def test_phase1b_scientific_experiment(case):
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root/"configs/phase1b"/f"{case}.toml").read_text())
    assert RUNNERS[case](config)["status"] == "PASS"
