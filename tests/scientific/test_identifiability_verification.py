from pathlib import Path
import tomllib
import pytest
from fatigue_dic_rl.evaluation.identifiability_verification import RUNNERS


@pytest.mark.scientific
@pytest.mark.parametrize("name", list(RUNNERS))
def test_phase1c_scientific_matrix(name):
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root/"configs"/"phase1c"/(name+".toml")).read_text())
    result = RUNNERS[name](config)
    assert result["status"] == "PASSED"
    assert result["checks"]
