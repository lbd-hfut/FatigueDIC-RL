import numpy as np
import pytest
from fatigue_dic_rl.evaluation.sequential_verification import RUNNERS, _calibration_trial


@pytest.mark.scientific
@pytest.mark.parametrize("case", ["uniform", "rank_deficiency", "likelihood", "time_holdout",
                                  "baseline_comparison", "model_discrepancy", "gaussian_static", "gaussian_evolution"])
def test_phase2a_fast_scientific_matrix(case):
    config = {"case": case, "geometry": {"nx": 6, "ny": 3, "seed": 19},
              "optimization": {"max_nfev": 120}, "observation": {"count": 16}}
    result = RUNNERS[case](config)
    assert result["status"] == "PASSED" and result["checks"]


@pytest.mark.scientific
@pytest.mark.parametrize("case", ["coverage_uniform", "coverage_fatigue"])
def test_fast_fem_coverage_protocol_does_not_claim_100_seed_qualification(case):
    config = {"case": case, "geometry": {"nx": 2, "ny": 1, "seed": 19},
        "optimization": {"max_nfev": 120}, "observation": {"count": 16},
        "calibration": {"seeds": 2, "first_seed": 1000, "workers": 1, "sigma_mm": 5e-5, "scenarios": ["independent"]}}
    result = RUNNERS[case](config)
    assert result["status"] == "PASSED" and not result["full_statistical_budget"]
    assert result["rows"][0]["seed_count"] == 2
    assert result["rows"][0]["failed_optimization_frequency"] == 0.
    assert all(len(r["damage_coverage"]["wilson_interval_95"]) == 2 for r in result["rows"][0]["future"])


@pytest.mark.scientific
def test_full_forward_inverse_noise_realization_is_deterministic():
    config = {"geometry": {"nx": 2, "ny": 1, "seed": 19}, "calibration": {"sigma_mm": 5e-5}}
    task = (config, False, "spatial", 1191, None)
    a, b = _calibration_trial(task), _calibration_trial(task)
    assert a["status"] == b["status"] == "CONVERGED"
    np.testing.assert_array_equal(a["physical_parameter_estimate"], b["physical_parameter_estimate"])
    np.testing.assert_array_equal(a["parameter_sd"], b["parameter_sd"])
    assert a["forecast"] == b["forecast"]
