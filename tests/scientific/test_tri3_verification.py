from pathlib import Path
import tomllib

import numpy as np
import pytest

from fatigue_dic_rl.evaluation.tri3_verification import RUNNERS, manufactured_fields

pytestmark = pytest.mark.scientific


@pytest.mark.parametrize("name", ["tri3_patch_test", "uniaxial_tension", "damage_stiffness", "mesh_convergence"])
def test_scientific_verification_from_config(name):
    path = Path(__file__).resolve().parents[2]/"configs"/"phase1"/f"{name}.toml"
    config = tomllib.loads(path.read_text())
    result = RUNNERS[config["case"]](config)
    assert result["status"] == "PASS"


def test_manufactured_body_matches_independent_stress_divergence():
    points = np.array([[0.13, 0.29], [0.43, 0.71], [0.83, 0.57]])
    young, nu, amplitude, h = 2000, 0.3, 0.001, 1e-6
    C = young/(1-nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    def stress(q):
        return manufactured_fields(q, amplitude, young, nu)[1] @ C.T

    dx, dy = np.array([h, 0]), np.array([0, h])
    sx = (stress(points+dx)-stress(points-dx))/(2*h)
    sy = (stress(points+dy)-stress(points-dy))/(2*h)
    divergence = np.column_stack((sx[:, 0]+sy[:, 2], sx[:, 2]+sy[:, 1]))
    body = manufactured_fields(points, amplitude, young, nu)[2]
    np.testing.assert_allclose(divergence+body, 0, atol=3e-9, rtol=0)
