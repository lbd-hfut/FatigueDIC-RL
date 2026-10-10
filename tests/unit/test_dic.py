from dataclasses import fields, replace

import numpy as np
import pytest

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.observations import DICObservation


@pytest.fixture
def observation():
    return DICObservation([[0.3, 0.2], [2, 3]], [[0.1, 0.2], [np.nan, np.nan]], [True, False], [0.95, 0],
                          3, "peak", "load", "specimen", "mm", "mm",
                          covariance=[[[0.01, 0], [0, 0.01]], [[np.nan, np.nan], [np.nan, np.nan]]])


def test_dic_has_no_damage_channel(observation):
    assert "damage" not in {f.name for f in fields(observation)}
    with pytest.raises(ContractError):
        replace(observation, metadata={"damage": [0.1, 0.9]})
    with pytest.raises(ContractError):
        replace(observation, metadata={"source": np.array([0.1, 0.9])})


@pytest.mark.parametrize("updates", [{"coordinates": [[0, 1, 2], [1, 2, 3]]},
                                     {"displacement": [[1, 2]]}, {"valid_mask": [True]},
                                     {"confidence": [1.1, 0]}, {"confidence": [np.nan, 0]},
                                     {"displacement": [[np.nan, 0], [0, 0]]},
                                     {"cycle_count": -1}, {"displacement_units": "px"},
                                     {"covariance": np.ones((2, 3, 3))},
                                     {"covariance": [[[1, 1], [0, 1]], [[0, 0], [0, 0]]]},
                                     {"covariance": [[[1, 0], [0, -0.1]], [[0, 0], [0, 0]]]}])
def test_invalid_dic_contract(observation, updates):
    with pytest.raises(ContractError):
        replace(observation, **updates)


def test_empty_dic():
    obs = DICObservation(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), np.empty(0),
                         0, "peak", "load", "specimen", "m", "m", covariance=np.empty((0, 2, 2)))
    assert obs.coordinates.shape == (0, 2)
