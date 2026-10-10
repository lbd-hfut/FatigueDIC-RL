import numpy as np
import pytest

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.materials import ConstitutiveResponse, IsotropicPlaneStress


def test_history_variables_are_explicit_and_cannot_be_ignored(damage):
    strain = damage.with_values(np.zeros((2, 3)))
    with pytest.raises(ContractError):
        IsotropicPlaneStress().evaluate(strain, damage, {"young_modulus_pa": 200e9, "poisson_ratio": 0.3},
                                        internal_variables={"plastic_strain": strain})
    stress = strain.with_values(np.zeros((2, 3)), units="Pa")
    response = ConstitutiveResponse(stress, np.zeros((2, 3, 3)), "Pa", {"plastic_strain": strain})
    assert response.internal_variables["plastic_strain"].units == "1"
    with pytest.raises(TypeError):
        response.internal_variables["new"] = strain


def test_response_supports_explicit_tensor_component_count(damage):
    stress = damage.with_values(np.zeros((2, 2, 2)), units="Pa")
    response = ConstitutiveResponse(stress, np.zeros((2, 4, 4)), "Pa")
    assert response.tangent.shape == (2, 4, 4)
    with pytest.raises(ContractError):
        ConstitutiveResponse(stress, np.zeros((2, 3, 3)), "Pa")
    with pytest.raises(ContractError):
        ConstitutiveResponse(stress, np.zeros((2, 4, 4)), "MPa")
