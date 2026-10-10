"""Explicit mm/N adapter around the unchanged Phase 0 Pa constitutive model."""

from typing import Mapping

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError, scalar
from .constitutive import ConstitutiveModel, ConstitutiveResponse, IsotropicPlaneStress


class IsotropicPlaneStressMM(ConstitutiveModel):
    """E/stress/tangent: N/mm^2; strain: [exx, eyy, engineering gamma_xy]."""

    def evaluate(self, strain: FieldData, damage: FieldData,
                 parameters: Mapping[str, float], *,
                 internal_variables: Mapping[str, FieldData] | None = None) -> ConstitutiveResponse:
        if strain.metadata.get("shear_convention", "engineering") != "engineering":
            raise ContractError("plane-stress Voigt input requires engineering gamma_xy, not tensor epsilon_xy")
        if "young_modulus_n_per_mm2" not in parameters or "young_modulus_pa" in parameters:
            raise ContractError("mm material needs young_modulus_n_per_mm2, without a competing Pa parameter")
        young = scalar(parameters["young_modulus_n_per_mm2"], "young_modulus_n_per_mm2", minimum=0)
        converted = dict(parameters, young_modulus_pa=young*1e6)
        response = IsotropicPlaneStress().evaluate(strain, damage, converted,
                                                   internal_variables=internal_variables)
        return ConstitutiveResponse(response.stress.with_values(response.stress.values/1e6, units="N/mm^2"),
                                    response.tangent/1e6, "N/mm^2", response.internal_variables)
