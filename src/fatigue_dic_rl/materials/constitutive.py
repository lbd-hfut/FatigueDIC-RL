"""Constitutive mechanics lives here, never in the core field container."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core import FieldData, FieldLocation
from fatigue_dic_rl.core.validation import ContractError, array, scalar


@dataclass(frozen=True, eq=False)
class ConstitutiveResponse:
    stress: FieldData
    tangent: np.ndarray
    tangent_units: str
    internal_variables: Mapping[str, FieldData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tangent = array(self.tangent, name="constitutive tangent")
        self.stress.require(fully_valid=True)
        if self.stress.values.ndim < 2:
            raise ContractError("constitutive stress requires declared tensor/vector components")
        components = int(np.prod(self.stress.values.shape[1:]))
        if tangent.shape != (self.stress.count, components, components) or not np.all(np.isfinite(tangent)):
            raise ContractError("tangent shape must be [count, flattened components, flattened components]")
        if self.tangent_units != self.stress.units:
            raise ContractError("tangent and stress units differ")
        object.__setattr__(self, "tangent", tangent)
        if not isinstance(self.internal_variables, Mapping):
            raise ContractError("constitutive internal variables need a field mapping")
        for name, value in self.internal_variables.items():
            if not isinstance(name, str) or not name or not isinstance(value, FieldData):
                raise ContractError("constitutive internal variables need named FieldData")
            value.require(fully_valid=True)
            self.stress.assert_same_support(value)
        object.__setattr__(self, "internal_variables", MappingProxyType(dict(self.internal_variables)))


class ConstitutiveModel(ABC):
    @abstractmethod
    def evaluate(self, strain: FieldData, damage: FieldData,
                 parameters: Mapping[str, float], *,
                 internal_variables: Mapping[str, FieldData] | None = None) -> ConstitutiveResponse:
        """Evaluate on explicitly collocated fields; models define their tensor convention.

        Future history-dependent models receive and return local internal fields.
        Tensor ordering and tangent flattening are declared by each concrete
        model. Models needing more inputs (e.g. time increment) use a documented
        model-specific context extension, rather than overloading material units.
        """


class IsotropicPlaneStress(ConstitutiveModel):
    """Small-strain example C(D)=(1-D)C0; [exx, eyy, gamma_xy].

    Stress components: [sxx, syy, sxy], units Pa. E must be in Pa,
    -1 < nu < 0.5. A zero tangent at D=1 is allowed by this material
    contract; a solver must detect the resulting singularity if applicable.
    """

    def evaluate(self, strain: FieldData, damage: FieldData,
                 parameters: Mapping[str, float], *,
                 internal_variables: Mapping[str, FieldData] | None = None) -> ConstitutiveResponse:
        if internal_variables:
            raise ContractError("this stateless elastic model does not consume history variables")
        strain.require(units="1", components=(3,), fully_valid=True)
        damage.require(units="1", components=(), fully_valid=True)
        strain.assert_same_support(damage)
        if strain.location == FieldLocation.OBSERVATION:
            raise ContractError("constitutive response requires a mechanics field location")
        if np.any(damage.values < 0) or np.any(damage.values > 1):
            raise ContractError("constitutive damage outside [0, 1]")
        if "young_modulus_pa" not in parameters or "poisson_ratio" not in parameters:
            raise ContractError("plane stress needs young_modulus_pa and poisson_ratio")
        young = scalar(parameters["young_modulus_pa"], "young_modulus_pa", minimum=0)
        nu = scalar(parameters["poisson_ratio"], "poisson_ratio")
        if young == 0 or not -1 < nu < 0.5:
            raise ContractError("E must be positive and -1 < nu < 0.5")
        base = young / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        tangent = (1 - damage.values[:, None, None]) * base
        stress = np.einsum("nij,nj->ni", tangent, strain.values)
        return ConstitutiveResponse(strain.with_values(stress, units="Pa"), tangent, "Pa")
