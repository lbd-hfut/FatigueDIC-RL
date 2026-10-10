"""Test doubles exercise interface validation; none performs an FE solve."""

from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import StateRole
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.materials import IsotropicPlaneStress
from fatigue_dic_rl.mechanics import (DisplacementBC, MaterialAssignment, MechanicsBackendError,
    MechanicsConvergenceError, MechanicsSolver, NodalForceBC, SolverDiagnostics, SolverRequest, SolverResult, TractionBC)


@pytest.fixture
def solver_request(mesh, damage, state):
    material = MaterialAssignment(IsotropicPlaneStress(), {"young_modulus_pa": 200e9, "poisson_ratio": 0.3})
    return SolverRequest(mesh, damage, {0: material, 1: material}, state.loading_state, 0,
                         displacement_bcs=(DisplacementBC([0, 0, 1], [0, 1, 1], [0, 0, 0], "mm"),),
                         nodal_force_bcs=(NodalForceBC([2], [[0, 1]]),),
                         traction_bcs=(TractionBC("bottom", [0, 1]),), thickness_m=0.001)


def test_solver_is_abstract():
    with pytest.raises(TypeError):
        MechanicsSolver()


def test_heterogeneous_material_bc_request(solver_request):
    assert set(solver_request.materials) == {0, 1}
    assert solver_request.traction_bcs[0].units == "Pa"


@pytest.mark.parametrize("updates", [{"materials": {}}, {"cycle_count": -1}, {"thickness_m": 0},
                                     {"displacement_bcs": (DisplacementBC([20], [0], [0], "mm"),)},
                                     {"displacement_bcs": (DisplacementBC([0], [0], [0], "m"),)},
                                     {"nodal_force_bcs": (NodalForceBC([50], [[0, 1]]),)},
                                     {"traction_bcs": (TractionBC("unknown", [0, 1]),)},
                                     {"displacement_bcs": (DisplacementBC([0, 0], [0, 0], [0, 1], "mm"),)}])
def test_invalid_solver_input(solver_request, updates):
    with pytest.raises(ContractError):
        replace(solver_request, **updates)


def test_backend_convergence_and_errors(solver_request, state, nodal_displacement, damage):
    response = replace(state, role=StateRole.PREDICTION, displacement=nodal_displacement,
                       strain=damage.with_values(np.zeros((2, 3))),
                       stress=damage.with_values(np.zeros((2, 3)), units="Pa"))

    class ContractDouble(MechanicsSolver):
        def __init__(self, result):
            self.result = result

        def _solve(self, solver_request):
            return self.result

    success = SolverResult(response, SolverDiagnostics(True, "contract-double-no-fe", 1, 0))
    assert ContractDouble(success).solve(solver_request).state is response
    with pytest.raises(ContractError):
        ContractDouble(replace(success, state=state)).solve(solver_request)
    failure = replace(success, diagnostics=SolverDiagnostics(False, "contract-double-no-fe", 2, 1, "singular"))
    with pytest.raises(MechanicsConvergenceError) as exc:
        ContractDouble(failure).solve(solver_request)
    assert exc.value.diagnostics.message == "singular"

    class BrokenBackend(MechanicsSolver):
        def _solve(self, solver_request):
            raise np.linalg.LinAlgError("singular")

    with pytest.raises(MechanicsBackendError) as exc:
        BrokenBackend().solve(solver_request)
    assert isinstance(exc.value.__cause__, np.linalg.LinAlgError)


def test_solver_must_not_evolve_damage(solver_request, state, nodal_displacement, damage):
    class WrongBackend(MechanicsSolver):
        def _solve(self, solver_request):
            return SolverResult(replace(state, role="prediction", damage=damage.with_values([0.1, 0.2]),
                                        displacement=nodal_displacement, strain=damage.with_values(np.zeros((2, 3))),
                                        stress=damage.with_values(np.zeros((2, 3)), units="Pa")),
                                SolverDiagnostics(True, "contract-double-no-fe", 1, 0))

    with pytest.raises(ContractError):
        WrongBackend().solve(solver_request)
