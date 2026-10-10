from .solver import (DisplacementBC, MaterialAssignment, MechanicsBackendError,
                     MechanicsConvergenceError, MechanicsSolver, NodalForceBC,
                     SolverDiagnostics, SolverRequest, SolverResult, TractionBC)

__all__ = ["DisplacementBC", "MaterialAssignment", "MechanicsBackendError",
           "MechanicsConvergenceError", "MechanicsSolver", "NodalForceBC",
           "SolverDiagnostics", "SolverRequest", "SolverResult", "TractionBC"]

# SciPy stays optional: importing Phase 0 contracts does not load the backend.
def __getattr__(name):
    if name in {"Tri3MechanicsSolver", "Tri3SolverResult", "Tri3Assembly", "Tri3Diagnostics"}:
        from . import tri3
        return getattr(tri3, name)
    if name in {"Tri3SolverRequest", "Tri3SolverConfig", "BodyForceDensity"}:
        from . import tri3_contracts
        return getattr(tri3_contracts, name)
    raise AttributeError(name)
