"""Physics-only local scalar fatigue evolution; no learning components."""

from .history import HistoryStatus, MaterialHistoryState
from .loading import FatigueLoadCycle, LoadingMode
from .models import (ConstantFatigueHazardLaw, EnergyAmplitudeDriver, FatigueDamageLaw,
                     FatigueDrivingForce, FatigueDrivingForceResult, FatigueLawParameters,
                     SyntheticFatigueDamageLaw)
from .solver import (FatigueEvolutionSolver, FatigueFailureDiagnostic, FatigueSolverConfig,
                     FatigueStepError, FatigueStepRequest, FatigueStepResult, FatigueSubstepResult)
from .trajectory import FatigueTrajectory, load_observations, run_trajectory, save_observations

__all__ = ["HistoryStatus", "MaterialHistoryState", "FatigueLoadCycle", "LoadingMode",
           "ConstantFatigueHazardLaw", "EnergyAmplitudeDriver", "FatigueDamageLaw", "FatigueDrivingForce",
           "FatigueDrivingForceResult", "FatigueLawParameters", "SyntheticFatigueDamageLaw",
           "FatigueEvolutionSolver", "FatigueFailureDiagnostic", "FatigueSolverConfig", "FatigueStepError",
           "FatigueStepRequest", "FatigueStepResult", "FatigueSubstepResult", "FatigueTrajectory",
           "load_observations", "run_trajectory", "save_observations"]
