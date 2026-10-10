"""Shared data contracts independent of solvers and neural networks."""

from .fields import FieldData, FieldLocation, FieldReference
from .state import CycleLoading, LoadingState, MechanicsState, StateRole

__all__ = ["FieldData", "FieldLocation", "FieldReference", "CycleLoading",
           "LoadingState", "MechanicsState", "StateRole"]
