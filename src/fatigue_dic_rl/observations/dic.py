"""Displacement-only measurement contract; no hidden-state access."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from fatigue_dic_rl.core.validation import ContractError, array, label, metadata, scalar


@dataclass(frozen=True, eq=False)
class DICObservation:
    coordinates: np.ndarray
    displacement: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    cycle_count: float
    loading_phase: str
    loading_state_ref: str
    coordinate_system: str
    coordinate_units: str
    displacement_units: str
    covariance: np.ndarray | None = None
    metadata: Mapping[str, str | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = array(self.coordinates, name="DIC coordinates")
        displacement = array(self.displacement, name="DIC displacement")
        mask = array(self.valid_mask, kind="bool", name="DIC mask")
        confidence = array(self.confidence, name="DIC confidence")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.all(np.isfinite(coordinates)):
            raise ContractError("DIC coordinates must be finite [Q, 2]")
        count = len(coordinates)
        if displacement.shape != (count, 2) or mask.shape != (count,) or confidence.shape != (count,):
            raise ContractError("DIC displacement/mask/confidence shapes must be [Q,2]/[Q]/[Q]")
        if not np.all(np.isfinite(displacement[mask])):
            raise ContractError("valid DIC displacements must be finite")
        if not np.all(np.isfinite(confidence)) or np.any(confidence < 0) or np.any(confidence > 1):
            raise ContractError("DIC confidence must be finite in [0, 1]")
        for name in ("loading_phase", "loading_state_ref", "coordinate_system"):
            label(getattr(self, name), name)
        if self.coordinate_units not in {"m", "mm"} or self.displacement_units not in {"m", "mm"}:
            raise ContractError("DIC coordinate/displacement units must explicitly be m or mm")
        if self.covariance is not None:
            cov = array(self.covariance, name="DIC covariance")
            if cov.shape != (count, 2, 2) or not np.all(np.isfinite(cov[mask])):
                raise ContractError("covariance must be [Q,2,2] and finite on valid points")
            valid_cov = cov[mask]
            scale = np.max(np.abs(valid_cov), axis=(1, 2))
            tol = 64 * np.finfo(float).eps * scale
            if np.any(np.max(np.abs(valid_cov - valid_cov.transpose(0, 2, 1)), axis=(1, 2)) > tol):
                raise ContractError("covariance must be symmetric")
            if np.any(np.linalg.eigvalsh(valid_cov) < -tol[:, None]):
                raise ContractError("covariance must be positive semidefinite")
            object.__setattr__(self, "covariance", cov)
        # Restrict metadata to acquisition information, avoiding a generic tensor
        # bag that could accidentally carry synthetic damage truth.
        allowed = {"source", "specimen_id", "camera_id", "calibration_id", "timestamp", "acquisition_id"}
        if not isinstance(self.metadata, Mapping) or set(self.metadata) - allowed:
            raise ContractError("DIC metadata accepts only acquisition keys, never damage labels")
        for value in self.metadata.values():
            if not isinstance(value, str):
                scalar(value, "acquisition metadata")
        for name, value in (("coordinates", coordinates), ("displacement", displacement),
                            ("valid_mask", mask), ("confidence", confidence)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "cycle_count", scalar(self.cycle_count, "cycle_count", minimum=0))
        object.__setattr__(self, "metadata", metadata(self.metadata))
