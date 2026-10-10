"""Synthetic DIC-like measurement generation, with no damage input or output."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from .dic import DICObservation
from .covariance import DisplacementCovariance
from .registration import PixelCalibration, RegistrationError


@dataclass(frozen=True, eq=False)
class DICMeasurement:
    observation: DICObservation
    covariance: DisplacementCovariance
    roi_mask: np.ndarray

    def __post_init__(self):
        o = self.observation
        if not isinstance(o, DICObservation) or not isinstance(self.covariance, DisplacementCovariance):
            raise ContractError("measurement needs DICObservation and DisplacementCovariance")
        roi = array(self.roi_mask, kind="bool", name="ROI mask")
        if (o.coordinate_units != "mm" or o.displacement_units != "mm" or
                roi.shape != o.valid_mask.shape or len(self.covariance.matrix) != 2*len(roi)):
            raise ContractError("measurement must use physical mm and matching point support")
        object.__setattr__(self, "roi_mask", roi)

    @property
    def active_mask(self):
        return self.roi_mask & self.observation.valid_mask

    def save(self, folder):
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=False)
        o = self.observation
        np.savez_compressed(path/"observation.npz", coordinates_mm=o.coordinates,
            displacement_mm=o.displacement, valid_mask=o.valid_mask, roi_mask=self.roi_mask,
            confidence=o.confidence, covariance_mm2=self.covariance.matrix)
        (path/"metadata.json").write_text(json.dumps({"schema": "observation-only-1c-v1",
            "cycle_count": o.cycle_count, "loading_phase": o.loading_phase,
            "loading_state_ref": o.loading_state_ref, "coordinate_system": o.coordinate_system,
            "units": {"coordinates": "mm", "displacement": "mm", "covariance": "mm^2"},
            "acquisition": dict(o.metadata)}, indent=2)+"\n")

    @classmethod
    def load(cls, folder):
        path = Path(folder)
        m = json.loads((path/"metadata.json").read_text())
        if m["schema"] != "observation-only-1c-v1" or m["units"] != {"coordinates": "mm", "displacement": "mm", "covariance": "mm^2"}:
            raise ContractError("invalid observation-only schema/units")
        with np.load(path/"observation.npz", allow_pickle=False) as a:
            expected = {"coordinates_mm", "displacement_mm", "valid_mask", "roi_mask", "confidence", "covariance_mm2"}
            if set(a.files) != expected:
                raise ContractError("observation-only data contains unexpected channels")
            o = DICObservation(a["coordinates_mm"], a["displacement_mm"], a["valid_mask"], a["confidence"],
                m["cycle_count"], m["loading_phase"], m["loading_state_ref"], m["coordinate_system"],
                "mm", "mm", metadata=m["acquisition"])
            return cls(o, DisplacementCovariance(a["covariance_mm2"]), a["roi_mask"])


def calibrated_measurement(coordinates, displacement, *, coordinate_units, displacement_units,
                           calibration, covariance, valid_mask, roi_mask, confidence,
                           covariance_units="mm^2", cycle_count=0., load_id="static"):
    if not isinstance(calibration, PixelCalibration):
        raise ContractError("an explicit PixelCalibration is required")
    if isinstance(covariance, DisplacementCovariance):
        if covariance_units != "mm^2":
            raise ContractError("unit-bearing physical covariance cannot be relabelled pixel^2")
    else:
        covariance = DisplacementCovariance(calibration.covariance_to_mm2(covariance, covariance_units))
    o = DICObservation(calibration.coordinates_to_mm(coordinates, coordinate_units),
        calibration.displacement_to_mm(displacement, displacement_units), valid_mask, confidence,
        cycle_count, "static", load_id, "specimen", "mm", "mm")
    return DICMeasurement(o, covariance, roi_mask)


def simulate_measurement(coordinates_mm, exact_displacement_mm, covariance, *, seed,
                         domain_mask=None, roi_mask=None, missing_fraction=0., confidence=None,
                         registration=None, cycle_count=0., load_id="static"):
    """Returns only noisy displacement, masks and covariance; no true state channel.

    Registration is deterministic acquisition bias, not absorbed into the
    covariance. Its generating values belong in the experiment configuration.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("seed must be a nonnegative integer")
    x = np.asarray(coordinates_mm, dtype=float)
    u = np.asarray(exact_displacement_mm, dtype=float)
    q = len(x)
    if x.shape != (q, 2) or u.shape != (q, 2) or not np.isfinite(x).all():
        raise ContractError("measurement coordinates/displacement need matching [Q,2] mm")
    if not isinstance(covariance, DisplacementCovariance) or len(covariance.matrix) != 2*q:
        raise ContractError("noise covariance and coordinates disagree")
    missing = scalar(missing_fraction, "missing fraction", minimum=0)
    if missing > 1:
        raise ContractError("missing fraction must be <=1")
    valid = np.ones(q, dtype=bool) if domain_mask is None else array(domain_mask, kind="bool", name="domain mask").copy()
    if valid.shape != (q,):
        raise ContractError("domain mask needs [Q]")
    rng = np.random.default_rng(seed)
    valid &= rng.random(q) >= missing
    noisy = u+covariance.sample(rng)[0]
    if registration is not None:
        if not isinstance(registration, RegistrationError):
            raise ContractError("registration needs RegistrationError")
        noisy += registration.field(x)
    noisy[~valid] = np.nan
    o = DICObservation(x, noisy, valid, np.ones(q) if confidence is None else confidence,
        cycle_count, "static", load_id, "specimen", "mm", "mm", metadata={"source": "synthetic_DIC_like"})
    return DICMeasurement(o, covariance, np.ones(q, dtype=bool) if roi_mask is None else roi_mask)
