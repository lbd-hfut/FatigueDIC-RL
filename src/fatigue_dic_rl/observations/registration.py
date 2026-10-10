"""Explicit pixel calibration and additive small-angle registration error."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar


@dataclass(frozen=True)
class PixelCalibration:
    mm_per_pixel: float
    origin_mm: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        s = scalar(self.mm_per_pixel, "mm_per_pixel", minimum=0)
        if s == 0 or np.shape(self.origin_mm) != (2,) or not np.isfinite(self.origin_mm).all():
            raise ContractError("calibration needs positive scale and finite origin mm")
        object.__setattr__(self, "mm_per_pixel", s)
        object.__setattr__(self, "origin_mm", tuple(float(x) for x in self.origin_mm))

    def coordinates_to_mm(self, coordinates, units):
        x = np.asarray(coordinates, dtype=float)
        if units == "pixel":
            return x*self.mm_per_pixel+np.asarray(self.origin_mm)
        if units == "mm":
            return x.copy()
        raise ContractError("coordinates require explicit mm or pixel units")

    def displacement_to_mm(self, displacement, units):
        u = np.asarray(displacement, dtype=float)
        if units == "pixel":
            return u*self.mm_per_pixel
        if units == "mm":
            return u.copy()
        raise ContractError("displacement requires explicit mm or pixel units")

    def covariance_to_mm2(self, covariance, units):
        a = np.asarray(covariance, dtype=float)
        if units == "pixel^2":
            return a*self.mm_per_pixel**2
        if units == "mm^2":
            return a.copy()
        raise ContractError("covariance requires explicit mm^2 or pixel^2 units")


@dataclass(frozen=True, eq=False)
class RegistrationError:
    translation_mm: np.ndarray
    rotation_radians: float = 0.
    center_mm: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        t = array(self.translation_mm, name="registration translation mm")
        theta = scalar(self.rotation_radians, "registration radians")
        if t.shape != (2,) or not np.isfinite(t).all() or abs(theta) > .05 or np.shape(self.center_mm) != (2,) or not np.isfinite(self.center_mm).all():
            raise ContractError("registration requires finite mm translation and small rotation <=.05 rad")
        object.__setattr__(self, "translation_mm", t)
        object.__setattr__(self, "rotation_radians", theta)
        object.__setattr__(self, "center_mm", tuple(float(x) for x in self.center_mm))

    def field(self, coordinates_mm):
        x = np.asarray(coordinates_mm)-self.center_mm
        return self.translation_mm+self.rotation_radians*np.column_stack((-x[:, 1], x[:, 0]))
