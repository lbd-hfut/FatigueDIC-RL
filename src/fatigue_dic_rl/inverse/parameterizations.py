"""Continuous coordinate-evaluated damage models and dimensionless parameters."""
from dataclasses import dataclass, field
from types import MappingProxyType
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float
    units: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.units not in {"1", "mm", "rad", "N/mm^2", "cycle^-1"}:
            raise ContractError("parameter needs a name and explicit physical units")
        lo, hi = scalar(self.lower, "lower bound"), scalar(self.upper, "upper bound")
        if lo >= hi:
            raise ContractError("parameter bounds must have positive width")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)


@dataclass(frozen=True)
class ParameterSpace:
    parameters: tuple[Parameter, ...]
    fixed: dict = field(default_factory=dict)

    def __post_init__(self):
        ps = tuple(self.parameters)
        if not ps or any(not isinstance(p, Parameter) for p in ps) or len({p.name for p in ps}) != len(ps):
            raise ContractError("parameter space needs unique Parameter entries")
        if set(self.fixed) & {p.name for p in ps}:
            raise ContractError("fixed and unknown parameters overlap")
        object.__setattr__(self, "parameters", ps)
        object.__setattr__(self, "fixed", MappingProxyType({k: scalar(v, k) for k, v in self.fixed.items()}))

    @property
    def names(self):
        return tuple(p.name for p in self.parameters)

    @property
    def scales(self):
        return np.array([(p.upper-p.lower)/2 for p in self.parameters])

    def decode(self, dimensionless):
        z = np.asarray(dimensionless, dtype=float)
        if z.shape != (len(self.parameters),) or not np.isfinite(z).all() or np.any(np.abs(z) > 1):
            raise ContractError("dimensionless parameters must lie in [-1,1]^K")
        return dict(self.fixed, **{p.name: (p.lower+p.upper)/2+zi*(p.upper-p.lower)/2 for p, zi in zip(self.parameters, z)})

    def encode(self, physical):
        z = np.array([(physical[p.name]-(p.lower+p.upper)/2)/((p.upper-p.lower)/2) for p in self.parameters])
        self.decode(z)
        return z


def _coordinates(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ContractError("damage evaluation needs finite [Q,2] physical mm coordinates")
    return x


def _legal(d, maximum):
    if not np.isfinite(d).all() or np.any(d < 0) or np.any(d >= maximum):
        raise ContractError("candidate damage must satisfy 0<=D<Dmax; clipping is forbidden")
    return d


@dataclass(frozen=True)
class UniformDamage:
    damage_max: float = .95

    def __post_init__(self):
        if not 0 < scalar(self.damage_max, "Dmax") < 1:
            raise ContractError("Dmax must be in (0,1)")

    @property
    def parameters(self):
        return (Parameter("d0", 0., .8*self.damage_max, "1"),)

    def evaluate(self, coordinates_mm, physical):
        return _legal(np.full(len(_coordinates(coordinates_mm)), scalar(physical["d0"], "d0")), self.damage_max)


@dataclass(frozen=True)
class GaussianDamage:
    """Rectangular bounds guarantee d_base+A<=.75<Dmax everywhere."""
    damage_max: float = .95

    def __post_init__(self):
        if not .75 < scalar(self.damage_max, "Dmax") < 1:
            raise ContractError("this Gaussian parameter box requires .75<Dmax<1")

    @property
    def parameters(self):
        return (Parameter("d_base", 0., .15, "1"), Parameter("amplitude", 0., .6, "1"),
                Parameter("xc", .3, 1.7, "mm"), Parameter("yc", .1, .9, "mm"),
                Parameter("sx", .12, .7, "mm"), Parameter("sy", .08, .45, "mm"))

    def evaluate(self, coordinates_mm, physical):
        x = _coordinates(coordinates_mm)
        values = {s.name: scalar(physical[s.name], s.name) for s in self.parameters}
        if any(not s.lower <= values[s.name] <= s.upper for s in self.parameters):
            raise ContractError("Gaussian parameters outside declared physical bounds")
        p = values
        exponent = -.5*((x[:, 0]-p["xc"])/p["sx"])**2-.5*((x[:, 1]-p["yc"])/p["sy"])**2
        return _legal(p["d_base"]+p["amplitude"]*np.exp(exponent), self.damage_max)


@dataclass(frozen=True, eq=False)
class LowRankDamage:
    centers_mm: np.ndarray
    width_mm: float = .35
    base: float = .25
    coefficient_bound: float = .012
    damage_max: float = .95

    def __post_init__(self):
        c = array(_coordinates(self.centers_mm), name="basis centers mm")
        w, b, a, dm = (scalar(v, n) for v, n in ((self.width_mm, "basis width mm"), (self.base, "basis base"),
                            (self.coefficient_bound, "coefficient bound"), (self.damage_max, "Dmax")))
        if not len(c) or w <= 0 or a <= 0 or not 0 < dm < 1 or b-len(c)*a < 0 or b+len(c)*a >= dm:
            raise ContractError("basis box must guarantee legal damage globally")
        object.__setattr__(self, "centers_mm", c)

    @property
    def parameters(self):
        return tuple(Parameter(f"a{i}", -self.coefficient_bound, self.coefficient_bound, "1") for i in range(len(self.centers_mm)))

    def basis(self, coordinates_mm):
        x = _coordinates(coordinates_mm)
        return np.exp(-np.sum((x[:, None]-self.centers_mm[None])**2, axis=2)/(2*self.width_mm**2))

    def evaluate(self, coordinates_mm, physical):
        a = np.array([scalar(physical[s.name], s.name) for s in self.parameters])
        if np.any(np.abs(a) > self.coefficient_bound):
            raise ContractError("basis coefficients outside physical box")
        return _legal(self.base+self.basis(coordinates_mm) @ a, self.damage_max)
