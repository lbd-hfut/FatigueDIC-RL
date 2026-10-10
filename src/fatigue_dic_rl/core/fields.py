"""Location-aware fields; no implicit resampling or unit conversion."""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .validation import ContractError, array, label, metadata


class FieldLocation(str, Enum):
    NODE = "node"
    ELEMENT = "element"
    QUADRATURE = "quadrature"
    OBSERVATION = "observation"


@dataclass(frozen=True)
class FieldReference:
    space_id: str
    location: FieldLocation
    count: int
    mesh_id: str | None = None

    def __post_init__(self) -> None:
        label(self.space_id, "space_id")
        object.__setattr__(self, "location", FieldLocation(self.location))
        if isinstance(self.count, bool) or not isinstance(self.count, (int, np.integer)) or self.count < 0:
            raise ContractError("reference count must be a nonnegative integer")
        if self.mesh_id is not None:
            label(self.mesh_id, "mesh_id")


@dataclass(frozen=True, eq=False)
class FieldData:
    values: np.ndarray
    location: FieldLocation
    units: str
    reference: FieldReference | None = None
    coordinates: np.ndarray | None = None
    coordinate_units: str | None = None
    coordinate_system: str | None = None
    valid_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = array(self.values, name="field values")
        location = FieldLocation(self.location)
        label(self.units, "units")
        if values.ndim < 1:
            raise ContractError("field values require a leading spatial axis")
        count = values.shape[0]
        mask = np.ones(count, dtype=bool) if self.valid_mask is None else array(self.valid_mask, kind="bool")
        if mask.shape != (count,):
            raise ContractError("valid_mask must have shape [count]")
        if not np.all(np.isfinite(values[mask])):
            raise ContractError("valid field values must be finite")
        if self.reference is None and self.coordinates is None:
            raise ContractError("a field needs coordinates or a location reference")
        if self.reference is not None:
            if self.reference.location != location or self.reference.count != count:
                raise ContractError("reference location/count does not match the field")
        if self.coordinates is not None:
            coordinates = array(self.coordinates, name="field coordinates")
            if coordinates.shape != (count, 2) or not np.all(np.isfinite(coordinates)):
                raise ContractError("coordinates must be finite [count, 2]")
            if self.coordinate_units not in {"m", "mm"}:
                raise ContractError("coordinate_units must explicitly be m or mm")
            label(self.coordinate_system, "coordinate_system")
            object.__setattr__(self, "coordinates", coordinates)
        elif self.coordinate_units is not None or self.coordinate_system is not None:
            raise ContractError("coordinate metadata without coordinates is ambiguous")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "valid_mask", array(mask, kind="bool"))
        object.__setattr__(self, "metadata", metadata(self.metadata))

    @property
    def count(self) -> int:
        return self.values.shape[0]

    def require(self, *, location: FieldLocation | None = None,
                units: str | None = None, components: tuple[int, ...] | None = None,
                fully_valid: bool = False) -> None:
        if location is not None and self.location != location:
            raise ContractError(f"expected {location}, received {self.location}")
        if units is not None and self.units != units:
            raise ContractError(f"expected units {units}, received {self.units}; convert explicitly")
        if components is not None and self.values.shape[1:] != components:
            raise ContractError(f"expected component shape {components}")
        if fully_valid and not np.all(self.valid_mask):
            raise ContractError("this operation requires a fully valid field")

    def assert_same_support(self, other: "FieldData") -> None:
        if self.location != other.location or self.count != other.count:
            raise ContractError("field locations/counts differ; explicit mapping is required")
        if self.reference != other.reference:
            raise ContractError("field references differ")
        if (self.coordinates is None) != (other.coordinates is None):
            raise ContractError("coordinate declarations differ")
        if self.coordinates is not None:
            if (self.coordinate_units != other.coordinate_units or
                    self.coordinate_system != other.coordinate_system or
                    not np.array_equal(self.coordinates, other.coordinates)):
                raise ContractError("field coordinates or coordinate systems differ")
        if not np.array_equal(self.valid_mask, other.valid_mask):
            raise ContractError("field validity masks differ")

    def assert_compatible(self, other: "FieldData") -> None:
        self.assert_same_support(other)
        if self.units != other.units or self.values.shape != other.values.shape:
            raise ContractError("field units/component shapes differ")

    def with_values(self, values: np.ndarray, *, units: str | None = None) -> "FieldData":
        return replace(self, values=values, units=self.units if units is None else units)
