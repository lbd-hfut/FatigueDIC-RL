"""Validation primitives. Arrays are copied into immutable byte-backed storage."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


class ContractError(ValueError):
    """A shape, unit, support or physical admissibility contract was violated."""


def scalar(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, str)) or np.ndim(value) != 0:
        raise ContractError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractError(f"{name} must be finite and >= {minimum}")
    return result


def array(value: Any, *, kind: str = "float", name: str = "array") -> np.ndarray:
    raw = np.asarray(value)
    if kind == "int":
        if raw.size and raw.dtype.kind not in "iu":
            raise ContractError(f"{name} must contain integers, without implicit rounding")
        if raw.size and raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
            raise ContractError(f"{name} indices exceed int64 range")
        dtype = np.int64
    elif kind == "bool":
        if raw.dtype.kind != "b":
            raise ContractError(f"{name} must contain booleans")
        dtype = np.bool_
    else:
        if raw.dtype.kind not in "fiu":
            raise ContractError(f"{name} must contain real numbers")
        dtype = np.float64
    owned = np.array(raw, dtype=dtype, copy=True, order="C")
    return np.frombuffer(owned.tobytes(), dtype=dtype).reshape(owned.shape)


def label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value


def metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise ContractError("metadata must be a mapping with string keys")
    # Metadata is descriptive, never a source of numerical contract information.
    return MappingProxyType(dict(value))
