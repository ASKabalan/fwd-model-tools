"""Shared internal validation helpers for field metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _ensure_matplotlib_installed() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting functionality. Please install it via 'pip install matplotlib'."
        ) from e


def _ensure_tuple3(name: str, values: Iterable[Any], *, cast: type) -> tuple[Any, Any, Any]:
    seq = tuple(values)
    if len(seq) != 3:
        raise ValueError(f"{name} must be a tuple of length 3, got {seq}")
    return tuple(cast(v) for v in seq)


def _ensure_tuple2(name: str, values: Iterable[Any], *, cast: type) -> tuple[Any, Any]:
    seq = tuple(values)
    if len(seq) != 2:
        raise ValueError(f"{name} must be a tuple of length 2, got {seq}")
    return tuple(cast(v) for v in seq)


def _normalize_halo_size(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        left, right = _ensure_tuple2("halo_size", value, cast=int)
    else:
        left = right = int(value)
    if left < 0 or right < 0:
        raise ValueError("halo_size entries must be non-negative")
    return left, right


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return ivalue


def _optional_tuple2_positive(name: str, value: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    x, y = _ensure_tuple2(name, value, cast=int)
    if x <= 0 or y <= 0:
        raise ValueError(f"{name} entries must be positive, got {(x, y)}")
    return x, y
