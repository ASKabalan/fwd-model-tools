"""
Core base classes for field objects.

This module contains the foundational abstract classes that define the
interface for all field types in fwd_model_tools.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["AbstractField"]


class AbstractField(ABC):
    """
    Minimal base class capturing the array payload shared by all field objects.
    """

    __slots__ = ("array",)
    __array_priority__ = 1000

    def __init__(self, *, array: jax.Array):
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shorthand for ``array.shape``."""
        return tuple(self.array.shape)

    @property
    def dtype(self) -> jnp.dtype:
        """Shorthand for ``array.dtype``."""
        return self.array.dtype

    @abstractmethod
    def replace(self, **updates: Any) -> "AbstractField":
        """Return a copy with the provided attributes replaced."""

    # ----------------------------------------------------------- math helpers
    def _coerce_other(self, other: Any) -> Any:
        return other.array if isinstance(other, AbstractField) else other

    def _binary_op(self, other: Any, op) -> "AbstractField":
        return self.replace(array=op(self.array, self._coerce_other(other)))

    def _binary_rop(self, other: Any, op) -> "AbstractField":
        return self.replace(array=op(self._coerce_other(other), self.array))

    def __add__(self, other: Any):
        return self._binary_op(other, lambda x, y: x + y)

    def __radd__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x + y)

    def __sub__(self, other: Any):
        return self._binary_op(other, lambda x, y: x - y)

    def __rsub__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x - y)

    def __mul__(self, other: Any):
        return self._binary_op(other, lambda x, y: x * y)

    def __rmul__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x * y)

    def __truediv__(self, other: Any):
        return self._binary_op(other, lambda x, y: x / y)

    def __rtruediv__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x / y)

    def __neg__(self):
        return self.replace(array=-self.array)

    def __pos__(self):
        return self
