"""Legacy diffrax-based ODE integration for N-body simulations.

This module provides the original diffrax-based integration code that has been
moved to a legacy submodule. Diffrax is now an optional dependency.

Install with: pip install fwd-model-tools[legacy]
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

# Type variables for decorator
Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")


def require_diffrax(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Decorator that raises ImportError if diffrax is not available."""
    try:
        from diffrax import diffeqsolve  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 'diffrax'. Install with: pip install fwd-model-tools[legacy]"
        raise ImportError(msg)

    return deferred_func


# Lazy imports to avoid requiring diffrax at module import time
def __getattr__(name: str):
    """Lazy import of diffrax-dependent modules."""
    if name == "integrate":
        from .diffrax_integrate import integrate

        return integrate
    if name == "reverse_adjoint_integrate":
        from .diffrax_integrate import reverse_adjoint_integrate

        return reverse_adjoint_integrate
    if name == "scan_integrate":
        from .diffrax_integrate import scan_integrate

        return scan_integrate
    if name == "ReversibleBaseSolver":
        from .diffrax_solver import ReversibleBaseSolver

        return ReversibleBaseSolver
    if name == "single_ode":
        from .ode import single_ode

        return single_ode
    if name == "symplectic_fpm":
        from .ode import symplectic_fpm

        return symplectic_fpm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "require_diffrax",
    "integrate",
    "reverse_adjoint_integrate",
    "scan_integrate",
    "ReversibleBaseSolver",
    "single_ode",
    "symplectic_fpm",
]
