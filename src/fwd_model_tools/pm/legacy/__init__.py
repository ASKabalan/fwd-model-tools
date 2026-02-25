"""Legacy diffrax-based ODE integration for N-body simulations.

This module provides the original diffrax-based integration code that has been
moved to a legacy submodule. Diffrax is now an optional dependency.

Install with: pip install fwd-model-tools[legacy]
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from .diffrax_integrate import integrate, reverse_adjoint_integrate, scan_integrate
from .ode import single_ode, symplectic_fpm
from .reversible_base_solver import ReversibleBaseSolver

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


__all__ = [
    "require_diffrax",
    "integrate",
    "reverse_adjoint_integrate",
    "scan_integrate",
    "ReversibleBaseSolver",
    "single_ode",
    "symplectic_fpm",
]
