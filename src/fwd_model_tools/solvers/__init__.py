"""
ODE solvers and custom integration methods.

This module provides custom ODE integration routines with reverse-mode automatic
differentiation support, including semi-implicit Euler solver and memory-efficient
adjoint integration.
"""

from fwd_model_tools.solvers.integrate import integrate, scan_integrate
from fwd_model_tools.solvers.semi_implicit_euler import SemiImplicitEuler

__all__ = [
    "integrate",
    "scan_integrate",
    "SemiImplicitEuler",
]
