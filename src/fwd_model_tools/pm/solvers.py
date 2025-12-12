from collections.abc import Callable
from typing import ClassVar, TypeAlias

from diffrax import AbstractSolver, ODETerm
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from .ode import single_ode, symplectic_fpm_ode, symplectic_ode

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]
Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]


def setup_odeterms(solver, cosmo, dx_field, dt0):
    if isinstance(solver, ReversibleSymplecticEuler) or len(solver.term_structure) == 2:
        drift, kick = symplectic_ode(cosmo, reference_field=dx_field)
        ode_term = (ODETerm(drift), ODETerm(kick))
        first_step_term = None
    elif isinstance(solver, ReversibleEfficientFastPM):
        drift, kick, first_step = symplectic_fpm_ode(cosmo, reference_field=dx_field, dt0=dt0, use_growth=True)
        ode_term = (ODETerm(drift), ODETerm(kick))
        first_step_term = ODETerm(first_step)
    elif isinstance(solver, ReversibleKickDriftKick):
        drift, kick, first_step = symplectic_fpm_ode(cosmo, reference_field=dx_field, dt0=dt0, use_growth=False)
        ode_term = (ODETerm(drift), ODETerm(kick))
        first_step_term = ODETerm(first_step)
    elif len(solver.term_structure) == 1:
        ode_term = ODETerm(single_ode(cosmo, reference_field=dx_field))
        first_step_term = None
    else:
        raise ValueError(f"Unknown solver type {type(solver)}.")
    return ode_term, first_step_term


class ReversibleBaseSolver(AbstractSolver):
    """Reversible Efficient FastPM integrator (semi-implicit Euler).

    Symplectic, reversible method optimized for particle-mesh simulations.
    Does not support adaptive step sizing. Uses 1st order local linear
    interpolation for dense/ts output. Supports reverse() for efficient
    adjoint computation in gradient-based inference.
    """

    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        return None

    def first_step(self, term, t0, dt0, y0, args):
        t1 = t0 + dt0
        control = term.contr(t0, t1)
        y0_1, y0_2 = y0

        y1_2 = (y0_2**ω + term.vf_prod(t0, y0_1, args, control)**ω).ω

        return (y0_1, y1_2)

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        term_1, term_2 = terms[:2]
        y0_1, y0_2 = y0

        control1 = term_1.contr(t0, t1)
        control2 = term_2.contr(t0, t1)
        y1_1 = (y0_1**ω + term_1.vf_prod(t0, y0_2, args, control1)**ω).ω
        y1_2 = (y0_2**ω + term_2.vf_prod(t0, y1_1, args, control2)**ω).ω

        y1 = (y1_1, y1_2)

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def reverse(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y1: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        term_1, term_2 = terms[:2]
        y1_1, y1_2 = y1
        control1 = term_1.contr(t0, t1)
        control2 = term_2.contr(t0, t1)

        y0_2 = (y1_2**ω - term_2.vf_prod(t0, y1_1, args, control2)**ω).ω
        y0_1 = (y1_1**ω - term_1.vf_prod(t0, y0_2, args, control1)**ω).ω

        y0 = (y0_1, y0_2)

        return y0

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        term_1, term_2 = terms[:2]
        y0_1, y0_2 = y0
        f1 = term_1.vf(t0, y0_2, args)
        f2 = term_2.vf(t0, y0_1, args)
        return f1, f2


class ReversibleSymplecticEuler(ReversibleBaseSolver):
    pass


class ReversibleEfficientFastPM(ReversibleBaseSolver):
    term_structure: ClassVar = (AbstractTerm, AbstractTerm, AbstractTerm)


class ReversibleKickDriftKick(ReversibleBaseSolver):
    term_structure: ClassVar = (AbstractTerm, AbstractTerm, AbstractTerm)
