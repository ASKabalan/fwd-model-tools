from functools import partial
from typing import Any, Literal, Tuple

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact

from ..fields import ParticleField
from .solvers import AbstractNBodySolver, NBodyState

AdjointType = Literal['reverse', 'checkpointed']


def _clip_to_end(tprev, tnext, t1):
    """
    Clips the computed next time step to ensure it does not exceed the final time t1, within a tolerance.

    The tolerance is set based on the data type of tnext (1e-10 for float64, otherwise 1e-6).

    Args:
        tprev: The previous time step value.
        tnext: The proposed next time step value.
        t1: The final time for integration.

    Returns:
        The adjusted next time step, set to t1 if tnext is within the tolerance of t1.
    """
    tol = 1e-10 if tnext.dtype == jnp.dtype("float64") else 1e-6
    clip = tnext > t1 - tol
    return jnp.where(clip, t1, tnext)


def _clip_to_start(tprev, tnext, t0):
    """
    Clips the computed previous time step to ensure it does not fall below the initial time t0, within a tolerance.

    The tolerance is determined by the data type of tprev (1e-10 for float64, otherwise 1e-6).

    Args:
        tprev: The computed previous time step value.
        tnext: The current time step value.
        t0: The initial time for integration.

    Returns:
        The adjusted previous time step, set to t0 if tprev is within the tolerance of t0.
    """
    tol = 1e-10 if tprev.dtype == jnp.dtype("float64") else 1e-6
    clip = tprev < t0 + tol
    return jnp.where(clip, t0, tprev)


def integrate(displacements: ParticleField,
              velocities: ParticleField,
              cosmo: Any,
              ts: jnp.ndarray,
              solver: AbstractNBodySolver,
              t0: float,
              t1: float,
              dt0: float,
              adjoint: AdjointType = 'checkpointed') -> Any:
    """
    Main integration entry point.

    Integrates an N-body system from t0 through snapshot times ts using the provided solver.
    Supports two adjoint modes for gradient computation:
      - 'checkpointed': Memory-efficient autodiff using equinox checkpointed scan
      - 'reverse': Custom VJP with explicit backward pass using solver.reverse()

    Args:
        y0: Initial state tuple (displacement, velocities) as ParticleFields.
        cosmo: Cosmology parameters.
        ts: Array of snapshot times at which to save output.
        solver: An AbstractNBodySolver instance.
        t0: Initial time for integration.
        t1: Final time for integration (used to initialize solver).
        dt0: Step size for integration.
        adjoint: Adjoint mode, either 'checkpointed' or 'reverse'.

    Returns:
        Snapshots at each time in ts, as returned by solver.save_at().

    Raises:
        ValueError: If adjoint='reverse' is used with a non-reversible correction kernel.
    """
    # Validate reversibility compatibility
    if adjoint == 'reverse':
        if not solver.pgd_kernel.reversible:
            raise ValueError(f"Cannot use adjoint='reverse' with {type(solver.pgd_kernel).__name__}. "
                             f"Use SharpeningKernel for reversible integration, or use adjoint='checkpointed'.")

    (ts, ) = promote_dtypes_inexact(ts)

    # Initialize solver OUTSIDE the loop
    disp0, vel0 = displacements, velocities
    t1_init = t0 + dt0
    disp, vel, state , ts = solver.init(disp0, vel0, t0, t1_init, dt0, ts,  cosmo)

    # Bundle all differentiable args
    y0_cosmo_ts_solver = ((disp, vel, state), cosmo, ts, solver)

    if adjoint == 'checkpointed':
        return _integrate_checkpointed(y0_cosmo_ts_solver, t0=t0, t1=t1, dt0=dt0)
    elif adjoint == 'reverse':
        return _integrate_reverse_adjoint(y0_cosmo_ts_solver, t0=t0, t1=t1, dt0=dt0)
    else:
        raise ValueError(f"Unknown adjoint type: {adjoint}")


def _integrate_checkpointed(
    y0_cosmo_ts_solver: Tuple,
    *,
    t0: float,
    t1: float,
    dt0: float,
) -> Any:
    """
    Simple forward pass with checkpointing for memory-efficient backprop.

    Uses equinox's checkpointed scan to trade computation for memory during
    automatic differentiation.

    Args:
        y0_cosmo_ts_solver: Bundled differentiable args (y0, cosmo, ts, solver).
        t0: Initial time.
        dt0: Step size.

    Returns:
        Snapshots at each time in ts.
    """
    snapshots, _ = _fwd_loop(y0_cosmo_ts_solver, t0=t0, t1=t1, dt0=dt0, kind='checkpointed')
    return snapshots


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _integrate_reverse_adjoint(
    y0_cosmo_ts_solver: Tuple,
    t0: float,
    t1: float,
    dt0: float,
) -> Any:
    """
    Integration with custom VJP using explicit backward pass.

    The backward pass uses solver.reverse() to step backwards through time,
    accumulating gradients. This requires the solver to implement reverse().

    Args:
        y0_cosmo_ts_solver: Bundled differentiable args (y0, cosmo, ts, solver).
        t0: Initial time.
        dt0: Step size.

    Returns:
        Snapshots at each time in ts.
    """
    snapshots, _ = _fwd_loop(y0_cosmo_ts_solver, t0=t0, t1=t1, dt0=dt0, kind='lax')
    return snapshots


def _integrate_fwd(
    y0_cosmo_ts_solver: Tuple,
    t0: float,
    t1: float,
    dt0: float,
) -> Tuple[Any, Tuple]:
    """
    Forward pass for the custom VJP.

    Executes forward integration and saves residuals for the backward pass.

    Args:
        y0_cosmo_ts_solver: Bundled differentiable args.
        t0: Initial time.
        dt0: Step size.

    Returns:
        Tuple of (snapshots, residuals) where residuals contain state needed for backward.
    """
    snapshots, y_final = _fwd_loop(y0_cosmo_ts_solver, t0=t0, t1=t1, dt0=dt0, kind='lax')
    y0, cosmo, ts, solver = y0_cosmo_ts_solver
    return snapshots, (y_final, cosmo, ts, solver)


def _fwd_loop(y0_cosmo_ts_solver: Tuple[Tuple[ParticleField, ParticleField, NBodyState], Any, jnp.ndarray,
                                        AbstractNBodySolver],
              *,
              t0: float,
              t1: float,
              dt0: float,
              kind: str = 'lax') -> Tuple[Any, Tuple[ParticleField, ParticleField, NBodyState]]:
    """
    Forward integration loop for AbstractNBodySolver.

    The integration process is organized into two nested loops:
      - The inner loop steps forward in increments of dt0 until reaching the current snapshot time.
      - The outer loop iterates over each snapshot time in ts, applying save_at to record the state.

    Args:
        y0_cosmo_ts_solver: A tuple containing:
            - y0: (disp, vel, state) tuple after solver.init() has been called
            - cosmo: Cosmology parameters (unused here, passed through for gradients)
            - ts: Array of snapshot times
            - solver: The AbstractNBodySolver instance
        t0: The starting time for integration.
        dt0: The step size used for forward integration.
        kind: Loop type ('lax' or 'checkpointed').

    Returns:
        A tuple containing:
          - The collection of snapshots obtained via solver.save_at().
          - The final state (disp, vel, state) after integration.
    """
    (disp, vel, state), cosmo, ts, solver = y0_cosmo_ts_solver
    max_steps = int(jnp.ceil((t1 - t0) / dt0)) + 1

    def inner_forward_step(carry):
        """Single integration step."""
        disp_, vel_, state_, t_curr, t_target = carry
        t_next = t_curr + dt0
        t_next = _clip_to_end(t_curr, t_next, t_target)
        disp_next, vel_next, state_next = solver.step(disp_, vel_, t_curr, t_next, dt0, state_, cosmo)
        return (disp_next, vel_next, state_next, t_next, t_target)

    def inner_forward_cond(carry):
        """Continue while t_curr < t_target."""
        _, _, _, t_curr, t_target = carry
        return t_curr < t_target

    def outer_forward_step(outer_carry, t_target):
        """
        Integrate from the current time up to t_target, then save a snapshot.

        Args:
            outer_carry: (disp, vel, state, t_curr)
            t_target: The snapshot time to integrate to.

        Returns:
            Updated outer carry and the snapshot.
        """
        disp_, vel_, state_, t_curr = outer_carry
        inner_carry = (disp_, vel_, state_, t_curr, t_target)

        # Inner loop: step until reaching t_target
        disp_, vel_, state_, _, _ = eqxi.while_loop(inner_forward_cond,
                                                    inner_forward_step,
                                                    inner_carry,
                                                    max_steps=max_steps,
                                                    kind=kind)

        # Save snapshot at t_target
        snapshot = solver.save_at(disp_, vel_, t_target, dt0, state_, cosmo)

        outer_carry = (disp_, vel_, state_, t_target)
        return outer_carry, snapshot

    # Initialize carry
    init_carry = (disp, vel, state, t0)

    # Run outer scan over all snapshot times
    (disp_final, vel_final, state_final,
     _), snapshots = eqxi.scan(outer_forward_step,
                               init_carry,
                               ts,
                               kind=kind,
                               checkpoints=len(ts) if kind == 'checkpointed' else None)

    y_final = (disp_final, vel_final, state_final)
    return snapshots, y_final
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any

def _integrate_bwd(
    t0: float,
    t1: float,
    dt0: float,
    residuals: Tuple,
    cotangents: Any,
) -> Tuple[Tuple]:
    """
    Backward pass for the custom VJP, computing adjoint sensitivities.
    """
    y_final, cosmo, ts, solver = residuals
    ys_ct = cotangents

    # Partition differentiable and non-differentiable parts
    diff_cosmo, nondiff_cosmo = eqx.partition(cosmo, eqx.is_inexact_array_like)
    diff_solver, nondiff_solver = eqx.partition(solver, eqx.is_inexact_array_like)

    disp_final, vel_final, state_final = y_final
    diff_state_final, nondiff_state_final = eqx.partition(state_final, eqx.is_inexact_array_like)

    # Initialize zero adjoints
    adj_disp = jax.tree.map(jnp.zeros_like, disp_final)
    adj_vel = jax.tree.map(jnp.zeros_like, vel_final)
    adj_state = jax.tree.map(jnp.zeros_like, diff_state_final)
    adj_cosmo = jax.tree.map(jnp.zeros_like, diff_cosmo)
    adj_solver = jax.tree.map(jnp.zeros_like, diff_solver)
    adj_ts_scalar = jnp.zeros_like(ts[0])

    def inner_backward_step(carry):
        (disp, vel, state, diff_cosmo_, diff_solver_, adj_disp_, adj_vel_, adj_state_, 
         adj_cosmo_, adj_solver_, t0_, tc) = carry

        t_prev = tc - dt0
        t_prev = _clip_to_start(t_prev, tc, t0_)

        # Reconstruct full objects for reverse
        solver_ = eqx.combine(diff_solver_, nondiff_solver)
        cosmo_ = eqx.combine(diff_cosmo_, nondiff_cosmo)

        # Reverse the forward step to get previous state
        disp_prev, vel_prev, state_prev = solver_.reverse(disp, vel, t_prev, tc, dt0, state, cosmo_)
        diff_state_prev, nondiff_state_prev = eqx.partition(state_prev, eqx.is_inexact_array_like)

        # VJP for the step function
        def _to_vjp_step(disp_in, vel_in, diff_cosmo_in, diff_solver_in, diff_state_in):
            solver_in = eqx.combine(diff_solver_in, nondiff_solver)
            cosmo_in = eqx.combine(diff_cosmo_in, nondiff_cosmo)
            # IMPORTANT: Use the nondiff part from the PREVIOUS state (reconstructed above)
            state_in = eqx.combine(diff_state_in, nondiff_state_prev) 
            disp_out, vel_out, state_out = solver_in.step(disp_in, vel_in, t_prev, tc, dt0, state_in, cosmo_in)
            state_out , _ = eqx.partition(state_out, eqx.is_inexact_array_like)
            return (disp_out, vel_out, state_out)

        _, f_vjp_step = jax.vjp(_to_vjp_step, disp_prev, vel_prev, diff_cosmo_, diff_solver_, diff_state_prev)
        
        # Propagate adjoints backwards
        new_adj_disp, new_adj_vel, new_adj_cosmo_step, new_adj_solver_step, new_adj_state_step = f_vjp_step(
            (adj_disp_, adj_vel_, adj_state_)
        )

        # Accumulate parameter adjoints
        adj_cosmo_new = jax.tree.map(jnp.add, adj_cosmo_, new_adj_cosmo_step)
        adj_solver_new = jax.tree.map(jnp.add, adj_solver_, new_adj_solver_step)

        # FIX: Return new_adj_state_step, not adj_state_
        return (disp_prev, vel_prev, state_prev, diff_cosmo_, diff_solver_, 
                new_adj_disp, new_adj_vel, new_adj_state_step, 
                adj_cosmo_new, adj_solver_new, t0_, t_prev)

    def inner_backward_cond(carry):
        *_, t0_, tc = carry
        return tc > t0_

    def outer_backward_step(outer_carry, vals):
        y_ct, t0_prev_snap = vals
        (disp, vel, state, diff_cosmo_, diff_solver_, adj_disp_, adj_vel_, adj_state_, adj_cosmo_, adj_solver_,
         adj_ts_scalar_, tc) = outer_carry

        solver_ = eqx.combine(diff_solver_, nondiff_solver)
        cosmo_ = eqx.combine(diff_cosmo_, nondiff_cosmo)
        diff_state_, nondiff_state_ = eqx.partition(state, eqx.is_inexact_array_like)

        # VJP of the snapshot function
        def _to_vjp_save(disp_in, vel_in, diff_state_in, diff_cosmo_in, diff_solver_in, tc_in):
            s_in = eqx.combine(diff_solver_in, nondiff_solver)
            c_in = eqx.combine(diff_cosmo_in, nondiff_cosmo)
            s_in_state = eqx.combine(diff_state_in, nondiff_state_)
            return s_in.save_at(disp_in, vel_in, tc_in, dt0, s_in_state, c_in)

        _, f_vjp_save = jax.vjp(_to_vjp_save, disp, vel, diff_state_, diff_cosmo_, diff_solver_, tc)
        save_adj_disp, save_adj_vel, save_adj_state, save_adj_cosmo, save_adj_solver, adj_t_save = f_vjp_save(y_ct)

        # ACCUMULATE: Add snapshot contributions to current adjoint carry
        adj_disp_sum = jax.tree.map(jnp.add, adj_disp_, save_adj_disp)
        adj_vel_sum = jax.tree.map(jnp.add, adj_vel_, save_adj_vel)
        adj_state_sum = jax.tree.map(jnp.add, adj_state_, save_adj_state)
        adj_cosmo_sum = jax.tree.map(jnp.add, adj_cosmo_, save_adj_cosmo)
        adj_solver_sum = jax.tree.map(jnp.add, adj_solver_, save_adj_solver)

        # Reverse to find the linearization point for the step VJP
        t_prev = tc - dt0
        t_prev = _clip_to_start(t_prev, tc, t0_prev_snap)
        disp_prev, vel_prev, state_prev = solver_.reverse(disp, vel, t_prev, tc, dt0, state, cosmo_)
        
        diff_state_prev, nondiff_state_prev = eqx.partition(state_prev, eqx.is_inexact_array_like)

        # VJP of the forward step (includes time gradient)
        def _to_vjp_step_full(disp_in, vel_in, diff_state_in, diff_cosmo_in, diff_solver_in, tc_in):
            tp_local = tc_in - dt0
            tp_local = _clip_to_start(tp_local, tc_in, t0_prev_snap)
            s_in = eqx.combine(diff_solver_in, nondiff_solver)
            c_in = eqx.combine(diff_cosmo_in, nondiff_cosmo)
            s_in_state_prev = eqx.combine(diff_state_in, nondiff_state_prev)
            
            disp_out, vel_out, state_out = s_in.step(disp_in, vel_in, tp_local, tc_in, dt0, s_in_state_prev, c_in)
            diff_state_out = eqx.filter(state_out, eqx.is_inexact_array_like)
            return disp_out, vel_out, diff_state_out

        _, f_vjp_step = jax.vjp(_to_vjp_step_full, disp_prev, vel_prev, diff_state_prev, diff_cosmo_, diff_solver_, tc)

        # Pull the SUMMED adjoints through the step VJP
        (step_adj_disp, step_adj_vel, step_adj_state, step_adj_cosmo, step_adj_solver, step_adj_ts) = f_vjp_step(
            (adj_disp_sum, adj_vel_sum, adj_state_sum))

        # Time gradient logic
        step_adj_ts = jnp.where(tc == t_prev, jnp.zeros_like(step_adj_ts), step_adj_ts)
        f_adj_ts = adj_t_save + step_adj_ts
        adj_ts_out = f_adj_ts - adj_ts_scalar_

        # Update global param adjoints
        adj_cosmo_final = jax.tree.map(jnp.add, adj_cosmo_sum, step_adj_cosmo)
        adj_solver_final = jax.tree.map(jnp.add, adj_solver_sum, step_adj_solver)

        # Run inner backward loop
        inner_carry = (disp_prev, vel_prev, state_prev, diff_cosmo_, diff_solver_, 
                       step_adj_disp, step_adj_vel, step_adj_state, # Use step_adj_state as seed
                       adj_cosmo_final, adj_solver_final, t0_prev_snap, t_prev)

        (disp_out, vel_out, state_out, _, _, adj_disp_out, adj_vel_out, adj_state_out, 
         adj_cosmo_out, adj_solver_out, _, t_out) = jax.lax.while_loop(inner_backward_cond, inner_backward_step, inner_carry)

        outer_carry = (disp_out, vel_out, state_out, diff_cosmo_, diff_solver_, adj_disp_out, adj_vel_out,
                       adj_state_out, adj_cosmo_out, adj_solver_out, step_adj_ts, t_out)

        return outer_carry, adj_ts_out

    # Set up reverse scan
    t1_ = ts[-1]
    t_steps = jnp.concatenate((jnp.asarray([t0]), ts[:-1]))
    
    init_carry = (disp_final, vel_final, state_final, diff_cosmo, diff_solver, adj_disp, adj_vel, adj_state, adj_cosmo,
                  adj_solver, adj_ts_scalar, t1_)

    vals = (ys_ct, t_steps)

    (_, _, _, _, _, adj_disp_final, adj_vel_final, adj_state_final, adj_cosmo_final, adj_solver_final, _,
     _), adj_ts_contributions = jax.lax.scan(outer_backward_step, init_carry, vals, reverse=True)

    adj_cosmo_full = eqx.combine(adj_cosmo_final, nondiff_cosmo)
    adj_solver_full = eqx.combine(adj_solver_final, nondiff_solver)
    adj_y0 = (adj_disp_final, adj_vel_final, adj_state_final)

    return ((adj_y0, adj_cosmo_full, adj_ts_contributions, adj_solver_full), )


_integrate_reverse_adjoint.defvjp(_integrate_fwd, _integrate_bwd)
