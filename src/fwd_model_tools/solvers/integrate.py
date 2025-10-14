from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractSolver, ODETerm, SaveAt
from jax._src.numpy.util import promote_dtypes_inexact


def handle_saveat(save_at: SaveAt, t0: float, t1: float) -> SaveAt:
    """
    Prepares a SaveAt configuration for ODE integration by ensuring that the snapshot times array (ts)
    is a valid JAX array and conditionally includes the start (t0) and/or end (t1) times.

    This function performs the following steps:
      - Verifies that at least one of t0, t1, or ts is specified.
      - Replaces a None value in ts with an empty JAX array to enable safe concatenation.
      - Prepends t0 to ts if save_at.subs.t0 is True.
      - Appends t1 to ts if save_at.subs.t1 is True.

    Args:
        save_at: A diffrax SaveAt instance with optional flags (subs.t0, subs.t1) and an optional ts array.
        t0: The initial time of the integration interval.
        t1: The final time of the integration interval.

    Returns:
        A modified SaveAt object where the ts attribute is a valid JAX array, potentially updated to include t0 and/or t1.
    """
    assert save_at.subs is not None, "You must set at least one of t0, t1, or ts."

    def _where_subs_ts(s):
        return s.subs.ts

    # Replace `None` with an empty array to make concatenation valid
    if save_at.subs.ts is None:
        save_at = eqx.tree_at(_where_subs_ts, save_at, replace=jnp.array([]))

    # If both t0 and t1 are True, prepend t0 and append t1
    if save_at.subs.t0 and save_at.subs.t1:
        save_at = eqx.tree_at(
            _where_subs_ts,
            save_at,
            replace_fn=lambda x: jnp.concatenate((jnp.array([t0]), x, jnp.array([t1]))),
        )
    # If only t0 is True, prepend t0
    elif save_at.subs.t0:
        save_at = eqx.tree_at(
            _where_subs_ts,
            save_at,
            replace_fn=lambda x: jnp.concatenate((jnp.array([t0]), x)),
        )
    # If only t1 is True, append t1
    elif save_at.subs.t1:
        save_at = eqx.tree_at(
            _where_subs_ts,
            save_at,
            replace_fn=lambda x: jnp.concatenate((x, jnp.array([t1]))),
        )

    return save_at


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


def integrate(
    terms: tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    y0: Any,
    args: Any,
    saveat: SaveAt | None = SaveAt(t1=True),
) -> Any:
    """
    Integrates an ODE system from time t0 to t1 using a specified solver, returning solution snapshots
    at times defined by the saveat configuration.

    This function leverages diffrax's solver capabilities and processes the snapshot schedule provided by saveat.
    It also promotes the snapshot times array to an inexact data type for compatibility.

    Args:
        terms: A tuple of ODETerm instances that describe the system dynamics.
        solver: A diffrax.AbstractSolver instance specifying the integration method.
        t0: The starting time for integration.
        t1: The ending time for integration.
        dt0: The step size for each integration increment.
        y0: The initial state of the system (scalar, array, or PyTree).
        args: Additional parameters for the ODE system (any PyTree).
        saveat: A diffrax.SaveAt object specifying when and how to save solution snapshots.
                Defaults to saving at the final time.

    Returns:
        The computed solution snapshots at the times specified by the saveat configuration.
    """
    saveat = handle_saveat(saveat, t0, t1)
    save_y = saveat.subs.fn
    ts = saveat.subs.ts
    (ts,) = promote_dtypes_inexact(ts)
    y0_args_ts = (y0, args, ts)
    return integrate_impl(
        y0_args_ts, terms=terms, solver=solver, t0=t0, t1=t1, dt0=dt0, save_y=save_y
    )


def _fwd_loop(
    y0_args_ts: Any,
    *,
    terms: tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
) -> Any:
    """
    Executes the forward integration loop for an ODE system and collects snapshots at specified times.

    The integration process is organized into two nested loops:
      - The inner loop steps forward in uniform increments of dt0 until reaching the current snapshot time.
      - The outer loop iterates over each snapshot time in ts, applying the snapshot function (save_y)
        to record the system state.

    Args:
        y0_args_ts: A tuple containing the initial state (y0), ODE parameters (args), and snapshot times (ts).
        terms: A tuple of ODETerm instances defining the system dynamics.
        solver: The solver instance responsible for performing each integration step.
        t0: The starting time for integration.
        t1: The final time for integration.
        dt0: The step size used for forward integration.
        save_y: A user-defined function to process and record the state at snapshot times.

    Returns:
        A tuple containing:
          - The collection of snapshots obtained via save_y.
          - The final state achieved after integration.
    """
    y0, args, ts = y0_args_ts
    args = jax.tree.map(jnp.asarray, args)

    def inner_forward_step(carry):
        y, args_, tc, t1 = carry
        t_next = tc + dt0
        t_next = _clip_to_end(tc, t_next, t1)
        # The solver call returns (y_next, solver_state, new_t, result, made_jump)
        y_next, _, _, _, _ = solver.step(
            terms, tc, t_next, y, args_, solver_state=None, made_jump=False
        )
        return (y_next, args_, t_next, t1)

    def inner_forward_cond(carry):
        # Continue the while_loop while the current time is less than the final time
        _, _, tc, t1 = carry
        return tc < t1

    def outer_forward_step(outer_carry, t1):
        """
        For each designated snapshot time, integrate from the current time up to that time,
        then apply the snapshot function (save_y) to record the system state.

        Args:
            outer_carry: A tuple containing the current state, parameters, and time.
            t1: The snapshot time at which to record the state.

        Returns:
            A tuple where:
              - The first element is the updated outer carry.
              - The second element is the snapshot computed via save_y.
        """
        y, args_, t0 = outer_carry
        inner_carry = (y, args_, t0, t1)
        y, _, _, _ = jax.lax.while_loop(inner_forward_cond, inner_forward_step, inner_carry)

        outer_carry = (y, args_, t1)
        # Apply the user-defined function at this "snapshot" time
        return outer_carry, save_y(t1, y, args_)

    # Initialize carry
    init_carry = (y0, args, t0)

    # The outer scan runs over each requested snapshot time
    (y_final, _, _), ys_final = jax.lax.scan(outer_forward_step, init_carry, ts)

    # Return snapshots plus final state+args
    return ys_final, y_final


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def integrate_impl(
    y0_args_ts: Any,
    terms: tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
) -> Any:
    """
    Core implementation of the ODE integration routine that wraps the forward integration loop
    and supports custom vector-Jacobian product (VJP) rules for automatic differentiation.

    Args:
        y0_args_ts: A tuple containing the initial state, ODE parameters, and snapshot times.
        terms: A tuple of ODETerm instances defining the system dynamics.
        solver: The solver instance used to perform the integration.
        t0: The starting time for integration.
        t1: The final time for integration.
        dt0: The integration step size.
        save_y: A function to process and record the state at designated snapshot times.

    Returns:
        The collection of solution snapshots obtained from the forward integration loop.
    """
    ys_final, _ = _fwd_loop(
        y0_args_ts, terms=terms, solver=solver, t0=t0, t1=t1, dt0=dt0, save_y=save_y
    )
    return ys_final


def integrate_fwd(
    y0_args_ts: Any,
    terms: tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
) -> tuple[Any, tuple[Any, Any]]:
    """
    Forward pass for the custom VJP (vector-Jacobian product) of the integration process.

    This function executes the forward integration loop, returning both the computed snapshots
    and residuals required for the backward (adjoint) pass.

    Args:
        y0_args_ts: A tuple containing the initial state, parameters, and snapshot times.
        terms: A tuple of ODETerm instances representing the system dynamics.
        solver: The solver used to advance the integration.
        t0: The initial time for integration.
        t1: The final time for integration.
        dt0: The step size used for the integration.
        save_y: A function that records snapshots at specified times.

    Returns:
        A tuple containing:
          - The snapshots from the forward integration.
          - A residual tuple (final state, parameters, and snapshot times) for use in the backward pass.
    """
    ys_final, y_final = _fwd_loop(
        y0_args_ts, terms=terms, solver=solver, t0=t0, t1=t1, dt0=dt0, save_y=save_y
    )
    _, args, ts = y0_args_ts
    return ys_final, (y_final, args, ts)


def integrate_bwd(
    terms: tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
    residuals: tuple[Any, tuple[Any, Any]],
    cotangents: Any,
) -> tuple[Any, Any]:
    """
    Backward pass for the custom VJP of the integration routine, computing the adjoint sensitivities.

    This function reverses the forward integration process to compute gradients with respect to the
    initial state and parameters. It carefully accounts for the contributions of snapshots and
    accumulates the adjoint values through a reverse scan.

    Args:
        terms: A tuple of ODETerm instances defining the system dynamics.
        solver: The solver instance used for the integration.
        t0: The initial time for integration.
        t1: The final time for integration.
        dt0: The step size used during integration.
        save_y: A function that records the state at snapshot times.
        residuals: A tuple containing the final state, parameters, and snapshot times from the forward pass.
        cotangents: The gradient contributions (cotangents) corresponding to the forward pass snapshots.

    Returns:
        A tuple of gradients with respect to the initial state and parameters, formatted to match the custom VJP signature.
    """
    y_final, args, ts = residuals
    ys_ct = cotangents  # Gradient w.r.t. the forward pass snapshots

    # Initialize adjoint for args and final y
    args = jax.tree.map(jnp.asarray, args)

    diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array_like)

    adj_y = jax.tree.map(lambda x: jnp.zeros_like(x), y_final)
    adj_args = jax.tree.map(lambda x: jnp.zeros_like(x), diff_args)
    adj_ts = jax.tree.map(lambda x: jnp.zeros_like(x), ts[0])

    def inner_backward_step(carry):
        """
        Reverses a single forward integration step from time tc down to t_prev and updates adjoint values.

        Args:
            carry: A tuple containing the current state, differentiable parameters,
                   current adjoints for state and parameters, the lower bound time t0_,
                   and the current time tc.

        Returns:
            An updated carry tuple with the state and adjoints computed at the new time step.
        """
        y, diff_args, adj_y, adj_args, t0_, tc = carry

        t_prev = tc - dt0
        t_prev = _clip_to_start(t_prev, tc, t0_)
        # Reverse the forward step
        y_prev = solver.reverse(terms, t_prev, tc, y, args, solver_state=None, made_jump=False)

        # Differentiate with respect to the "forward step" to obtain partial derivatives.
        def _to_vjp(y, diff_args):
            args_ = eqx.combine(diff_args, nondiff_args)
            y_next, _, _, _, _ = solver.step(
                terms, t_prev, tc, y, args_, solver_state=None, made_jump=False
            )
            return y_next

        _, f_vjp = jax.vjp(_to_vjp, y_prev, diff_args)
        adj_y, new_adj_args = f_vjp(adj_y)

        # Accumulate these into the existing adjoints.
        new_d_args = jax.tree.map(jnp.add, new_adj_args, adj_args)

        return (y_prev, diff_args, adj_y, new_d_args, t0_, t_prev)

    def inner_backward_cond(carry):
        """
        Determines whether to continue the backward integration loop by checking if the current time is above t0.

        Args:
            carry: A tuple containing the current state and time information.

        Returns:
            A boolean indicating if the backward loop should continue.
        """
        _, _, _, _, t0_, tc = carry
        return tc > t0_

    def outer_backward_step(outer_carry, vals):
        """
        Processes each snapshot in reverse order, accumulating gradient contributions and stepping backward.

        For each snapshot, this function adds the snapshot cotangent to the current adjoint,
        performs a backward integration step, and differentiates the snapshot function to propagate gradients.

        Args:
            outer_carry: A tuple containing the current state, differentiable parameters,
                         current adjoints, snapshot adjoint, and current time.
            vals: A tuple pairing the snapshot cotangent with its corresponding time value.

        Returns:
            A tuple where:
              - The first element is the updated outer carry.
              - The second element is the computed gradient contribution for the snapshot.
        """
        y_ct, t0_ = vals
        y, diff_args, adj_y, adj_args, adj_ts, tc = outer_carry

        t_prev = tc - dt0
        t_prev = _clip_to_start(t_prev, tc, t0_)

        # Reverse the forward step
        y_prev = solver.reverse(terms, t_prev, tc, y, args, solver_state=None, made_jump=False)

        # Differentiate the "save function" at snapshot time tc:
        def _to_vjp_snap(tc_, y_, diff_args_):
            args_ = eqx.combine(diff_args_, nondiff_args)
            return save_y(tc_, y_, args_)

        def _to_vjp_step(tc_, y, diff_args):
            t_prev = tc - dt0
            t_prev = _clip_to_start(t_prev, tc, t0_)
            args_ = eqx.combine(diff_args, nondiff_args)
            y_next, _, _, _, _ = solver.step(
                terms, t_prev, tc_, y, args_, solver_state=None, made_jump=False
            )
            return y_next

        # Compute the adjoints with respect to the snapshot function
        _, f_vjp_snap = jax.vjp(_to_vjp_snap, tc, y, diff_args)
        snap_adj_ts, new_adj_y, new_adj_args = f_vjp_snap(y_ct)
        # Accumulate the adjoint for the snapshot time
        adj_y = jax.tree.map(jnp.add, adj_y, new_adj_y)
        adj_args = jax.tree.map(jnp.add, adj_args, new_adj_args)

        # Compute the adjoints with respect to the forward step
        _, f_vjp_step = jax.vjp(_to_vjp_step, tc, y_prev, diff_args)
        step_adj_ts, adj_y, new_adj_args = f_vjp_step(adj_y)
        # If we are at the initial time, set the gradient w.r.t. the time step to zero
        step_adj_ts = jnp.where(tc == t_prev, jnp.zeros_like(step_adj_ts), step_adj_ts)
        # Accumulate the adjoint for the forward step
        adj_args = jax.tree.map(jnp.add, adj_args, new_adj_args)
        f_adj_ts = jax.tree.map(jnp.add, snap_adj_ts, step_adj_ts)

        inner_carry = (y_prev, diff_args, adj_y, adj_args, t0_, t_prev)
        y_prev, diff_args, adj_y, adj_args, tc, _ = jax.lax.while_loop(
            inner_backward_cond, inner_backward_step, inner_carry
        )

        # The gradient w.r.t. the snapshot time is the sum of adjoints at this time minus the adjoint at the previous step
        adj_subs = jax.tree.map(jnp.subtract, f_adj_ts, adj_ts)

        outer_carry = (y_prev, diff_args, adj_y, adj_args, step_adj_ts, tc)
        return outer_carry, adj_subs

    # Reverse through the snapshot times

    # Define t1 as the last snapshot if available
    t1_ = ts[-1]

    # Shift the array of snapshot times to incorporate the initial time
    t_steps = jnp.concatenate((jnp.asarray([t0]), ts[:-1]))

    # Initial carry is the final state and final adjoint
    init_carry = (y_final, diff_args, adj_y, adj_args, adj_ts, t1_)

    # Pair the cotangents with the corresponding snapshot times
    vals = (ys_ct, t_steps)
    # Perform the reverse scan over the snapshots
    (_, _, adj_y, adj_args, _, _), adj_ts = jax.lax.scan(
        outer_backward_step, init_carry, vals, reverse=True
    )
    zero_nondiff = jax.tree.map(jnp.zeros_like, nondiff_args)
    adj_args = eqx.combine(adj_args, zero_nondiff)

    # Return the adjoints for the initial state and parameters (others are placeholders).
    return ((adj_y, adj_args, adj_ts),)


integrate_impl.defvjp(integrate_fwd, integrate_bwd)


def scan_integrate(
    terms: tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    y0: Any,
    args: Any,
    saveat: SaveAt | None = None,
) -> Any:
    """
    Integrates an ODE system using a scanning approach with uniform time steps and extracts snapshots
    at specified times.

    This "vanilla" integrator advances the solution from t0 to t1 in fixed increments of dt0,
    storing the full state at each step. The final snapshots are then extracted at times defined by
    the saveat configuration. This method is particularly useful for debugging or comparison with the
    main integrator, though it may consume significant memory for fine-grained or long-duration integrations.

    Args:
        y0: The initial state of the system.
        args: Parameters for the ODE system.
        terms: A tuple of ODETerm instances defining the system's dynamics.
        solver: The solver instance providing the .step method for integration.
        t0: The initial time for integration.
        t1: The final time for integration.
        dt0: The fixed step size for the forward integration.
        saveat: (Optional) A SaveAt object specifying the times at which to save snapshots.

    Returns:
        A PyTree containing the solution snapshots at the times specified in the saveat configuration.
    """
    saveat = handle_saveat(saveat, t0, t1)
    save_y = saveat.subs.fn
    ts = saveat.subs.ts
    (ts,) = promote_dtypes_inexact(ts)
    y0_args_ts = (y0, args, ts)
    ys_final, _ = _fwd_loop(
        y0_args_ts, terms=terms, solver=solver, t0=t0, t1=t1, dt0=dt0, save_y=save_y
    )
    return ys_final
