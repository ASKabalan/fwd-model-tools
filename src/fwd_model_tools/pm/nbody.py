from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import FieldStatus, ParticleField
from ..fields.painting import PaintingOptions
from ..utils import distances
from ._resolve_geometry import resolve_ts_geometry
from .correction import NoCorrection
from .integrate import AdjointType, integrate
from .interp import NoInterp
from .solvers import AbstractNBodySolver, EfficientDriftDoubleKick

__all__ = ["nbody"]


def _validate_t0_cb(lpt_t0, t0):
    lpt_t0 = jnp.atleast_1d(lpt_t0).squeeze()
    if not jnp.isscalar(t0) or not jnp.isscalar(lpt_t0):
        raise ValueError("Starting scale factor t0 and LPT fields' scale factor must be scalars.")
    if lpt_t0 != t0:
        raise ValueError(f"Starting scale factor t0={t0} does not match LPT fields' scale factor {lpt_t0}.")


@partial(jax.jit, static_argnames=["t0", "t1", "dt0", "nb_shells", "adjoint", "checkpoints"])
def nbody(
    cosmo,
    dx_field: ParticleField,
    p_field: ParticleField,
    *,
    t0: float = 0.1,
    t1: float = 1.0,
    dt0: float = 0.05,
    ts: jnp.ndarray | None = None,
    nb_shells: int | None = None,
    density_widths: float | jnp.ndarray | None = None,
    solver: AbstractNBodySolver = EfficientDriftDoubleKick(interp_kernel=NoInterp(painting=PaintingOptions(
        target="particles"))),
    adjoint: AdjointType = "checkpointed",
    checkpoints: int | None = None,
) -> jax.Array:
    """
    Evolve particles forward in time and save lightcone density planes.

    Takes LPT displacements and momenta, runs N-body integration using
    a symplectic solver, and returns density planes at shell centers.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for background expansion.
    dx_field : ParticleField
        Displacement field from LPT.
    p_field : ParticleField
        Momentum field from LPT.
    t1 : float, default=1.0
        Final scale factor. Used as the snapshot target if *ts* and *nb_shells* are None.
    dt0 : float, default=0.05
        Integration time step.
    ts : jnp.ndarray or None, optional
        Scale factor specification.  Mutually exclusive with *nb_shells*.
        Accepts scalar, 1-D array (shell centres), or 2-D ``(2, N)``
        (near/far per shell).
    nb_shells : int or None, optional
        Number of shells (alternative to *ts*).
    density_width : float or array, optional
        Override shell widths.
    solver : AbstractNBodySolver or None, optional
        Solver instance. If None, uses EfficientDriftDoubleKick with NoInterp/NoCorrection.
    adjoint : AdjointType, default='checkpointed'
        Adjoint mode: 'checkpointed' or 'reverse'.

    Returns
    -------
    Field
        Lightcone as a stacked Field PyTree.

    Raises
    ------
    ValueError
        If fields don't match.

    Examples
    --------
    >>> cosmo = Planck18()
    >>> dx, p = lpt(cosmo, initial_field, ts=0.1, order=1)
    >>> lightcone = nbody(cosmo, dx, p, t1=1.0, nb_shells=10)
    """

    assert dx_field.status == FieldStatus.LPT1 or dx_field.status == FieldStatus.LPT2, (
        "dx_field must have status FieldStatus.LPT1 or FieldStatus.LPT2.")
    assert p_field.status == FieldStatus.LPT1 or p_field.status == FieldStatus.LPT2, (
        "p_field must have status FieldStatus.LPT1 or FieldStatus.LPT2.")

    if dx_field.mesh_size != p_field.mesh_size:
        raise ValueError("dx_field and p_field must have matching mesh_size")
    if dx_field.box_size != p_field.box_size:
        raise ValueError("dx_field and p_field must have matching box_size")

    # Check that t0 matches the LPT fields' scale factor
    jax.debug.callback(_validate_t0_cb, dx_field.scale_factors, t0)

    if ts is None and nb_shells is None:
        # Snapshot mode: single output at t1
        ts = jnp.array([t1])

    # Resolve lightcone geometry via shared helper
    ts_resolved, r_centers, density_plane_width, _ = resolve_ts_geometry(cosmo,
                                                                         dx_field,
                                                                         painting=solver.interp_kernel.painting,
                                                                         ts=ts,
                                                                         nb_shells=nb_shells,
                                                                         density_widths=density_widths)

    max_comoving_distance = dx_field.max_comoving_radius

    # Update solver's interp_kernel with geometry
    updated_interp_kernel = solver.interp_kernel.update_geometry(
        ts=ts_resolved,
        r_centers=r_centers,
        density_widths=density_plane_width,
        max_comoving_distance=max_comoving_distance,
    )
    solver = eqx.tree_at(lambda s: s.interp_kernel, solver, updated_interp_kernel)

    # Run integration
    lightcone = integrate(
        displacements=dx_field,
        velocities=p_field,
        cosmo=cosmo,
        ts=ts_resolved,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        adjoint=adjoint,
        checkpoints=checkpoints,
    )

    if ts_resolved.ndim == 1 and ts_resolved.size == 1:
        # Add singleton plane dimension for consistent output shape
        lightcone = lightcone.apply_fn(lambda x: x.squeeze(axis=0))
        r_centers = r_centers.squeeze(axis=0)
        density_plane_width = density_plane_width.squeeze(axis=0)

    # Reverse to get near-to-far ordering
    if lightcone.is_batched():
        lightcone = lightcone[::-1]
        r_centers = r_centers[::-1]
        density_plane_width = density_plane_width[::-1]

    # Set metadata
    scale_factors = lightcone.scale_factors
    z_sources = jc.utils.a2z(scale_factors)
    lightcone = lightcone.replace(z_sources=z_sources,
                                  comoving_centers=r_centers,
                                  density_width=density_plane_width,
                                  status=FieldStatus.LIGHTCONE)

    return lightcone
