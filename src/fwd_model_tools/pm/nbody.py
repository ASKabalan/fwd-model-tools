from __future__ import annotations

from functools import partial
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import FieldStatus, ParticleField
from ..fields.painting import PaintingOptions
from ..utils import compute_lightcone_shells
from .correction import NoCorrection
from .integrate import AdjointType, integrate
from .interp import NoInterp
from .solvers import AbstractNBodySolver, EfficientDriftDoubleKick

__all__ = ["nbody"]


def _default_solver() -> EfficientDriftDoubleKick:
    """Create default solver with NoInterp and NoCorrection."""
    return EfficientDriftDoubleKick(
        pgd_kernel=NoCorrection(),
        interp_kernel=NoInterp(painting=PaintingOptions(target="particles")),
    )


@partial(jax.jit, static_argnames=["t1", "dt0", "adjoint"])
def nbody(cosmo,
          dx_field: ParticleField,
          p_field: ParticleField,
          t1: float = 1.0,
          dt0: float = 0.05,
          ts: Union[jnp.ndarray, None] = None,
          nb_shells: Union[int, None] = None,
          solver: Union[AbstractNBodySolver, None] = None,
          adjoint: AdjointType = 'checkpointed') -> jax.Array:
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
        Final scale factor.
    dt0 : float, default=0.05
        Integration time step.
    ts : jnp.ndarray | None
        Explicit snapshot times. Either ts or nb_shells must be provided.
    nb_shells : int | None
        Number of shells (alternative to ts).
    solver : AbstractNBodySolver | None
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
        If fields don't match or neither ts nor nb_shells is provided.

    Examples
    --------
    >>> cosmo = Planck18()
    >>> dx, p = lpt(cosmo, initial_field, a=0.1, order=1)
    >>> lightcone = nbody(cosmo, dx, p, t1=1.0)
    >>> lightcone.array.shape
    >>> lightcone.scale_factors.shape
    """
    # Validate inputs
    if ts is None and nb_shells is None:
        raise ValueError("Either ts or nb_shells must be provided.")

    # Create default solver if not provided
    if solver is None:
        solver = _default_solver()

    assert dx_field.status == FieldStatus.LPT1 or dx_field.status == FieldStatus.LPT2, (
        "dx_field must have status FieldStatus.LPT1 or FieldStatus.LPT2.")
    assert p_field.status == FieldStatus.LPT1 or p_field.status == FieldStatus.LPT2, (
        "p_field must have status FieldStatus.LPT1 or FieldStatus.LPT2.")

    if dx_field.mesh_size != p_field.mesh_size:
        raise ValueError("dx_field and p_field must have matching mesh_size")
    if dx_field.box_size != p_field.box_size:
        raise ValueError("dx_field and p_field must have matching box_size")

    # Extract t0 from fields
    t0 = jnp.atleast_1d(dx_field.scale_factors).squeeze()
    if not jnp.isscalar(t0):
        raise ValueError("Starting scale factor t0 must be a scalar.")

    # Compute lightcone geometry
    if ts is None:
        r_centers, ts = compute_lightcone_shells(cosmo, dx_field, nb_shells=nb_shells)
    else:
        assert jnp.atleast_1d(ts).size >= 2, "ts must have at least two entries for width computation"
        r_centers = jc.background.radial_comoving_distance(cosmo, ts)

    # Compute density widths
    inner_edges = 0.5 * (r_centers[1:] + r_centers[:-1])
    start_edge = r_centers[0] - (inner_edges[0] - r_centers[0])
    end_edge = r_centers[-1] + (r_centers[-1] - inner_edges[-1])
    r_edges = jnp.concatenate([jnp.array([start_edge]), inner_edges, jnp.array([end_edge])])
    density_plane_width = r_edges[:-1] - r_edges[1:]

    max_comoving_distance = dx_field.max_comoving_radius

    # Update solver's interp_kernel with geometry
    updated_interp_kernel = solver.interp_kernel.update_geometry(
        ts=ts,
        r_centers=r_centers,
        density_widths=density_plane_width,
        max_comoving_distance=max_comoving_distance,
    )
    solver = eqx.tree_at(lambda s: s.interp_kernel, solver, updated_interp_kernel)

    # Run integration
    y0 = (dx_field, p_field)
    lightcone = integrate(y0=y0, cosmo=cosmo, ts=ts, solver=solver, t0=t0, t1=t1, dt0=dt0, adjoint=adjoint)

    # Reverse to get near-to-far ordering
    lightcone = lightcone[::-1]

    # Set metadata
    scale_factors = lightcone.scale_factors
    z_sources = jc.utils.a2z(scale_factors)
    lightcone = lightcone.replace(z_sources=z_sources, comoving_centers=r_centers, status=FieldStatus.LIGHTCONE)

    return lightcone
