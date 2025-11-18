from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from diffrax import ODETerm, RecursiveCheckpointAdjoint, SaveAt, diffeqsolve

from ..fields import DensityField, FieldStatus, ParticleField
from ..solvers.integrate import integrate as reverse_adjoint_integrate
from ..solvers.ode import symplectic_ode
from ..solvers.reversible_efficient_fastpm import ReversibleEfficientFastPM
from ..utils import compute_snapshot_scale_factors

__all__ = ["nbody"]

@partial(jax.jit, static_argnames=['t1', 'dt0', 'nb_shells', 'geometry', 'solver', 'adjoint'])
def nbody(
    cosmo,
    dx_field: ParticleField,
    p_field: ParticleField,
    t1: float = 1.0,
    dt0: float = 0.05,
    ts: jnp.ndarray | None = None,
    nb_shells: int | None = None,
    geometry: str = "spherical",
    solver=ReversibleEfficientFastPM(),
    adjoint: str | object = RecursiveCheckpointAdjoint(),
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
        Final scale factor (present day).
    dt0 : float, default=0.05
        Integration time step.
    geometry : str, default="spherical"
        Output type: "spherical" (HEALPix), "flat" (Cartesian),
        "density" (3D density fields), or "particles" (particle displacements).
    density_plane_npix : int, default=256
        Output resolution for flat geometry. Ignored for spherical.
    solver : AbstractSolver, default=ReversibleEfficientFastPM()
        Diffrax solver for integration.
    adjoint : str | object, default=RecursiveCheckpointAdjoint()
        Adjoint for gradients. If "reverse_adjoint", uses custom reverse
        adjoint from solvers/integrate.py. Otherwise, uses diffrax adjoint.

    Returns
    -------
    ParticleField, DensityField, FlatDensity, or SphericalDensity
        Lightcone as a Field PyTree, stacked across shells:
        - geometry="spherical": SphericalDensity with shape (nb_shells, npix)
        - geometry="flat": FlatDensity with shape (nb_shells, npix, npix)
        - geometry="density": DensityField with shape (nb_shells, mesh_x, mesh_y, mesh_z)
        - geometry="particles": ParticleField with shape (nb_shells, mesh_x, mesh_y, mesh_z, 3)
        Each snapshot has scale_factors set to the corresponding scale factor.

    Raises
    ------
    ValueError
        If fields don't match or geometry is invalid.

    Examples
    --------
    >>> cosmo = Planck18()
    >>> dx, p = lpt(cosmo, initial_field, a=0.1, order=1)
    >>> lightcone = nbody(cosmo, dx, p, t1=1.0, geometry="spherical")
    >>> lightcone.array.shape  # (nb_shells, 12*nside**2)
    >>> lightcone.scale_factors.shape  # (nb_shells,)
    """
    # Validate inputs
    if geometry not in ["spherical", "flat", "density", "particles"]:
        raise ValueError(
            f"geometry must be 'spherical' or 'flat' or 'density' or 'particles', got {geometry}")
    if ts is None and nb_shells is None:
        raise ValueError("Either ts or nb_shells must be provided.")

    # Validate fields match
    if dx_field.mesh_size != p_field.mesh_size:
        raise ValueError("dx_field and p_field must have matching mesh_size")
    if dx_field.box_size != p_field.box_size:
        raise ValueError("dx_field and p_field must have matching box_size")

    # Extract metadata from fields
    t0 = jnp.atleast_1d(dx_field.scale_factors).squeeze()
    if not jnp.isscalar(t0):
        raise ValueError("Starting scale factor t0 must be a scalar.")

    mesh_size = dx_field.mesh_size
    box_size = dx_field.box_size
    observer_position_mpc = dx_field.observer_position_mpc
    sharding = dx_field.sharding
    halo_size = dx_field.halo_size
    nside = dx_field.nside
    flatsky_npix = dx_field.flatsky_npix

    # Compute shell centers using field properties
    if ts is None:
        ts = compute_snapshot_scale_factors(cosmo, dx_field, nb_shells=nb_shells)

    density_plane_width = dx_field.density_width(nb_shells=ts.shape[0])
    # Create ODE terms using local symplectic_ode with ParticleField
    drift, kick = symplectic_ode(dx_field, paint_mode="relative")
    ode_terms = (ODETerm(kick), ODETerm(drift))


    # Set up SaveAt with appropriate density function
    if geometry == "spherical":
        if nside is None:
            raise ValueError(
                "Field must have nside set for spherical geometry")

        def snapshot_fn(t, y, args):
            # Convert scale factor to comoving distance
            r_center = jc.background.radial_comoving_distance(cosmo, t)
            cosmo._workspace = {}

            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factors=t)
            # Paint to spherical
            return particle_field.paint_spherical(
                center=r_center,
                mode="relative",
                density_plane_width=density_plane_width,
            )
    elif geometry == "flat":
        if flatsky_npix is None:
            raise ValueError(
                "Field must have flatsky_npix set for flat geometry")

        def snapshot_fn(t, y, args):
            # Convert scale factor to comoving distance
            r_center = jc.background.radial_comoving_distance(cosmo, t)
            cosmo._workspace = {}

            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factors=t)

            # Paint to flat 2D
            return particle_field.paint_2d(
                center=r_center,
                mode="relative",
                density_plane_width=density_plane_width,
            )

    elif geometry == "density":
        def snapshot_fn(t, y, args):
            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factors=t)

            # Paint to 3D density grid
            return particle_field.paint(mode="relative")

    elif geometry == "particles":
        def snapshot_fn(t, y, args):
            # Return ParticleField with displacements directly
            particle_field = y[1]
            return particle_field.replace(scale_factors=t)

    saveat = SaveAt(ts=ts, fn=snapshot_fn)

    # Initial state
    y0 = (p_field, dx_field)
    args = cosmo

    # Run integration based on adjoint choice
    if adjoint == "reverse_adjoint":
        solution = reverse_adjoint_integrate(
            ode_terms,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
        )
    else:
        solution = diffeqsolve(
            ode_terms,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args,
            saveat=saveat,
            adjoint=adjoint,
        )
        solution = solution.ys

    # Reverse to get near-to-far ordering
    lightcone = solution[::-1]

    return lightcone
