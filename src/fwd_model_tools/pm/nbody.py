from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from diffrax import RecursiveCheckpointAdjoint, SaveAt, diffeqsolve

from ..fields import FieldStatus, ParticleField
from ..utils import compute_lightcone_shells
from .integrate import integrate as reverse_adjoint_integrate
from .solvers import ReversibleEfficientFastPM, setup_odeterms

__all__ = ["nbody"]


@partial(jax.jit, static_argnames=["t1", "dt0", "nb_shells", "geometry", "solver", "adjoint"])
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
        raise ValueError(f"geometry must be 'spherical' or 'flat' or 'density' or 'particles', got {geometry}")
    if ts is None and nb_shells is None:
        raise ValueError("Either ts or nb_shells must be provided.")
    assert dx_field.status == FieldStatus.LPT1 or dx_field.status == FieldStatus.LPT2, (
        "dx_field must have status FieldStatus.LPT1 or FieldStatus.LPT2."
    )
    assert p_field.status == FieldStatus.LPT1 or p_field.status == FieldStatus.LPT2, (
        "p_field must have status FieldStatus.LPT1 or FieldStatus.LPT2."
    )

    # Validate fields match
    if dx_field.mesh_size != p_field.mesh_size:
        raise ValueError("dx_field and p_field must have matching mesh_size")
    if dx_field.box_size != p_field.box_size:
        raise ValueError("dx_field and p_field must have matching box_size")

    # Extract metadata from fields
    t0 = jnp.atleast_1d(dx_field.scale_factors).squeeze()
    if not jnp.isscalar(t0):
        raise ValueError("Starting scale factor t0 must be a scalar.")

    nside = dx_field.nside
    flatsky_npix = dx_field.flatsky_npix

    # Compute shell centers using field properties
    if ts is None:
        r_centers, ts = compute_lightcone_shells(cosmo, dx_field, nb_shells=nb_shells)
    else:
        assert jnp.atleast_1d(ts).size >= 2, "ts must have at least two entries for width computation"
        r_centers = jc.background.radial_comoving_distance(cosmo, ts)

    # Determine density width based on shell spacing
    # Width of bin i (centered at ri​):
    inner_edges = 0.5 * (r_centers[1:] + r_centers[:-1])
    start_edge = r_centers[0] - (inner_edges[0] - r_centers[0])
    end_edge = r_centers[-1] + (r_centers[-1] - inner_edges[-1])
    r_edges = jnp.concatenate([jnp.array([start_edge]), inner_edges, jnp.array([end_edge])])
    density_plane_width = r_edges[:-1] - r_edges[1:]
    # Create ODE terms using local symplectic_ode with ParticleField
    ode_terms, first_kick_term = setup_odeterms(solver=solver, cosmo=cosmo, dx_field=dx_field, dt0=dt0)
    if first_kick_term is not None:
        p_field, dx_field = solver.first_step(first_kick_term, t0, dt0, (p_field, dx_field), args=None)

    # Set up SaveAt with appropriate density function
    if geometry == "spherical":
        if nside is None:
            raise ValueError("Field must have nside set for spherical geometry")

        def snapshot_fn(t, y, args):
            # Convert scale factor to comoving distance
            iter_indx = jnp.argwhere(ts == t, size=1)[0]
            r_center = r_centers[iter_indx]
            width = density_plane_width[iter_indx]

            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factors=t)
            # Paint to spherical
            return particle_field.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme="bilinear",
            )
    elif geometry == "flat":
        if flatsky_npix is None:
            raise ValueError("Field must have flatsky_npix set for flat geometry")

        def snapshot_fn(t, y, args):
            # Convert scale factor to comoving distance
            width_index = jnp.argwhere(ts == t, size=1)[0]
            r_center = r_centers[width_index]
            width = density_plane_width[width_index]

            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factors=t)

            # Paint to flat 2D
            return particle_field.paint_2d(
                center=r_center,
                density_plane_width=width,
            )

    elif geometry == "density":

        def snapshot_fn(t, y, args):
            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factors=t)

            # Paint to 3D density grid
            return particle_field.paint()

    elif geometry == "particles":

        def snapshot_fn(t, y, args):
            # Return ParticleField with displacements directly
            particle_field = y[1]
            return particle_field.replace(scale_factors=t)

    saveat = SaveAt(ts=ts, fn=snapshot_fn)

    # Initial state
    y0 = (p_field, dx_field)

    # Run integration based on adjoint choice
    if adjoint == "reverse_adjoint":
        solution = reverse_adjoint_integrate(
            ode_terms,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            args=density_plane_width,
        )
    else:
        solution = diffeqsolve(
            ode_terms,
            solver,
            t0,
            t1,
            dt0,
            y0,
            saveat=saveat,
            adjoint=adjoint,
            args=density_plane_width,
        )
        solution = solution.ys

    # Reverse to get near-to-far ordering
    lightcone = solution[::-1]

    # Set Z_Sources and r_comoving for each shell
    scale_factors = lightcone.scale_factors
    z_sources = jc.utils.a2z(scale_factors)
    lightcone = lightcone.replace(z_sources=z_sources, comoving_centers=r_centers, status=FieldStatus.LIGHTCONE)

    return lightcone
