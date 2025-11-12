from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from diffrax import ODETerm, RecursiveCheckpointAdjoint, SaveAt, diffeqsolve
from jaxpm.pm import lpt as jaxpm_lpt

from .field import DensityField, FieldStatus, ParticleField, particle_from_density
from .solvers.integrate import integrate as reverse_adjoint_integrate
from .solvers.ode import symplectic_ode
from .solvers.semi_implicit_euler import ReversibleEfficientFastPM
from .utils import compute_snapshot_scale_factors

__all__ = ["lpt", "nbody"]


def lpt(
    cosmo,
    initial_field: DensityField,
    a,
    order: int = 1,
) -> Tuple[ParticleField, ParticleField]:
    """
    Compute LPT displacements/momenta for a DensityField.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology describing the background expansion.
    initial_field : DensityField
        Linear density field packaged with mesh metadata.
    a : float or array-like
        Scale factor(s) at which to evaluate the growth.
        - If scalar: returns (X, Y, Z, 3) shaped ParticleFields
        - If array: returns (N, X, Y, Z, 3) shaped ParticleFields (batched lightcone)
    order : int, default=1
        LPT order (1 or 2 supported via underlying JAXPM implementation).

    Returns
    -------
    tuple[ParticleField, ParticleField]
        Displacement and momentum particle fields.
        Shape depends on input `a`:
        - Scalar a: (X, Y, Z, 3)
        - Array a of shape (N,): (N, X, Y, Z, 3)

    Examples
    --------
    Single scale factor:
    >>> dx, p = lpt(cosmo, field, a=0.5, order=1)
    >>> dx.array.shape  # (256, 256, 256, 3)

    Batched lightcone:
    >>> scale_factors = compute_snapshot_scale_factors(cosmo, field)
    >>> dx, p = lpt(cosmo, field, a=scale_factors, order=1)
    >>> dx.array.shape  # (10, 256, 256, 256, 3) if nb_shells=10
    """
    if not isinstance(initial_field, DensityField):
        raise TypeError("initial_field must be a DensityField instance.")
    if order not in (1, 2):
        raise ValueError("order must be either 1 or 2.")
    if initial_field.status != FieldStatus.INITIAL_FIELD:
        raise ValueError(
            "initial_field must have status FieldStatus.INITIAL_FIELD.")
    a = jnp.atleast_1d(a)

    if a.size > 1:
        a = a.reshape((1, 1, -1, 1))
        if a.shape[2] != initial_field.shape[2]:
            raise ValueError(
                "When passing multiple scale factors, the number of scale factors "
                "must match the number of shells in the DensityField.")

    dx, p, _ = jaxpm_lpt(
        cosmo,
        initial_field.array,
        particles=None,
        a=a,
        halo_size=initial_field.halo_size,
        sharding=initial_field.sharding,
        order=order,
    )

    status = FieldStatus.LPT1 if order == 1 else FieldStatus.LPT2
    dx_field = particle_from_density(dx,
                                      initial_field,
                                      scale_factor=a,
                                      status=status)
    p_field = particle_from_density(p,
                                     initial_field,
                                     scale_factor=a,
                                     status=status)
    return dx_field, p_field


def nbody(
    cosmo,
    dx_field: ParticleField,
    p_field: ParticleField,
    t1: float = 1.0,
    dt0: float = 0.05,
    ts: jnp.ndarray | None = None,
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
        Each snapshot has scale_factor set to the corresponding scale factor.

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
    >>> lightcone.scale_factor.shape  # (nb_shells,)
    """
    # Validate inputs
    if geometry not in ["spherical", "flat", "density", "particles"]:
        raise ValueError(
            f"geometry must be 'spherical' or 'flat' or 'density' or 'particles', got {geometry}")

    # Validate fields match
    if dx_field.mesh_size != p_field.mesh_size:
        raise ValueError("dx_field and p_field must have matching mesh_size")
    if dx_field.box_size != p_field.box_size:
        raise ValueError("dx_field and p_field must have matching box_size")

    # Extract metadata from fields
    t0 = jnp.atleast_1d(dx_field.scale_factor).squeeze()
    if not jnp.isscalar(t0):
        raise ValueError("Starting scale factor t0 must be a scalar.")

    mesh_size = dx_field.mesh_size
    box_size = dx_field.box_size
    observer_position_mpc = dx_field.observer_position_mpc
    sharding = dx_field.sharding
    halo_size = dx_field.halo_size
    nside = dx_field.nside
    flatsky_npix = dx_field.flatsky_npix
    nb_shells = dx_field.nb_shells

    # Compute shell centers using field properties
    if ts is None:
        ts = compute_snapshot_scale_factors(cosmo, dx_field)

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
            particle_field = particle_field.replace(scale_factor=t)
            # Paint to spherical
            return particle_field.paint_spherical(
                center=r_center,
                mode="relative"
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
            particle_field = particle_field.replace(scale_factor=t)

            # Paint to flat 2D
            return particle_field.paint_2d(
                center=r_center,
                mode="relative"
            )

    elif geometry == "density":
        def snapshot_fn(t, y, args):
            # Create temporary ParticleField from displacement array
            particle_field = y[1]
            particle_field = particle_field.replace(scale_factor=t)

            # Paint to 3D density grid
            return particle_field.paint(mode="relative")

    elif geometry == "particles":
        def snapshot_fn(t, y, args):
            # Return ParticleField with displacements directly
            particle_field = y[1]
            return particle_field.replace(scale_factor=t)

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
