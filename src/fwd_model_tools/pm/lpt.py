from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.pm import lpt as jaxpm_lpt

from ..fields import DensityField, DensityStatus, FieldStatus, FlatDensity, ParticleField

__all__ = ["lpt"]


@partial(jax.jit, static_argnames=["order"])
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
        raise ValueError("initial_field must have status FieldStatus.INITIAL_FIELD.")

    a = jnp.atleast_1d(a)
    is_lightcone = a.size > 1

    a_for_solver = a
    if is_lightcone:
        # For LPT lightcone mode, we assume one scale factor per voxel along
        # the z-axis of the mesh.
        a_for_solver = a.reshape((1, 1, -1, 1))
        if a_for_solver.shape[2] != initial_field.mesh_size[2]:
            raise ValueError("When passing multiple scale factors, the number of scale "
                             "factors must match the size of the z-axis in the mesh.")

    dx, p, _ = jaxpm_lpt(
        cosmo,
        initial_field.array,
        particles=None,
        a=a_for_solver,
        halo_size=initial_field.halo_size,
        sharding=initial_field.sharding,
        order=order,
    )

    status = FieldStatus.LPT1 if order == 1 else FieldStatus.LPT2
    if not is_lightcone:
        # Standard single-snapshot LPT: return ParticleField displacements
        # and momenta defined on the 3D mesh.
        dx_field = ParticleField.FromDensityMetadata(
            dx,
            initial_field,
            status=status,
            scale_factors=a,
        )
        p_field = ParticleField.FromDensityMetadata(
            p,
            initial_field,
            status=status,
            scale_factors=a,
        )
        return dx_field, p_field

    # LPT lightcone mode: interpret the box as Nz planes of (Nx, Ny) and
    # return a FlatDensity stack with one map per z-plane. The centers of
    # these planes correspond to the provided scale factors, converted to
    # comoving distances.
    #
    # First, wrap displacements as a ParticleField sharing the original
    # geometry, then paint to a 3D density field.
    dx_particles = ParticleField.FromDensityMetadata(
        dx,
        initial_field,
        status=status,
        scale_factors=a,
    )
    density_3d = dx_particles.paint()

    # Split the 3D density (nx, ny, nz) into nz planes of (ny, nx).
    density_array = jnp.asarray(density_3d.array)
    if density_array.ndim != 3:
        raise ValueError("LPT lightcone painting expects a 3D density field "
                         f"(nx, ny, nz); got array with shape {density_array.shape}.")
    flat_array = jnp.transpose(density_array, (2, 1, 0))  # (nz, ny, nx)

    # Convert the scale factors used into comoving distances to serve as
    # per-plane centers for the lightcone stack.
    r_centers = jc.background.radial_comoving_distance(cosmo, a)

    flat_lightcone = FlatDensity.FromDensityMetadata(
        array=flat_array,
        density_field=density_3d,
        status=DensityStatus.LIGHTCONE,
        scale_factors=a,
    )

    # Momentafield is still returned as a ParticleField on the 3D mesh for
    # callers that need it, but the primary lightcone representation is the
    # FlatDensity stack.
    p_field = ParticleField.FromDensityMetadata(
        p,
        initial_field,
        status=status,
        scale_factors=a,
    )
    return flat_lightcone, p_field
