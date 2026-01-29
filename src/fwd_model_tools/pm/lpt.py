from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax import lax
from jaxpm.distributed import uniform_particles
from jaxpm.pm import lpt as jaxpm_lpt
from jaxtyping import Array

from ..fields import DensityField, FieldStatus, PaintingOptions, ParticleField, PositionUnit
from ..utils import compute_particle_scale_factors, distances

__all__ = ["lpt"]


@partial(jax.jit, static_argnames=["order", "painting"])
def lpt(cosmo: Any,
        initial_field: DensityField,
        scale_factor_spec,
        *,
        order: int = 1,
        initial_particles: Array = None,
        painting: PaintingOptions | None = None) -> tuple[Any, ParticleField]:
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
        - If scalar: returns ParticleField displacements/momenta on the 3D mesh.
        - If array: returns (FlatDensity lightcone, ParticleField momenta).
    order : int, default=1
        LPT order (1 or 2 supported via underlying JAXPM implementation).

    Returns
    -------
    tuple
        If `a` is scalar:
            (dx_field: ParticleField, p_field: ParticleField)
        If `a` is an array (lightcone mode):
            (flat_lightcone: FlatDensity, p_field: ParticleField)
    """
    if not isinstance(initial_field, DensityField):
        raise TypeError("initial_field must be a DensityField instance.")
    if order not in (1, 2):
        raise ValueError("order must be either 1 or 2.")
    if initial_field.status != FieldStatus.INITIAL_FIELD:
        raise ValueError("initial_field must have status FieldStatus.INITIAL_FIELD.")

    if initial_particles is None:
        # positions in GRID_RELATIVE (0..1 over the box) or GRID_ABSOLUTE indices,
        # depending on your convention in PositionUnit.
        initial_particles = uniform_particles(initial_field.mesh_size, sharding=initial_field.sharding)
        user_defined_particles = False
    else:
        user_defined_particles = True

    initial_particles = ParticleField.FromDensityMetadata(
        array=initial_particles,
        field=initial_field,
        unit=PositionUnit.GRID_ABSOLUTE,
    )
    scale_factor_spec = jnp.asarray(scale_factor_spec)
    if jnp.isscalar(scale_factor_spec):
        a = jnp.atleast_1d(scale_factor_spec)
        snapshot_r = None
        density_plane_width = None
    elif scale_factor_spec.size == 2:
        a_near, a_far = scale_factor_spec
        # Find center scale factor for lightcone shell
        a = 0.5 * (a_near + a_far)
        r_near, r_far = r[0], r[1]
        snapshot_r = r[None, ...]
        density_plane_width = (r_far - r_near)[None, ...]
    # Snapshots at specific times
    elif scale_factor_spec.ndim == 1:
        a = compute_particle_scale_factors(cosmo, initial_particles)[..., None]
        r = jc.background.radial_comoving_distance(cosmo, scale_factor_spec)
        # Width of bin i (centered at ri​):
        # Set snap info
        snapshot_r = r
        density_plane_width = distances(r, initial_particles.max_comoving_radius)
    # Snapshot at near a and far a for each shell
    elif scale_factor_spec.ndim == 2:
        a_near, a_far = scale_factor_spec[:, 0], scale_factor_spec[:, 1]
        a = compute_particle_scale_factors(cosmo, initial_particles)[..., None]
        r_near = jc.background.radial_comoving_distance(cosmo, a_near)
        r_far = jc.background.radial_comoving_distance(cosmo, a_far)
        # Set snap info
        snapshot_r = 0.5 * (r_near + r_far)
        density_plane_width = r_far - r_near
    # User specified scale factors for each particle
    elif scale_factor_spec.ndim == 3:
        a = scale_factor_spec[..., None]
        # No snapshot ..
        # in this case we return only the displacements with the requested redshifts/scale factors
        snapshot_r = None
        density_plane_width = None
    else:
        raise ValueError("scale_factor_spec has invalid shape.")

    dx, p, _ = jaxpm_lpt(
        cosmo,
        initial_field.array,
        particles=initial_particles.array if user_defined_particles else None,
        a=a,
        halo_size=initial_field.halo_size,
        sharding=initial_field.sharding,
        order=order,
    )

    status = FieldStatus.LPT1 if order == 1 else FieldStatus.LPT2
    if a.size > 1:
        status = FieldStatus.LIGHTCONE

    z_sources = jc.utils.a2z(a)
    comoving_centers = jc.background.radial_comoving_distance(cosmo, a.flatten()).reshape(a.shape)

    dx_field = ParticleField.FromDensityMetadata(
        array=dx,
        field=initial_field,
        status=status,
        scale_factors=a,
        comoving_centers=comoving_centers,
        z_sources=z_sources,
        unit=PositionUnit.GRID_RELATIVE,
    )
    p_field = ParticleField.FromDensityMetadata(
        array=p,
        field=initial_field,
        status=status,
        scale_factors=a,
        comoving_centers=comoving_centers,
        z_sources=z_sources,
        unit=PositionUnit.GRID_RELATIVE,
    )

    if snapshot_r is not None:
        target = painting.target if painting is not None else "particles"

        if target == "flat":
            dx_field = dx_field.paint_2d(
                center=snapshot_r,
                density_plane_width=density_plane_width,
                weights=painting.weights,
                batch_size=painting.batch_size,
            )
            a_snapshot = jc.background.a_of_chi(cosmo, snapshot_r)
            z_snapshot = jc.utils.a2z(a_snapshot)
            dx_field = dx_field.replace(
                scale_factors=a_snapshot,
                z_sources=z_snapshot,
            )
            dx_field = dx_field[::-1]
        elif target == "spherical":
            dx_field = dx_field.paint_spherical(
                center=snapshot_r,
                density_plane_width=density_plane_width,
                scheme=painting.scheme,
                weights=painting.weights,
                kernel_width_arcmin=painting.kernel_width_arcmin,
                smoothing_interpretation=painting.smoothing_interpretation,
                paint_nside=painting.paint_nside,
                ud_grade_power=painting.ud_grade_power,
                ud_grade_order_in=painting.ud_grade_order_in,
                ud_grade_order_out=painting.ud_grade_order_out,
                ud_grade_pess=painting.ud_grade_pess,
                batch_size=painting.batch_size,
            )
            a_snapshot = jc.background.a_of_chi(cosmo, snapshot_r)
            z_snapshot = jc.utils.a2z(a_snapshot)
            dx_field = dx_field.replace(
                scale_factors=a_snapshot,
                z_sources=z_snapshot,
            )
            dx_field = dx_field[::-1]
        elif target == "density":
            dx_field = dx_field.paint(
                weights=painting.weights if painting else 1.0,
                chunk_size=painting.chunk_size if painting else 2**24,
                batch_size=painting.batch_size if painting else None,
            )
        elif target == "particles":
            pass
        else:
            raise ValueError(f"Unknown painting target {target}.")

    else:
        target = painting.target if painting is not None else "particles"
        if target != "particles":
            raise ValueError("painting.target must be 'particles' when scale_factor_spec is a scalar or 3D array.")

    return dx_field, p_field


# ==========================================================
