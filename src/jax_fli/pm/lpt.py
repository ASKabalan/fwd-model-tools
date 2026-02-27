from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax_cosmo as jc
from jaxpm.distributed import uniform_particles
from jaxpm.pm import lpt as jaxpm_lpt
from jaxtyping import Array

from ..fields import DensityField, FieldStatus, PaintingOptions, ParticleField, PositionUnit
from ..utils import compute_particle_scale_factors
from ._resolve_geometry import resolve_ts_geometry

__all__ = ["lpt"]


@partial(jax.jit, static_argnames=["order", "nb_shells", "painting"])
def lpt(
    cosmo: Any,
    initial_field: DensityField,
    *,
    ts=None,
    nb_shells: int | None = None,
    density_widths=None,
    order: int = 1,
    initial_particles: Array = None,
    painting: PaintingOptions = PaintingOptions(target="particles"),
) -> tuple[Any, ParticleField]:
    """
    Compute LPT displacements/momenta for a DensityField.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology describing the background expansion.
    initial_field : DensityField
        Linear density field packaged with mesh metadata.
    ts : float, 1-D array, or 2-D ``(2, N)`` array, optional
        Scale factor specification.  Mutually exclusive with *nb_shells*.

        - **scalar** (no *density_width*): snapshot mode — returns ``ParticleField``
          displacements/momenta at that single epoch.
        - **scalar + density_width**: single-shell lightcone.
        - **1-D array**: shell-centre scale factors; widths derived from
          ``distances()`` or broadcast from *density_width*.
        - **2-D (2, N)**: near/far scale factors per shell; centres and widths
          are derived automatically.

    nb_shells : int, optional
        Number of radial lightcone shells (alternative to *ts*).
    density_width : float or array, optional
        Override shell widths.  When *ts* is a scalar this activates
        single-shell lightcone mode.
    t1 : float, default=1.0
        End time or detailed time specification. Used if *ts* and *nb_shells* are None.
    order : int, default=1
        LPT order (1 or 2 supported via underlying JAXPM implementation).
    initial_particles : Array, optional
        Custom initial particle positions.
    painting : PaintingOptions, optional
        Painting configuration for lightcone output.

    Returns
    -------
    tuple
        If snapshot mode:
            ``(dx_field: ParticleField, p_field: ParticleField)``
        If lightcone mode:
            ``(lightcone_field, p_field: ParticleField)``

    Raises
    ------
    ValueError
        If both *ts* and *nb_shells* are ``None``, or both are set.
    """
    if not isinstance(initial_field, DensityField):
        raise TypeError("initial_field must be a DensityField instance.")
    if order not in (1, 2):
        raise ValueError("order must be either 1 or 2.")
    if initial_field.status != FieldStatus.INITIAL_FIELD:
        raise ValueError("initial_field must have status FieldStatus.INITIAL_FIELD.")

    if initial_particles is None:
        initial_particles = uniform_particles(initial_field.mesh_size, sharding=initial_field.sharding)
        user_defined_particles = False
    else:
        user_defined_particles = True

    initial_particles = ParticleField.FromDensityMetadata(
        array=initial_particles,
        field=initial_field,
        unit=PositionUnit.GRID_ABSOLUTE,
    )

    ts_resolved, r_centers, density_widths, is_lightcone = resolve_ts_geometry(
        cosmo, initial_field, painting=painting, ts=ts, nb_shells=nb_shells, density_widths=density_widths
    )

    if is_lightcone:
        a = compute_particle_scale_factors(cosmo, initial_particles)[..., None]
        snapshot_r = r_centers
        density_plane_width = density_widths
    else:
        a = ts_resolved
        snapshot_r = None
        density_plane_width = None

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
            if dx_field.is_batched():
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
            if dx_field.is_batched():
                dx_field = dx_field[::-1]
        elif target == "density":
            pass
        elif target == "particles":
            pass
        else:
            raise ValueError(f"Unknown painting target {target}.")

    else:
        if painting.target not in ("particles", "density"):
            raise ValueError(
                f"Painting target {painting.target} is incompatible with snapshot mode. Use 'particles' or 'density'."
            )

    if painting.target == "density":
        dx_field = dx_field.paint(
            weights=painting.weights, chunk_size=painting.chunk_size, batch_size=painting.batch_size
        )

    return dx_field, p_field
