"""Deterministic forward models used by the probabilistic layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import DensityField, FieldStatus
from ..fields.painting import PaintingOptions
from ..initial import interpolate_initial_conditions
from ..lensing import born, raytrace
from ..parameters import Planck18
from ..pm import EfficientDriftDoubleKick, NoCorrection, NoInterp, OnionTiler, lpt, nbody
from .config import Configurations

__all__ = ["make_full_field_model"]


def make_full_field_model(
    meta_data: DensityField,
    *,
    config: Configurations,
):
    """Build the deterministic forward model returning kappa maps and lightcone."""

    geometry = config.geometry
    if geometry not in {"flat", "spherical"}:
        raise ValueError("geometry must be either 'flat' or 'spherical'")

    def forward_model(cosmo, initial_conditions):
        lin_field = interpolate_initial_conditions(
            initial_field=initial_conditions,
            mesh_size=meta_data.mesh_size,
            box_size=meta_data.box_size,
            cosmo=cosmo,
            observer_position=meta_data.observer_position,
            flatsky_npix=meta_data.flatsky_npix,
            nside=meta_data.nside,
            halo_size=meta_data.halo_size,
            sharding=meta_data.sharding,
        )

        dx_field, p_field = lpt(
            cosmo,
            lin_field,
            scale_factor_spec=config.t0,
            order=config.lpt_order,
        )

        # Create solver with appropriate interp_kernel based on geometry
        if geometry == "spherical":
            interp_kernel = OnionTiler(painting=PaintingOptions(target="spherical"))
        else:
            interp_kernel = NoInterp(painting=PaintingOptions(target="flat"))

        solver = EfficientDriftDoubleKick(
            pgd_kernel=NoCorrection(),
            interp_kernel=interp_kernel,
        )

        lightcone = nbody(
            cosmo,
            dx_field,
            p_field,
            t1=config.t1,
            dt0=config.dt0,
            nb_shells=config.number_of_shells,
            solver=solver,
            adjoint=config.adjoint,
        )

        lensing_fn = raytrace if config.lensing == "raytrace" else born
        print(f"len of nz_shear: {len(config.nz_shear)}")
        kappa_fields = lensing_fn(
            cosmo,
            lightcone,
            nz_shear=config.nz_shear,
            min_z=config.min_redshift,
            max_z=config.max_redshift,
        )

        return kappa_fields, lightcone, lin_field

    return jax.jit(forward_model)
