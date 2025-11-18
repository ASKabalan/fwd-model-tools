"""Deterministic forward models used by the probabilistic layer."""

from __future__ import annotations

import jax
import jax_cosmo as jc

from .config import Configurations
from fwd_model_tools.fields import DensityField, FieldStatus
from fwd_model_tools.initial import interpolate_initial_conditions
from fwd_model_tools.lensing import born, raytrace
from fwd_model_tools.pm import lpt, nbody

__all__ = ["Planck18", "make_full_field_model"]


def Planck18(**overrides):
    """Return a Planck 2018 cosmology instance with optional overrides."""

    params = {
        "Omega_c": 0.2607,
        "Omega_b": 0.0490,
        "Omega_k": 0.0,
        "h": 0.6766,
        "n_s": 0.9665,
        "sigma8": 0.8102,
        "w0": -1.0,
        "wa": 0.0,
    }
    params.update(overrides)
    return jc.Cosmology(**params)


def make_full_field_model(
    template_field: DensityField,
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
            mesh_size=template_field.mesh_size,
            box_size=template_field.box_size,
            cosmo=cosmo,
            observer_position=template_field.observer_position,
            flatsky_npix=template_field.flatsky_npix,
            nside=template_field.nside,
            halo_size=template_field.halo_size,
            sharding=template_field.sharding,
        )

        lin_density = template_field.replace(
            array=lin_field,
            status=FieldStatus.INITIAL_FIELD,
            scale_factors=config.t0,
        )

        dx_field, p_field = lpt(
            cosmo,
            lin_density,
            a=config.t0,
            order=config.lpt_order,
        )

        lightcone = nbody(
            cosmo,
            dx_field,
            p_field,
            t1=config.t1,
            dt0=config.dt0,
            nb_shells=config.number_of_shells,
            geometry=geometry,
            adjoint=config.adjoint,
        )

        lensing_fn = raytrace if config.lensing == "raytrace" else born
        kappa_fields = lensing_fn(
            cosmo,
            lightcone,
            nz_shear=config.nz_shear,
            min_z=config.min_redshift,
            max_z=config.max_redshift,
        )

        if not isinstance(kappa_fields, list):
            kappa_fields = [kappa_fields]

        return kappa_fields, lightcone, lin_density

    return jax.jit(forward_model)
