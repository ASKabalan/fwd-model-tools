"""Deterministic forward models used by the probabilistic layer."""

from __future__ import annotations

import jax
import jax_cosmo as jc

from fwd_model_tools.fields import DensityField, FieldStatus
from fwd_model_tools.initial import interpolate_initial_conditions
from fwd_model_tools.lensing import born, raytrace
from fwd_model_tools.pm import lpt, nbody, ReversibleKickDriftKick , ReversibleSymplecticEuler
import jax.numpy as jnp
from .config import Configurations

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

        lightcone = nbody(
            cosmo,
            dx_field,
            p_field,
            t1=config.t1,
            dt0=config.dt0,
            nb_shells=config.number_of_shells,
            geometry=geometry,
            adjoint=config.adjoint,
            solver=ReversibleSymplecticEuler(),
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
