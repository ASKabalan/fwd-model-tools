"""Probabilistic wrappers that build on the deterministic forward model."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro

from ..fields import AbstractField, DensityField, DensityUnit, FieldStatus
from ..infer import DistributedIC, DistributedNormal
from .config import Configurations
from .forward_model import make_full_field_model

__all__ = ["make_full_field_model", "full_field_probmodel", "mock_probmodel"]


def _flat_pixel_area_arcmin2(lightcone: DensityField) -> float:
    if lightcone.flatsky_npix is None or lightcone.field_size is None:
        raise ValueError("Flat lightcones require both flatsky_npix and field_size metadata")
    size_y, size_x = lightcone.field_size
    ny, nx = lightcone.flatsky_npix
    pixel_y = size_y * 60.0 / ny
    pixel_x = size_x * 60.0 / nx
    return pixel_y * pixel_x


def _spherical_pixel_area_arcmin2(lightcone: DensityField) -> float:
    if lightcone.nside is None:
        raise ValueError("Spherical lightcones require nside metadata")
    arcmin_per_rad = (180.0 / jnp.pi) * 60.0
    pixel_area_sr = 4.0 * jnp.pi / (12.0 * (lightcone.nside**2))
    return pixel_area_sr * (arcmin_per_rad**2)


def full_field_probmodel(config: Configurations, ):
    """Return a NumPyro model for joint inference of cosmology and density fields."""

    geometry = config.geometry
    forward_model = make_full_field_model(config=config)

    def model():
        cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})
        numpyro.deterministic("cosmo", cosmo)

        initial_conditions = numpyro.sample(
            "initial_conditions",
            DistributedIC(
                mesh_size=config.mesh_size,
                box_size=config.box_size,
                observer_position=config.observer_position,
                halo_size=config.halo_size,
                nside=config.nside,
                flatsky_npix=config.flatsky_npix,
                field_size=config.field_size,
                cosmo=cosmo,
                sharding=config.sharding,
            ),
        )

        if not isinstance(initial_conditions, DensityField):
            initial_conditions = DensityField(
                array=initial_conditions,
                mesh_size=config.mesh_size,
                box_size=config.box_size,
                observer_position=config.observer_position,
                halo_size=config.halo_size,
                nside=config.nside,
                flatsky_npix=config.flatsky_npix,
                field_size=config.field_size,
                status=FieldStatus.INITIAL_FIELD,
                unit=DensityUnit.DENSITY,
                sharding=config.sharding,
            )

        numpyro.deterministic("initial_conditions_meta_data", initial_conditions.to_metadata())

        kappa_fields, lightcone = forward_model(cosmo, initial_conditions)

        if config.log_lightcone:
            numpyro.deterministic("lightcone", lightcone)

        if kappa_fields.shape[0] != len(config.nz_shear):
            raise ValueError("Number of convergence maps does not match nz_shear entries")

        if geometry == "spherical":
            pixel_area_arcmin2 = _spherical_pixel_area_arcmin2(lightcone)
        elif geometry == "flat":
            pixel_area_arcmin2 = _flat_pixel_area_arcmin2(lightcone)
        else:
            raise ValueError("geometry must be 'flat' or 'spherical'")

        observed_maps = []
        for idx, (kappa_field, nz) in enumerate(zip(kappa_fields.array, config.nz_shear)):
            sigma = config.sigma_e / jnp.sqrt(nz.gals_per_arcmin2 * pixel_area_arcmin2)
            observed_maps.append(
                numpyro.sample(
                    f"kappa_{idx}",
                    DistributedNormal(loc=kappa_field,
                                      scale=sigma,
                                      mesh_size=config.mesh_size,
                                      box_size=config.box_size,
                                      observer_position=config.observer_position,
                                      halo_size=config.halo_size,
                                      flatsky_npix=config.flatsky_npix,
                                      nside=config.nside,
                                      field_size=config.field_size,
                                      sharding=config.sharding,
                                      field_type=geometry),
                ))

        numpyro.deterministic("kappa_meta_data", kappa_fields.to_metadata())

        return observed_maps

    return model


def mock_probmodel(config: Configurations):
    """Return a NumPyro model that samples only cosmology and initial conditions.

    No forward simulation is run. Registered deterministic sites match those
    expected by :func:`~fwd_model_tools.probabilistic_models.sample2catalog`,
    so the callback can be used directly to save IC catalogs.
    """

    def model():
        cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})
        numpyro.deterministic("cosmo", cosmo)

        initial_conditions = numpyro.sample(
            "initial_conditions",
            DistributedIC(
                mesh_size=config.mesh_size,
                box_size=config.box_size,
                observer_position=config.observer_position,
                halo_size=config.halo_size,
                nside=config.nside,
                flatsky_npix=config.flatsky_npix,
                field_size=config.field_size,
                cosmo=cosmo,
                sharding=config.sharding,
            ),
        )

        if not isinstance(initial_conditions, DensityField):
            initial_conditions = DensityField(
                array=initial_conditions,
                mesh_size=config.mesh_size,
                box_size=config.box_size,
                observer_position=config.observer_position,
                halo_size=config.halo_size,
                nside=config.nside,
                flatsky_npix=config.flatsky_npix,
                field_size=config.field_size,
                status=FieldStatus.INITIAL_FIELD,
                unit=DensityUnit.DENSITY,
                sharding=config.sharding,
            )

        numpyro.deterministic("initial_conditions_meta_data", initial_conditions.to_metadata())
        return initial_conditions

    return model
