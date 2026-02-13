"""Probabilistic wrappers that build on the deterministic forward model."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..fields import DensityField
from ..parameters import Planck18
from ..sampling import DistributedNormal
from .config import Configurations
from .forward_model import make_full_field_model

__all__ = ["Planck18", "make_full_field_model", "full_field_probmodel"]


def _flat_pixel_area_arcmin2(lightcone: DensityField) -> float:
    if lightcone.flatsky_npix is None or lightcone.field_size is None:
        raise ValueError("Flat lightcones require both flatsky_npix and field_size metadata")
    if isinstance(lightcone.field_size, (tuple, list)):
        size_y, size_x = lightcone.field_size
    else:
        size_y = size_x = lightcone.field_size
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


def full_field_probmodel(
    template_field: DensityField,
    config: Configurations,
):
    """Return a NumPyro model for joint inference of cosmology and density fields."""

    geometry = config.geometry
    forward_model = make_full_field_model(template_field, config=config)

    def model():
        cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})

        initial_conditions = numpyro.sample(
            "initial_conditions",
            DistributedNormal(
                jnp.zeros(template_field.mesh_size),
                jnp.ones(template_field.mesh_size),
                template_field.sharding,
            ),
        )

        kappa_fields, lightcone, lin_field = forward_model(cosmo, initial_conditions)

        if config.log_lightcone:
            numpyro.deterministic("lightcone", lightcone.array)
            numpyro.deterministic("lightcone_z", lightcone.z_sources)
        if config.log_ic:
            numpyro.deterministic("ic", lin_field.array)

        print(f"Number of kappa fields: {kappa_fields.shape} number of nz_shear: {len(config.nz_shear)}")
        if kappa_fields.shape[0] != len(config.nz_shear):
            raise ValueError("Number of convergence maps does not match nz_shear entries")

        observed_maps = []
        if geometry == "spherical":
            pixel_area_arcmin2 = _spherical_pixel_area_arcmin2(lightcone)
            for idx, (kappa_field, nz) in enumerate(zip(kappa_fields.array, config.nz_shear)):
                sigma = config.sigma_e / jnp.sqrt(nz.gals_per_arcmin2 * pixel_area_arcmin2)
                observed_maps.append(numpyro.sample(f"kappa_{idx}", dist.Normal(kappa_field, sigma)))
        elif geometry == "flat":
            pixel_area_arcmin2 = _flat_pixel_area_arcmin2(lightcone)
            for idx, (kappa_field, nz) in enumerate(zip(kappa_fields.array, config.nz_shear)):
                sigma = config.sigma_e / jnp.sqrt(nz.gals_per_arcmin2 * pixel_area_arcmin2)
                observed_maps.append(numpyro.sample(f"kappa_{idx}", dist.Normal(kappa_field, sigma)))
        else:
            raise ValueError("geometry must be 'flat' or 'spherical'")

        return observed_maps

    return model
