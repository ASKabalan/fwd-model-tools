from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc

from .._src.lensing import _born_flat, _born_spherical
from ..fields import FieldStatus, FlatKappaField, SphericalDensity, SphericalKappaField

__all__ = ["born"]


def born(
    cosmo,
    lightcone,
    nz_shear,
    min_z=0.01,
    max_z=3.0,
    n_integrate=32,
):
    if nz_shear is None:
        raise ValueError("nz_shear must be provided for lensing")

    if lightcone.status != FieldStatus.LIGHTCONE:
        raise ValueError(f"Expected lightcone with status=LIGHTCONE, got {lightcone.status}")

    scale_factors = jnp.atleast_1d(lightcone.scale_factors)
    n_planes = scale_factors.size

    if lightcone.array.ndim not in [2, 3]:
        raise ValueError(f"Lightcone array must be 2D (spherical) or 3D (flat), got {lightcone.array.ndim}D")

    r_center = jc.background.radial_comoving_distance(cosmo, scale_factors)
    cosmo._workspace = {}

    max_radius = lightcone.max_comoving_radius
    density_plane_width = max_radius / n_planes

    is_spherical = isinstance(lightcone, SphericalDensity)

    if is_spherical:
        source_map = _born_spherical(
            cosmo,
            lightcone,
            r_center,
            scale_factors,
            nz_shear,
            density_plane_width,
            min_z,
            max_z,
            n_integrate,
        )
    else:
        source_map = _born_flat(
            cosmo,
            lightcone,
            r_center,
            scale_factors,
            nz_shear,
            density_plane_width,
            min_z,
            max_z,
            n_integrate,
        )

    base_field = lightcone.replace(status=FieldStatus.KAPPA)

    if is_spherical:
        kappas = SphericalKappaField.FromDensityMetadata(
            array=source_map,
            field=base_field,
            status=FieldStatus.KAPPA,
            z_sources=nz_shear,
        )
    else:
        kappas = FlatKappaField.FromDensityMetadata(
            array=source_map,
            field=base_field,
            status=FieldStatus.KAPPA,
            z_sources=nz_shear,
        )

    return kappas
