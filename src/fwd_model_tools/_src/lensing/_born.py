from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates
from jax_cosmo.scipy.integrate import simps

from ...fields import FlatDensity, SphericalDensity
from ._normalize_nz import _normalize_sources

__all__ = ["_born_core_impl", "_born_spherical", "_born_flat"]


def _born_core_impl(
    cosmo,
    density_planes,
    r,
    a,
    z_source,
    d_r,
    pixel_size=None,
    coords=None,
    field_size=None,
):
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    chi_s = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_source))
    n_planes = len(r)

    is_spherical = density_planes.ndim == 2

    if not is_spherical:
        if pixel_size is None:
            raise AssertionError("pixel_size is required for flat geometry")
        pixel_size = jnp.asarray(pixel_size)

        if coords is None:
            if field_size is None:
                raise AssertionError("field_size is required when coords not provided")
            ny, nx = density_planes.shape[-2:]
            if isinstance(field_size, (tuple, list, jnp.ndarray)):
                fx, fy = field_size
            else:
                fx = fy = field_size
            xgrid, ygrid = jnp.meshgrid(
                jnp.linspace(0, fx, nx, endpoint=False),
                jnp.linspace(0, fy, ny, endpoint=False),
            )
            coords = jnp.stack([xgrid, ygrid], axis=0) * (jnp.pi / 180)

    r_b = r.reshape(n_planes, *((1, ) * (density_planes.ndim - 1)))
    a_b = a.reshape(n_planes, *((1, ) * (density_planes.ndim - 1)))

    mean_axes = tuple(range(1, density_planes.ndim))
    rho_mean = jnp.mean(density_planes, axis=mean_axes, keepdims=True)
    eps = jnp.finfo(rho_mean.dtype).eps
    safe_rho_mean = jnp.where(rho_mean == 0, eps, rho_mean)
    delta = density_planes / safe_rho_mean - 1

    kappa_contributions = delta * (d_r * r_b / a_b)
    kappa_contributions *= constant_factor

    if not is_spherical:

        def interpolate_plane(delta_plane, chi_plane):
            physical_coords = coords * chi_plane / pixel_size[:, None, None]
            return map_coordinates(delta_plane, physical_coords - 0.5, order=1, mode="wrap")

        kappa_contributions = jax.vmap(interpolate_plane)(kappa_contributions, r)

    if jnp.ndim(z_source) > 0 and not is_spherical:
        chi_s = jnp.expand_dims(chi_s, axis=1)
    lensing_efficiency = jnp.clip(1.0 - (r_b / chi_s), 0, 1000)
    lensing_efficiency = jnp.expand_dims(lensing_efficiency, axis=-1)
    kappa_contributions = jnp.expand_dims(kappa_contributions, axis=1)
    final_contributions = lensing_efficiency * kappa_contributions
    convergence = jnp.sum(final_contributions, axis=0)

    if jnp.ndim(z_source) == 0:
        convergence = jnp.squeeze(convergence)

    return convergence


def _born_spherical(
    cosmo,
    lightcone: SphericalDensity,
    r_center,
    scale_factors,
    nz_shear,
    density_plane_width,
    min_z,
    max_z,
    n_integrate,
):
    source_kind, sources = _normalize_sources(nz_shear)

    if source_kind == "distribution":
        kappa_maps = []

        for nz in sources:

            def integrand_fn(z):
                return nz(z).reshape([-1, 1]) * _born_core_impl(
                    cosmo,
                    lightcone.array,
                    r_center,
                    scale_factors,
                    z,
                    density_plane_width,
                )

            kappa_maps.append(simps(integrand_fn, min_z, max_z, N=n_integrate))
        kappa_maps = jnp.stack(kappa_maps, axis=0)
    else:
        kappa_maps = _born_core_impl(cosmo, lightcone.array, r_center, scale_factors, sources, density_plane_width)
    return kappa_maps


def _born_flat(
    cosmo,
    lightcone: FlatDensity,
    r_center,
    scale_factors,
    nz_shear,
    density_plane_width,
    min_z,
    max_z,
    n_integrate,
):
    if lightcone.field_size is None:
        raise ValueError("field_size is required on lightcone for flat-sky convergence")

    source_kind, sources = _normalize_sources(nz_shear)

    field_size = lightcone.field_size
    if isinstance(field_size, (tuple, list, jnp.ndarray)):
        field_size_tuple = tuple(field_size)
    else:
        field_size_tuple = (field_size, field_size)

    if lightcone.flatsky_npix is None:
        raise ValueError("flatsky_npix must be set on lightcone for flat geometry")
    ny, nx = lightcone.flatsky_npix
    pixel_size = (
        lightcone.box_size[1] / ny,
        lightcone.box_size[0] / nx,
    )

    if source_kind == "distribution":
        # Precompute chi(z) for all quadrature points outside the lambda
        kappa_maps = []

        for nz in sources:

            def integrand_fn(z):
                return nz(z).reshape([-1, 1, 1]) * _born_core_impl(
                    cosmo,
                    lightcone.array,
                    r_center,
                    scale_factors,
                    z,
                    density_plane_width,
                    pixel_size=pixel_size,
                    field_size=field_size_tuple,
                )

            kappa_maps.append(simps(integrand_fn, min_z, max_z, N=n_integrate))
        kappa_maps = jnp.stack(kappa_maps, axis=0)
    else:
        kappa_maps = [
            _born_core_impl(
                cosmo,
                lightcone.array,
                r_center,
                scale_factors,
                z,
                density_plane_width,
                pixel_size=pixel_size,
                field_size=field_size_tuple,
            ) for z in sources
        ]

    return list(zip(sources, kappa_maps))
