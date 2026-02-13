from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates
from jax_cosmo.scipy.integrate import simps

from ...fields import FlatDensity, SphericalDensity

__all__ = ["_born_core_impl", "_born_spherical", "_born_flat"]


def _normalize_sources(nz_shear: Any) -> tuple[str, list[Any]]:
    """Accept either scalar redshifts or jc.redshift distributions (not both)."""

    if isinstance(nz_shear, (list, tuple)):
        entries = list(nz_shear)
    else:
        entries = [nz_shear]

    if not entries:
        raise ValueError("nz_shear must contain at least one entry")

    first = entries[0]
    first_is_distribution = isinstance(first, jc.redshift.redshift_distribution)

    if first_is_distribution:
        for entry in entries:
            if not isinstance(entry, jc.redshift.redshift_distribution):
                raise ValueError("Cannot mix redshift distributions with scalar sources")
        return "distribution", entries

    z_array = jnp.array(entries).squeeze()
    if z_array.ndim == 0:
        z_array = jnp.array([z_array])
    if z_array.ndim != 1:
        raise ValueError("Scalar redshift sources must be 1D array-like")

    return "redshift", z_array


def _radial_comoving_distance_safe(cosmo, a):
    """Compute radial comoving distance."""
    return jc.background.radial_comoving_distance(cosmo, a)


def _born_core_impl(
    cosmo,
    density_planes,
    r,
    a,
    z_source,
    d_r,
    chi_s=None,
    pixel_size=None,
    coords=None,
    field_size=None,
):
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    if chi_s is None:
        chi_s = _radial_comoving_distance_safe(cosmo, jc.utils.z2a(z_source))
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


def _born_core_with_precomputed_chi(
    cosmo,
    density_planes,
    r,
    a,
    z_quadrature,
    d_r,
    chi_s_quadrature,
    pixel_size=None,
    field_size=None,
):
    """Born core that uses precomputed chi_s for all quadrature points.

    This avoids calling radial_comoving_distance inside simps lambdas,
    preventing jax_cosmo workspace tracer leaks under JIT.
    """
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    n_planes = len(r)
    n_z = z_quadrature.shape[0]

    is_spherical = density_planes.ndim == 2

    coords = None
    if not is_spherical:
        if pixel_size is None:
            raise AssertionError("pixel_size is required for flat geometry")
        pixel_size = jnp.asarray(pixel_size)
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

    # chi_s_quadrature shape: (n_z,)
    # r_b shape: (n_planes, 1...) for spherical or (n_planes, 1, 1) for flat
    # We need lensing_efficiency shape: (n_z, n_planes, 1...) for broadcasting
    chi_s_expanded = chi_s_quadrature.reshape(n_z, *((1, ) * density_planes.ndim))
    lensing_efficiency = jnp.clip(1.0 - (r_b / chi_s_expanded), 0, 1000)

    # kappa_contributions: (n_planes, ...) → (1, n_planes, ...)
    kappa_expanded = jnp.expand_dims(kappa_contributions, axis=0)
    # lensing_efficiency: (n_z, n_planes, 1...) already has right dims
    lensing_eff_expanded = jnp.expand_dims(lensing_efficiency, axis=-1)

    # Multiply and sum over planes: result shape (n_z, ...)
    final = lensing_eff_expanded * kappa_expanded
    convergence = jnp.sum(final, axis=1)  # sum over planes

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
        # Precompute chi(z) for all Simpson quadrature points ONCE,
        # outside the simps lambda. This avoids calling radial_comoving_distance
        # inside the lambda, preventing jax_cosmo workspace tracer leaks.
        N = n_integrate
        z_quad = jnp.linspace(min_z, max_z, N + 1)
        a_quad = jc.utils.z2a(z_quad)
        chi_s_quad = _radial_comoving_distance_safe(cosmo, a_quad)

        # Precompute the per-plane contributions (independent of source z)
        constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
        n_planes = len(r_center)
        r_b = r_center.reshape(n_planes, 1)
        a_b = scale_factors.reshape(n_planes, 1)

        mean_axes = tuple(range(1, lightcone.array.ndim))
        rho_mean = jnp.mean(lightcone.array, axis=mean_axes, keepdims=True)
        eps = jnp.finfo(rho_mean.dtype).eps
        safe_rho_mean = jnp.where(rho_mean == 0, eps, rho_mean)
        delta = lightcone.array / safe_rho_mean - 1

        # kappa_contributions: (n_planes, npix)
        kappa_contributions = delta * (density_plane_width * r_b / a_b)
        kappa_contributions *= constant_factor

        # lensing_efficiency: (n_z, n_planes, 1) for broadcasting with (n_planes, npix)
        chi_s_for_eff = chi_s_quad.reshape(-1, 1, 1)
        r_for_eff = r_center.reshape(1, -1, 1)
        lensing_efficiency = jnp.clip(1.0 - (r_for_eff / chi_s_for_eff), 0, 1000)

        # convergence_per_z: (n_z, npix) = sum over planes of lensing_eff * kappa
        convergence_per_z = jnp.sum(lensing_efficiency * kappa_contributions[None, :, :], axis=1)

        # Now integrate over z for each source distribution using Simpson's rule
        dz = (max_z - min_z) / N
        kappa_maps = []
        for nz in sources:
            nz_vals = nz(z_quad).reshape(-1, 1)  # (n_z, 1)
            integrand = nz_vals * convergence_per_z  # (n_z, npix)
            S = dz / 3 * jnp.sum(integrand[0:-1:2] + 4 * integrand[1::2] + integrand[2::2], axis=0)
            kappa_maps.append(S)

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
        N = n_integrate
        z_quad = jnp.linspace(min_z, max_z, N + 1)
        a_quad = jc.utils.z2a(z_quad)
        chi_s_quad = _radial_comoving_distance_safe(cosmo, a_quad)

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
                    chi_s=_radial_comoving_distance_safe(cosmo, jc.utils.z2a(z)),
                    pixel_size=pixel_size,
                    field_size=field_size_tuple,
                )

            kappa_maps.append(simps(integrand_fn, min_z, max_z, N=n_integrate))
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
