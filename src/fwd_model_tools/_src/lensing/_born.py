from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates
from jaxtyping import ArrayLike

from ...fields import FlatDensity, SphericalDensity
from ._normalize_nz import _normalize_sources

__all__ = ["_born_core_impl", "_born_spherical", "_born_flat"]


def _simps_weights(a: float, b: float, N: int):
    """Simpson's rule quadrature weights for N subintervals (N+1 points).

    Reproduces the exact weights used by ``jax_cosmo.scipy.integrate.simps``:
    ``dx/3 * [1, 4, 2, 4, ..., 4, 1]``.
    """
    dx = (b - a) / N
    w = jnp.ones(N + 1)
    w = w.at[1:-1:2].set(4.0)  # odd indices → weight 4
    w = w.at[2:-2:2].set(2.0)  # interior even indices → weight 2
    return w * (dx / 3.0)  # shape (N+1,)


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
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
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
            if isinstance(field_size, (tuple | list | ArrayLike)):
                fx, fy = field_size
            else:
                fx = fy = field_size
            xgrid, ygrid = jnp.meshgrid(
                jnp.linspace(0, fx, nx, endpoint=False),
                jnp.linspace(0, fy, ny, endpoint=False),
            )
            coords = jnp.stack([xgrid, ygrid], axis=0) * (jnp.pi / 180)

    r_b = r.reshape(n_planes, *((1,) * (density_planes.ndim - 1)))
    a_b = a.reshape(n_planes, *((1,) * (density_planes.ndim - 1)))

    mean_axes = tuple(range(1, density_planes.ndim))
    rho_mean = jnp.mean(density_planes, axis=mean_axes, keepdims=True)
    eps = jnp.finfo(rho_mean.dtype).eps
    safe_rho_mean = jnp.where(rho_mean == 0, eps, rho_mean)
    delta = density_planes / safe_rho_mean - 1

    kappa_contributions = delta * (d_r * r_b / a_b)
    kappa_contributions *= constant_factor

    if not is_spherical:

        def interpolate_plane(delta_plane, chi_plane):
            assert pixel_size is not None and coords is not None, "pixel_size and coords are required for interpolation"
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
        # Evaluate _born_core_impl once for all quadrature points (shape: n_z, npix),
        # then apply per-distribution n(z) weights via a manual Simpson's rule.
        z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)  # (n_z,)
        kappa_grid = _born_core_impl(
            cosmo,
            lightcone.array,
            r_center,
            scale_factors,
            z_grid,
            density_plane_width,
        )  # (n_z, npix)
        weights = _simps_weights(min_z, max_z, n_integrate)  # (n_z,)
        nz_weights = jnp.stack([nz(z_grid) for nz in sources], axis=0)  # (K, n_z)
        kappa_maps = jnp.einsum("kz,z,zp->kp", nz_weights, weights, kappa_grid)  # (K, npix)
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
    if isinstance(field_size, (tuple | list | ArrayLike)):
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
        # Evaluate _born_core_impl once for all quadrature points (shape: n_z, ny, nx),
        # then apply per-distribution n(z) weights via a manual Simpson's rule.
        z_grid = jnp.linspace(min_z, max_z, n_integrate + 1)  # (n_z,)
        kappa_grid = _born_core_impl(
            cosmo,
            lightcone.array,
            r_center,
            scale_factors,
            z_grid,
            density_plane_width,
            pixel_size=pixel_size,
            field_size=field_size_tuple,
        )  # (n_z, ny, nx)
        weights = _simps_weights(min_z, max_z, n_integrate)  # (n_z,)
        nz_weights = jnp.stack([nz(z_grid) for nz in sources], axis=0)  # (K, n_z)
        kappa_maps = jnp.einsum("kz,z,zyx->kyx", nz_weights, weights, kappa_grid)  # (K, ny, nx)
    else:
        kappa_maps = _born_core_impl(
            cosmo,
            lightcone.array,
            r_center,
            scale_factors,
            sources,
            density_plane_width,
            pixel_size=pixel_size,
            field_size=field_size_tuple,
        )

    return kappa_maps
