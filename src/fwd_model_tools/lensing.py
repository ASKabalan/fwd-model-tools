"""
Weak lensing computations using Born approximation.

This module provides functions for computing lensing convergence from
density lightcones in both spherical (HEALPix) and flat-sky geometries.
"""

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates
from jax_cosmo.scipy.integrate import simps

from fwd_model_tools.field import FlatDensity, SphericalDensity, DensityStatus
from fwd_model_tools.kappa import FlatKappaField, SphericalKappaField


def convergence_Born(
    cosmo,
    density_planes,
    r,
    a,
    z_source,
    d_r,
    dx=None,
    coords=None,
    field_size=None,
):
    """
    Born approximation convergence for both spherical and flat geometries.

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology object
    density_planes : ndarray
        - Spherical: [n_planes, npix] - density on HEALPix grid
        - Flat: [n_planes, nx, ny] - density on Cartesian grid
        Note: d_R is already included in the density normalization
    r : ndarray
        Comoving distances to plane centers [n_planes]
    a : ndarray
        Scale factors at plane centers [n_planes]
    z_source : float or ndarray
        Source redshift(s)
    d_r : float
        Density plane width (Mpc/h)
    dx : float, optional
        Pixel size for flat-sky case (required for flat if coords not provided)
    coords : ndarray, optional
        Angular coordinates for flat-sky [2, ny, nx] in radians.
        If None and flat geometry, will be generated from field_size.
    field_size : float, optional
        Field of view in degrees for flat-sky coordinate generation.
        Required if coords is None for flat geometry.

    Returns
    -------
    convergence : ndarray
        Convergence map

    Examples
    --------
    Spherical geometry:

    >>> kappa = convergence_Born(cosmo, planes, r, a, z_source=1.0, d_r=100.0)

    Flat geometry with auto-generated coordinates:

    >>> kappa = convergence_Born(cosmo, planes, r, a, z_source=1.0, d_r=100.0,
    ...                          dx=pixel_size, field_size=10.0)
    """
    # Constants
    # --- 1. Pre-computation and Shape Setup ---
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    chi_s = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_source))
    n_planes = len(r)

    # Detect geometry from input shape
    is_spherical = density_planes.ndim == 2  # [n_planes, npix]

    if not is_spherical:
        # Flat geometry - need dx and either coords or field_size
        assert dx is not None, "dx is required for flat geometry"

        if coords is None:
            # Auto-generate coordinates
            assert (
                field_size is not None
            ), "field_size is required for flat geometry when coords not provided"
            ny, nx = density_planes.shape[-2:]
            xgrid, ygrid = jnp.meshgrid(
                jnp.linspace(0, field_size, nx, endpoint=False),
                jnp.linspace(0, field_size, ny, endpoint=False),
            )
            coords = jnp.stack([xgrid, ygrid], axis=0) * (jnp.pi / 180)

    # Reshape 1D arrays to [n_planes, 1, 1] for broadcasting with [n_planes, nx, ny]
    # Or to [n_planes, 1] for spherical geometry
    r_b = r.reshape(n_planes, *((1,) * (density_planes.ndim - 1)))
    a_b = a.reshape(n_planes, *((1,) * (density_planes.ndim - 1)))

    # --- 2. Vectorized Overdensity Calculation ---
    # Calculate mean density across spatial dimensions for each plane
    mean_axes = tuple(range(1, density_planes.ndim))
    rho_mean = jnp.mean(density_planes, axis=mean_axes, keepdims=True)
    # Avoid division by zero by adding a small epsilon where mean density is zero
    eps = jnp.finfo(rho_mean.dtype).eps
    safe_rho_mean = jnp.where(rho_mean == 0, eps, rho_mean)
    delta = density_planes / safe_rho_mean - 1

    # --- 3. Vectorized Lensing Kernel and Weighting ---
    # Combine all factors except interpolation
    # This includes the geometric term: dχ * χ / a(χ)
    kappa_contributions = delta * (d_r * r_b / a_b)
    kappa_contributions *= constant_factor
    # --- 4. Interpolation (for Flat-Sky only) ---
    if not is_spherical:
        # Define the interpolation function for a SINGLE plane
        def interpolate_plane(delta_plane, chi_plane):
            physical_coords = coords * chi_plane / dx
            return map_coordinates(
                delta_plane, physical_coords - 0.5, order=1, mode="wrap"
            )

        # Use vmap to apply the function across all planes efficiently
        kappa_contributions = jax.vmap(interpolate_plane)(kappa_contributions, r)

    # --- 5. Final Assembly ---
    # In case of multiple source redshifts, and a flat sky approximation,
    # We need to add a dimension to match the 2D shape of the kappa contributions
    if jnp.ndim(z_source) > 0 and not is_spherical:
        chi_s = jnp.expand_dims(chi_s, axis=1)
    # Apply the constant factor and the lensing efficiency kernel: (χs - χ) / χs
    lensing_efficiency = jnp.clip(1.0 - (r_b / chi_s), 0, 1000)
    # Add a dimension for broadcasting the redshift dimension
    lensing_efficiency = jnp.expand_dims(lensing_efficiency, axis=-1)
    kappa_contributions = jnp.expand_dims(kappa_contributions, axis=1)
    # Multiply the weighted delta by the lensing kernel and constant
    final_contributions = lensing_efficiency * kappa_contributions

    # Sum contributions from all planes to get the final convergence map
    # For multiple redshifts, preserve the redshift dimension
    convergence = jnp.sum(final_contributions, axis=0)

    # Handle single vs multiple redshift cases
    if jnp.ndim(z_source) == 0:  # Single redshift case
        convergence = jnp.squeeze(convergence, axis=0)

    return convergence


def _compute_born_spherical(
    cosmo,
    lightcone,
    r_center,
    scale_factors,
    z_source,
    density_plane_width,
    nz_shear,
    min_z,
    max_z,
    N_integrate,
):
    """
    Compute convergence for spherical (HEALPix) geometry.

    Returns kappa_array.
    """
    if nz_shear is not None:
        # Integrate over source distribution(s)
        nz_list = nz_shear if isinstance(nz_shear, list) else [nz_shear]

        kappa_maps = [
            simps(
                lambda z: nz(z).reshape([-1, 1])
                * convergence_Born(
                    cosmo,
                    lightcone.array,
                    r_center,
                    scale_factors,
                    z,
                    density_plane_width,
                ),
                min_z,
                max_z,
                N=N_integrate,
            )
            for nz in nz_list
        ]

        kappa_array = (
            jnp.stack(kappa_maps, axis=0) if len(kappa_maps) > 1 else kappa_maps[0]
        )
    else:
        # Direct evaluation at z_source
        kappa_array = convergence_Born(
            cosmo,
            lightcone.array,
            r_center,
            scale_factors,
            z_source,
            density_plane_width,
        )

    return kappa_array


def _compute_born_flat(
    cosmo,
    lightcone,
    r_center,
    scale_factors,
    z_source,
    density_plane_width,
    nz_shear,
    min_z,
    max_z,
    N_integrate,
):
    """
    Compute convergence for flat-sky (Cartesian) geometry.

    Returns kappa_array.
    """
    # Validate field_size
    if lightcone.field_size is None:
        raise ValueError(
            "field_size is required in lightcone for flat-sky convergence computation"
        )

    dx = lightcone.mesh_size[0] / lightcone.flatsky_npix[0]
    field_size = lightcone.field_size

    if nz_shear is not None:
        # Integrate over source distribution(s)
        nz_list = nz_shear if isinstance(nz_shear, list) else [nz_shear]

        kappa_maps = [
            simps(
                lambda z: nz(z).reshape([-1, 1, 1])
                * convergence_Born(
                    cosmo,
                    lightcone.array,
                    r_center,
                    scale_factors,
                    z,
                    density_plane_width,
                    dx=dx,
                    field_size=field_size,
                ),
                min_z,
                max_z,
                N=N_integrate,
            )
            for nz in nz_list
        ]

        kappa_array = (
            jnp.stack(kappa_maps, axis=0) if len(kappa_maps) > 1 else kappa_maps[0]
        )
    else:
        # Direct evaluation at z_source
        kappa_array = convergence_Born(
            cosmo,
            lightcone.array,
            r_center,
            scale_factors,
            z_source,
            density_plane_width,
            dx=dx,
            field_size=field_size,
        )

    return kappa_array


def born(
    cosmo,
    lightcone,
    z_source,
    nz_shear=None,
    min_z=0.01,
    max_z=3.0,
    N_integrate=32,
):
    """
    Compute lensing convergence using Born approximation.

    Functional API similar to: lpt(cosmo, density_field, a, order)

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object (NOT stored in output)
    lightcone : FlatDensity | SphericalDensity
        Multi-plane lightcone with status=LIGHTCONE.
        Must have array shape (n_planes, ...) and scale_factors (n_planes,)
    z_source : float | jnp.ndarray
        Source redshift(s) - evaluation centers for convergence.
        ALWAYS REQUIRED even with nz_shear.
    nz_shear : callable | list[callable] | None
        Optional source distributions for integration.
        Integrates nz(z) over [min_z, max_z] around z_source centers.
    min_z : float, default=0.01
        Minimum redshift for integration
    max_z : float, default=3.0
        Maximum redshift for integration
    N_integrate : int, default=32
        Number of integration points

    Returns
    -------
    FlatKappaField | SphericalKappaField
        Convergence map(s) - NO cosmology stored

    Examples
    --------
    Direct evaluation (no integration):

    >>> kappa = born(cosmo, lightcone, z_source=1.0)

    Multiple sources:

    >>> kappa = born(cosmo, lightcone, z_source=jnp.array([0.5, 1.0, 1.5]))

    With nz integration (z_source = bin centers):

    >>> kappa = born(cosmo, lightcone, z_source=jnp.array([0.7, 1.2, 1.8]),
    ...              nz_shear=[nz1, nz2, nz3])
    """
    # Validate lightcone
    if lightcone.status != DensityStatus.LIGHTCONE:
        raise ValueError(
            f"Expected lightcone with status=LIGHTCONE, got {lightcone.status}"
        )

    # Extract metadata
    scale_factors = jnp.atleast_1d(lightcone.scale_factors)
    n_planes = scale_factors.size

    if lightcone.array.ndim not in [2, 3]:
        raise ValueError(
            f"Lightcone array must be 2D (spherical) or 3D (flat), got {lightcone.array.ndim}D"
        )

    # Compute comoving distances from scale factors
    r_center = jc.background.radial_comoving_distance(cosmo, scale_factors)
    cosmo._workspace = {}  # Clear cache

    # Compute density_plane_width from max_radius
    max_radius = lightcone.max_comoving_radius
    density_plane_width = max_radius / n_planes

    # Determine geometry and compute convergence
    is_spherical = isinstance(lightcone, SphericalDensity)

    if is_spherical:
        kappa_array = _compute_born_spherical(
            cosmo,
            lightcone,
            r_center,
            scale_factors,
            z_source,
            density_plane_width,
            nz_shear,
            min_z,
            max_z,
            N_integrate,
        )
    else:
        kappa_array = _compute_born_flat(
            cosmo,
            lightcone,
            r_center,
            scale_factors,
            z_source,
            density_plane_width,
            nz_shear,
            min_z,
            max_z,
            N_integrate,
        )

    # Create appropriate KappaField
    # Use z_source for scale_factors in the reference field
    z_source_arr = (
        z_source if jnp.ndim(z_source) == 0 else jnp.atleast_1d(z_source)
    )
    ref_field = lightcone.replace(scale_factors=z_source_arr)

    if is_spherical:
        return SphericalKappaField(
            array=kappa_array,
            density_field=ref_field,
            status=DensityStatus.KAPPA,
            z_source=z_source,
        )
    else:
        return FlatKappaField(
            array=kappa_array,
            density_field=ref_field,
            status=DensityStatus.KAPPA,
            z_source=z_source,
        )

