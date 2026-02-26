from __future__ import annotations

from collections.abc import Callable
from itertools import combinations_with_replacement

import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist

from .config import Configurations


def pixel_window_function(ell, pixel_size_arcmin):
    """
    Calculate the pixel window function W_l for a given angular wave number l and pixel size.

    Parameters
    ----------
    ell : array_like
        Angular wave number.
    pixel_size_arcmin : float
        Pixel size in arcminutes.

    Returns
    -------
    W_l : array_like
        Pixel window function.
    """
    pixel_size_rad = pixel_size_arcmin * (jnp.pi / (180.0 * 60.0))
    W_l = (jnp.sinc(ell * pixel_size_rad / (2 * jnp.pi))) ** 2
    return W_l


def make_2pt_model(pixel_scale, ell, sigma_e=0.3, nonlinear_fn=jc.power.halofit):
    """
    Create a function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcminutes.
    ell : array_like
        Angular wave number (numpy array).
    sigma_e : float, optional
        Intrinsic ellipticity dispersion per component. Default is 0.3.

    Returns
    -------
    forward_model : callable
        Function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.
    """

    def forward_model(cosmo, nz_shear):
        tracer = jc.probes.WeakLensing(nz_shear, sigma_e=sigma_e)
        cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [tracer], nonlinear_fn=nonlinear_fn)
        cell_theory = cell_theory * pixel_window_function(ell, pixel_scale)
        cell_noise = jc.angular_cl.noise_cl(ell, [tracer])
        return cell_theory, cell_noise

    return forward_model


def powerspec_probmodel(config: Configurations) -> Callable:
    """
    Create NumPyro probabilistic model for power spectrum inference.

    The likelihood uses the Knox formula to account for both shape noise and
    cosmic variance:

    - Auto-spectra (i == j):
        Var = 2 * (C_ii + N_ii)^2 / ((2*ell + 1) * f_sky)
    - Cross-spectra (i != j, enabled when config.use_cross is True):
        Var = ((C_ii + N_ii) * (C_jj + N_jj) + C_ij^2) / ((2*ell + 1) * f_sky)

    Pixel scale is derived automatically from config:
    - Spherical geometry: from config.nside
    - Flat geometry: from config.field_size and config.flatsky_npix

    Cosmological parameters are sampled dynamically from config.priors,
    mirroring full_field_probmodel.

    Parameters
    ----------
    config : Configurations
        Configuration object. Must have geometry-appropriate resolution metadata
        (nside for spherical, flatsky_npix + field_size for flat).

    Returns
    -------
    callable
        NumPyro model function with no arguments that registers C_ell sample sites.
    """
    nb_bins = len(config.nz_shear)
    pair_order = sorted(combinations_with_replacement(range(nb_bins), 2))

    # Pre-build index map for O(1) auto-spectrum lookup (needed for cross-variance)
    pair_to_idx = {pair: idx for idx, pair in enumerate(pair_order)}

    # Derive pixel scale from config at model-creation time (static metadata)
    if config.geometry == "flat":
        if config.flatsky_npix is None or config.field_size is None:
            raise ValueError("flat geometry requires config.flatsky_npix and config.field_size")
        ny, nx = config.flatsky_npix
        size_y, size_x = config.field_size
        pixel_scale = 0.5 * (size_y * 60.0 / ny + size_x * 60.0 / nx)
    else:
        if config.nside is None:
            raise ValueError("spherical geometry requires config.nside")
        pixel_scale = jnp.sqrt(4 * jnp.pi / (12 * (config.nside**2))) * (180.0 * 60.0 / jnp.pi)

    forward_model = make_2pt_model(pixel_scale, config.ells, sigma_e=config.sigma_e)

    def model():
        cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})
        numpyro.deterministic("cosmo", cosmo)

        cell_theory, cell_noise = forward_model(cosmo, config.nz_shear)

        mode_count = (2 * config.ells + 1) * config.f_sky

        for idx, (i, j) in enumerate(pair_order):
            is_auto = i == j
            if not is_auto and not config.use_cross:
                continue

            cl_ij = cell_theory[idx]

            if is_auto:
                n_ii = cell_noise[idx]
                variance = 2.0 * (cl_ij + n_ii) ** 2 / mode_count
                name = f"C_ell_auto_{i}"
            else:
                idx_ii = pair_to_idx[(i, i)]
                idx_jj = pair_to_idx[(j, j)]
                cl_ii = cell_theory[idx_ii]
                cl_jj = cell_theory[idx_jj]
                n_ii = cell_noise[idx_ii]
                n_jj = cell_noise[idx_jj]
                variance = ((cl_ii + n_ii) * (cl_jj + n_jj) + cl_ij**2) / mode_count
                name = f"C_ell_cross_{i}_{j}"

            numpyro.sample(name, dist.Normal(cl_ij, jnp.sqrt(variance)))

    return model
