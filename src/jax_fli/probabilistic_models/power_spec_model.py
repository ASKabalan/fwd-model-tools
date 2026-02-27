from __future__ import annotations

from collections.abc import Callable

import jax
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


def make_2pt_model(config, nonlinear_fn=jc.power.halofit):
    """
    Create a function that computes the windowed theoretical Cl signal
    and noise Cl separately.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcminutes.
    ell : array_like
        Angular wave number.
    sigma_e : float, optional
        Intrinsic ellipticity dispersion per component. Default is 0.3.
    nonlinear_fn : callable, optional
        Nonlinear power spectrum function. Default is jc.power.halofit.

    Returns
    -------
    forward_model : callable
        Function (cosmo, nz_shear) -> (cell_theory, cell_noise) where
        cell_theory is the pixel-windowed signal with shape (N_pairs, N_ell)
        and cell_noise is the noise with shape (N_pairs, N_ell).
    """
    # Derive pixel scale from config (static metadata)
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

    def forward_model(cosmo):
        tracer = jc.probes.WeakLensing(config.nz_shear, sigma_e=config.sigma_e)
        cell_theory = jc.angular_cl.angular_cl(cosmo, config.ells, [tracer], nonlinear_fn=nonlinear_fn)
        cell_theory = cell_theory * pixel_window_function(config.ells, pixel_scale)
        cell_noise = jc.angular_cl.noise_cl(config.ells, [tracer])
        return cell_theory, cell_noise

    return jax.jit(forward_model)


def powerspec_probmodel(config: Configurations, nonlinear_fn=jc.power.halofit) -> Callable:
    """
    Create NumPyro probabilistic model for power spectrum inference.

    Uses a batched MultivariateNormal likelihood with the full Gaussian
    covariance from jax_cosmo. At each multipole ell, the covariance
    captures correlations across all tomographic bin pairs (auto + cross),
    which is essential for correct posterior inference in tomographic
    analyses.

    The observable mean is ``cell_theory + cell_noise``, matching the
    ``cross_angular_cl`` measured from simulated kappa maps.

    Pixel scale is derived automatically from config:
    - Spherical geometry: from config.nside
    - Flat geometry: from config.field_size and config.flatsky_npix

    Cosmological parameters are sampled dynamically from config.priors.

    Parameters
    ----------
    config : Configurations
        Configuration object. Must have geometry-appropriate resolution metadata
        (nside for spherical, flatsky_npix + field_size for flat).
    nonlinear_fn : callable, optional
        Nonlinear power spectrum function. Default is jc.power.halofit.

    Returns
    -------
    callable
        NumPyro model function with no arguments that registers a single
        C_ell sample site with shape (N_ell, N_pairs).
    """

    forward_model = make_2pt_model(config, nonlinear_fn=nonlinear_fn)

    fiducial_cell_theory, fiducial_cell_noise = forward_model(config.fiducial_cosmology())

    tracer = jc.probes.WeakLensing(config.nz_shear, sigma_e=config.sigma_e)
    C = jc.angular_cl.gaussian_cl_covariance(
        config.ells, [tracer], fiducial_cell_theory, fiducial_cell_noise, f_sky=config.f_sky, sparse=True
    )
    C = jc.sparse.to_dense(C)
    L_matrix = jnp.linalg.cholesky(C + 1e-7 * jnp.eye(C.shape[0]))

    def model():
        cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, prior) for k, prior in config.priors.items()})
        numpyro.deterministic("cosmo", cosmo)

        cell_theory, cell_noise = forward_model(cosmo)

        # Observable = windowed signal + noise (matches cross_angular_cl from maps)
        cell_obs = cell_theory + cell_noise

        numpyro.sample(
            "C_ell",
            dist.MultivariateNormal(loc=cell_obs.flatten(), scale_tril=L_matrix),
        )

    return model
