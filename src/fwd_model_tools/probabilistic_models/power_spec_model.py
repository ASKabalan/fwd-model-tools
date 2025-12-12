from itertools import combinations_with_replacement

import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import numpyro
import numpyro.distributions as dist

from .config import Configurations


def compute_cl_from_convergence_map(kappas, lmax):
    nb_bins = len(kappas)
    pair_order = sorted(combinations_with_replacement(range(nb_bins), 2))
    observed_cls = {}

    for i, j in pair_order:
        kappa_i = np.asarray(kappas[f"kappa_{i}"])
        kappa_j = np.asarray(kappas[f"kappa_{j}"])

        cl_full = hp.anafast(kappa_i, kappa_j, lmax=lmax)
        cl_obs = cl_full[2:]

        entry_name = f"C_ell_auto_{i}" if i == j else f"C_ell_cross_{i}_{j}"
        observed_cls[entry_name] = jnp.array(cl_obs)

    return observed_cls


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
    W_l = (jnp.sinc(ell * pixel_size_rad / (2 * jnp.pi)))**2
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


def powerspec_probmodel(
    config: Configurations,
    *,
    pixel_size_arcmin: float | None = None,
    nside: int | None = None,
):
    """
    Create NumPyro probabilistic model for power spectrum inference.

    Parameters
    ----------
    config : Configurations
        Configuration object containing priors, nz_shear, sigma_e, etc.
    pixel_size_arcmin : float, optional
        Pixel size in arcminutes for flat-sky maps. Required when
        ``config.geometry == 'flat'``.
    nside : int, optional
        HEALPix nside resolution for spherical maps. Required when
        ``config.geometry == 'spherical'``.

    Returns
    -------
    callable
        NumPyro model function with signature: model(ell, kappa_obs_spectra) -> observed_spectra
        where kappa_obs_spectra is a list of observed auto-spectra (one per redshift bin).
    """
    nb_bins = len(config.nz_shear)
    pair_order = sorted(combinations_with_replacement(range(nb_bins), 2))

    def model():
        Omega_c = numpyro.sample("Omega_c", config.priors["Omega_c"])
        sigma8 = numpyro.sample("sigma8", config.priors["sigma8"])

        cosmo = jc.Cosmology(
            Omega_c=Omega_c,
            Omega_b=config.fiducial_cosmology().Omega_b,
            h=config.fiducial_cosmology().h,
            n_s=config.fiducial_cosmology().n_s,
            sigma8=sigma8,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
        )
        cosmo._workspace = {}

        if config.geometry == "flat":
            if pixel_size_arcmin is None:
                raise ValueError("pixel_size_arcmin must be provided for flat geometry")
            pixel_scale = pixel_size_arcmin
        else:
            if nside is None:
                raise ValueError("nside must be provided for spherical geometry")
            pixel_scale = jnp.sqrt(4 * jnp.pi / (12 * (nside**2))) * (180.0 * 60.0 / jnp.pi)

        forward_model = make_2pt_model(pixel_scale, config.ells, sigma_e=config.sigma_e)

        cell_theory, cell_noise = forward_model(cosmo, config.nz_shear)

        observed_spectra = []
        for idx, (i, j) in enumerate(pair_order):
            if i == j:
                kappa_obs_spectra = numpyro.sample(f"C_ell_auto_{i}",
                                                   dist.Normal(cell_theory[idx], jnp.sqrt(cell_noise[idx])))

                observed_spectra.append(kappa_obs_spectra)
            # Deactivate cross-spectra for now
            # Because the noise is equal to 0 And I don't know what to sample
            elif False:
                kappa_obs_spectra = numpyro.sample(f"C_ell_cross_{i}_{j}",
                                                   dist.Normal(cell_theory[idx], jnp.sqrt(cell_noise[idx])))
                observed_spectra.append(kappa_obs_spectra)

        return observed_spectra

    return model
