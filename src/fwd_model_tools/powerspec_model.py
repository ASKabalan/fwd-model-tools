import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist


def pixel_window_function(l, pixel_size_arcmin):
    """
    Calculate the pixel window function W_l for a given angular wave number l and pixel size.

    Parameters
    ----------
    l : array_like
        Angular wave number.
    pixel_size_arcmin : float
        Pixel size in arcminutes.

    Returns
    -------
    W_l : array_like
        Pixel window function.
    """
    pixel_size_rad = pixel_size_arcmin * (jnp.pi / (180.0 * 60.0))
    W_l = (jnp.sinc(l * pixel_size_rad / (2 * jnp.pi)))**2
    return W_l


def make_2pt_model(pixel_scale, ell, sigma_e=0.3):
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
        cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [tracer], nonlinear_fn=jc.power.linear)
        cell_theory = cell_theory * pixel_window_function(ell, pixel_scale)
        cell_noise = jc.angular_cl.noise_cl(ell, [tracer])
        return cell_theory, cell_noise

    return forward_model


def powerspec_probmodel(config, ell, kappa_obs_spectra):
    """
    NumPyro probabilistic model for power spectrum inference.

    Parameters
    ----------
    config : Configurations
        Configuration object containing priors, nz_shear, sigma_e, etc.
    ell : array_like
        Angular wave numbers for power spectrum.
    kappa_obs_spectra : dict
        Dictionary mapping (i, j) tuples to observed power spectra arrays.
        Keys are (i, j) where i <= j are redshift bin indices.

    Returns
    -------
    None
        NumPyro model that samples cosmological parameters and conditions on observed power spectra.
    """
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

    nbins = len(config.nz_shear)
    pixel_scale = config.field_size * 60.0 / config.field_npix

    tracers = [jc.probes.WeakLensing([nz], sigma_e=config.sigma_e) for nz in config.nz_shear]

    for i in range(nbins):
        for j in range(i, nbins):
            cell_theory = jc.angular_cl.angular_cl(
                cosmo, ell, [tracers[i], tracers[j]], nonlinear_fn=jc.power.linear
            )
            cell_theory = cell_theory[0] * pixel_window_function(ell, pixel_scale)

            cell_noise_i = jc.angular_cl.noise_cl(ell, [tracers[i]])[0]
            cell_noise_j = jc.angular_cl.noise_cl(ell, [tracers[j]])[0]

            if i == j:
                cell_total = cell_theory + cell_noise_i
            else:
                cell_total = cell_theory

            sigma_cl = jnp.sqrt(2.0 / (2.0 * ell + 1.0)) * cell_total

            numpyro.sample(
                f"C_ell_{i}_{j}",
                dist.Normal(cell_theory, sigma_cl),
                obs=kappa_obs_spectra[(i, j)],
            )
