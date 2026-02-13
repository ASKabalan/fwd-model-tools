from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import jax
import jax.core
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from scipy.integrate import simpson

from ..fields import SphericalDensity

try:
    from dorian.lensing import raytrace_from_density
except ImportError:
    raytrace_from_density = None

from ..fields import FieldStatus, SphericalKappaField

__all__ = ["raytrace"]

# Type variables for decorator
Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")


def require_dorian(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Decorator that raises ImportError if datasets library is not available."""
    try:
        from dorian.lensing import raytrace_from_density  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 'dorian'. Install with: pip install fwd-model-tools[raytrace]"
        raise ImportError(msg)

    return deferred_func


def _normalize_sources(nz_shear: Any) -> tuple[str, list[Any]]:
    """Accept either scalar redshifts or jc.redshift distributions (not both).

    Parameters
    ----------
    nz_shear : Any
        Either a scalar redshift, array of redshifts, or jax_cosmo redshift
        distribution(s).

    Returns
    -------
    tuple[str, list[Any]]
        A tuple of (source_kind, sources) where source_kind is either
        "distribution" or "redshift", and sources is the normalized list.
    """
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

    z_array = np.atleast_1d(np.asarray(entries)).flatten()
    return "redshift", z_array


def _raytrace_z_grid(
    density_maps: np.ndarray,
    shell_redshifts: np.ndarray,
    z_sources: np.ndarray,
    box_size: float,
    n_particles: int,
    omega_m: float,
    h: float,
    omega_l: float,
    nside: int,
    interp: str,
    parallel_transport: bool,
    born: bool,
) -> np.ndarray:
    """Run dorian ray-tracing for each source redshift.

    Parameters
    ----------
    density_maps : np.ndarray
        HEALPix density maps, shape (n_shells, npix).
    shell_redshifts : np.ndarray
        Redshift of each shell.
    z_sources : np.ndarray
        Source redshifts to compute convergence for.
    box_size : float
        Simulation box size in Mpc/h.
    n_particles : int
        Total number of particles in the simulation.
    omega_m : float
        Matter density parameter.
    h : float
        Dimensionless Hubble parameter.
    omega_l : float
        Dark energy density parameter.
    nside : int
        HEALPix NSIDE parameter.
    interp : str
        Interpolation method ('bilinear', 'ngp', 'nufft').
    parallel_transport : bool
        Whether to apply parallel transport of distortion matrix.

    Returns
    -------
    np.ndarray
        Convergence maps, shape (n_sources, npix).
    """
    kappa_list = []
    for z_s in z_sources:
        result = raytrace_from_density(
            density_maps=list(density_maps),
            redshifts=list(shell_redshifts),
            z_source=float(z_s),
            box_size=box_size,
            n_particles=n_particles,
            omega_m=omega_m,
            h=h,
            omega_l=omega_l,
            nside=nside,
            interp=interp,
            parallel_transport=parallel_transport,
        )
        res = result["convergence_born" if born else "convergence_raytraced"]
        kappa_list.append(res)
    return np.stack(kappa_list, axis=0)


def _integrate_nz(
    kappa_grid: np.ndarray,
    z_grid: np.ndarray,
    nz_distributions: list,
) -> np.ndarray:
    """Integrate kappa_grid weighted by nz distributions.

    Parameters
    ----------
    kappa_grid : np.ndarray
        Pre-computed convergence at each z, shape (n_z, npix).
    z_grid : np.ndarray
        Redshift sample points, shape (n_z,).
    nz_distributions : list
        List of jax_cosmo redshift distribution objects.

    Returns
    -------
    np.ndarray
        Integrated convergence maps, shape (n_distributions, npix).
    """
    kappa_list = []
    for nz in nz_distributions:
        # Evaluate nz(z) at all grid points
        nz_of_ = nz(next(iter(z_grid)))
        nz_weights = np.array(nz(z_grid))  # Shape (n_z,)
        # Weight kappa by nz: (n_z, npix) * (n_z, 1) -> (n_z, npix)
        weighted_kappa = kappa_grid * nz_weights[:, None]
        # Integrate over z using Simpson's rule
        integrated_kappa = simpson(weighted_kappa, x=z_grid, axis=0)
        kappa_list.append(integrated_kappa)
    return np.stack(kappa_list, axis=0)


@require_dorian
def raytrace(
    cosmo,
    lightcone,
    nz_shear,
    min_z=0.01,
    max_z=3.0,
    n_integrate=32,
    interp="bilinear",
    parallel_transport=True,
    born=False,
):
    """Multi-plane ray-tracing using dorian.

    Computes weak lensing convergence maps by propagating light rays through
    a series of mass shells, including post-Born corrections from the full
    distortion matrix.

    This function wraps dorian's ray-tracing and provides the same API as
    :func:`fwd_model_tools.lensing.born`.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object containing Omega_m, h, etc.
    lightcone : SphericalDensity
        HEALPix lightcone with status=LIGHTCONE containing density shells.
    nz_shear : float, array-like, or jc.redshift.redshift_distribution
        Source redshift(s) or redshift distribution(s). Can be:
        - A single scalar redshift
        - An array of scalar redshifts
        - A jax_cosmo redshift distribution object
        - A list of redshift distribution objects
    min_z : float, optional
        Minimum redshift for integration (only used for distributions).
        Default: 0.01.
    max_z : float, optional
        Maximum redshift for integration (only used for distributions).
        Default: 3.0.
    n_integrate : int, optional
        Number of integration points for Simpson's rule (only used for
        distributions). Default: 32.
    interp : {'bilinear', 'ngp', 'nufft'}, optional
        Interpolation method for sampling deflection fields at ray positions:
        - 'bilinear': Bilinear interpolation (default, good balance)
        - 'ngp': Nearest grid point (fastest, lowest accuracy)
        - 'nufft': Non-uniform FFT (highest accuracy, slowest)
    parallel_transport : bool, optional
        Whether to parallel transport the distortion matrix along geodesics.
        Default: True.

    Returns
    -------
    SphericalKappaField
        Convergence maps with shape (n_sources, npix) if multiple sources,
        or (npix,) if single source.

    Raises
    ------
    ValueError
        If called inside jit (requires concrete arrays).
        If lightcone status is not LIGHTCONE.
    TypeError
        If lightcone is not SphericalDensity (flat-sky not supported).

    Notes
    -----
    Unlike :func:`born`, this function cannot be JIT-compiled because dorian
    uses NumPy/healpy internally. The function will raise an error if called
    with traced (non-concrete) arrays.

    The convergence is computed from the full ray-traced distortion matrix A:
    ``kappa = 1 - 0.5 * (A[0,0] + A[1,1])``, which includes all post-Born
    corrections.

    Examples
    --------
    >>> import jax_cosmo as jc
    >>> from fwd_model_tools.lensing import raytrace
    >>> # With scalar redshifts
    >>> kappa = raytrace(cosmo, lightcone, nz_shear=[0.5, 1.0, 1.5])
    >>> kappa.array.shape  # (3, npix)
    >>> # With redshift distribution
    >>> nz = jc.redshift.smail_nz(1.0, 2.0, 1.0)
    >>> kappa = raytrace(cosmo, lightcone, nz_shear=nz)

    See Also
    --------
    born : Born approximation (faster, JAX-compatible, no post-Born corrections).
    """
    # 1. Concrete check - cannot run inside jit
    if not jax.core.is_concrete(lightcone.array):
        raise ValueError("raytrace() requires concrete arrays. Use outside of jit.")

    # 2. Validate input
    if lightcone.status != FieldStatus.LIGHTCONE:
        raise ValueError(f"Expected lightcone with status=LIGHTCONE, got {lightcone.status}")
    if not isinstance(lightcone, SphericalDensity):
        raise TypeError("raytrace() only supports SphericalDensity (HEALPix) lightcones")

    # 3. Extract cosmology from jax_cosmo
    omega_m = float(cosmo.Omega_m)
    h = float(cosmo.h)
    omega_l = 1.0 - omega_m  # Assuming flat cosmology

    # 4. Extract simulation parameters from lightcone
    box_size = lightcone.box_size[0]  # Assumes cubic box in Mpc/h
    n_particles = int(np.prod(lightcone.mesh_size))
    nside = lightcone.nside

    # 5. Convert JAX arrays to numpy
    density_maps = np.asarray(lightcone.array)
    scale_factors = np.atleast_1d(np.asarray(lightcone.scale_factors))
    shell_redshifts = 1.0 / scale_factors - 1.0

    # Convert density (N/V) to particle counts (N) for Dorian
    # Dorian's prepare_density_shells(unit='number_density') expects counts per pixel
    if lightcone.comoving_centers is None or lightcone.density_width is None:
        raise ValueError("Lightcone must have comoving_centers and density_width for raytracing")

    comoving_centers = np.atleast_1d(np.asarray(lightcone.comoving_centers))
    density_widths = np.atleast_1d(np.asarray(lightcone.density_width))

    npix = 12 * nside**2
    # Calculate pixel volume for each shell
    # V_pix = (Omega_pix / 4pi) * V_shell = (1/npix) * (4/3 pi (Rmax^3 - Rmin^3))
    # Note: density_maps is (n_shells, npix).

    counts_maps = []
    for i in range(len(density_maps)):
        r = comoving_centers[i]
        dr = density_widths[i]
        rmax = r + dr / 2.0
        rmin = max(0.0, r - dr / 2.0)
        shell_vol = (4.0 / 3.0) * np.pi * (rmax**3 - rmin**3)
        pixel_vol = shell_vol / npix

        # density is N / V, so N = density * V
        counts_maps.append(density_maps[i] * pixel_vol)

    density_maps = np.stack(counts_maps)

    # 6. Normalize nz_shear sources
    source_kind, sources = _normalize_sources(nz_shear)

    # 7. Run ray-tracing
    if source_kind == "distribution":
        # For distributions: compute kappa on z_grid, then integrate with nz weights
        z_grid = np.linspace(min_z, max_z, n_integrate + 1)
        kappa_grid = _raytrace_z_grid(
            density_maps,
            shell_redshifts,
            z_grid,
            box_size,
            n_particles,
            omega_m,
            h,
            omega_l,
            nside,
            interp,
            parallel_transport,
            born=born,
        )
        kappa_maps = _integrate_nz(kappa_grid, z_grid, sources)
    else:
        # For scalar redshifts: compute kappa directly at each z
        kappa_maps = _raytrace_z_grid(
            density_maps,
            shell_redshifts,
            sources,
            box_size,
            n_particles,
            omega_m,
            h,
            omega_l,
            nside,
            interp,
            parallel_transport,
            born=born,
        )

    # Handle single source case - squeeze to 1D
    if kappa_maps.shape[0] == 1:
        kappa_maps = kappa_maps[0]

    # 8. Return SphericalKappaField
    base_field = lightcone.replace(status=FieldStatus.KAPPA)
    return SphericalKappaField.FromDensityMetadata(
        array=jnp.asarray(kappa_maps),
        field=base_field,
        status=FieldStatus.KAPPA,
        z_sources=nz_shear,
    )
