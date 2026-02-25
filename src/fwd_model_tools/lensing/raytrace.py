from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson

from ..fields import SphericalDensity

try:
    from dorian.lensing import raytrace_from_density
except ImportError:
    raytrace_from_density = None

from .._src.base import ConvergenceUnit
from .._src.lensing import _attach_source_metadata
from .._src.lensing._normalize_nz import _normalize_sources
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
    shell_widths: np.ndarray | None = None,
    nufft_threads: int = 4,
    n_workers=None,
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
    born : bool
        Whether to return Born approximation convergence.
    shell_widths : np.ndarray or None, optional
        Shell thickness d_R per shell in Mpc/h.

    Returns
    -------
    np.ndarray
        Convergence maps, shape (n_sources, npix).
    """
    assert (
        raytrace_from_density is not None
    ), "raytrace_from_density is None — dorian not installed. Use @require_dorian to guard callers."
    result = raytrace_from_density(
        density_maps=list(density_maps),
        redshifts=list(shell_redshifts),
        z_source=list(z_sources),
        box_size=box_size,
        n_particles=n_particles,
        omega_m=omega_m,
        h=h,
        omega_l=omega_l,
        nside=nside,
        interp=interp,
        shell_widths=list(shell_widths) if shell_widths is not None else None,
        parallel_transport=parallel_transport,
        n_workers=n_workers,
    )
    return result["convergence_born" if born else "convergence_raytraced"]


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
    # Dorian options
    interp="bilinear",
    parallel_transport=True,
    born=False,
    n_workers=None,
):
    """Multi-plane ray-tracing using dorian.

    Computes weak lensing convergence maps by propagating light rays through
    a series of mass shells, including post-Born corrections from the full
    distortion matrix.

    This function wraps dorian's ray-tracing and provides the same API as
    :func:`fwd_model_tools.lensing.born`. Unlike the previous version, this
    function is JIT-compatible via :func:`jax.pure_callback`.

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
        Whether to apply parallel transport of distortion matrix.
        Default: True.

    Returns
    -------
    SphericalKappaField
        Convergence maps with shape (n_sources, npix) if multiple sources,
        or (npix,) if single source.

    Raises
    ------
    ValueError
        If lightcone status is not LIGHTCONE.
    TypeError
        If lightcone is not SphericalDensity (flat-sky not supported).

    Notes
    -----
    The dorian computation is wrapped in :func:`jax.pure_callback`, making
    this function safe to call inside :func:`jax.jit`. JAX dispatches the
    callback once per device shard, passing a concrete numpy array.

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
    >>> # JIT-compatible
    >>> kappa = jax.jit(lambda lc: raytrace(cosmo, lc, [1.0]))(lightcone)

    See Also
    --------
    born : Born approximation (faster, fully-JAX, no post-Born corrections).
    """
    if jax.process_count() > 1:
        raise NotImplementedError(
            """
            raytrace() does not support multi-process execution.
            Run with a single process (e.g. --pdim 1 1) or use the born() function for multi-process compatibility.
            However it supports multi-processing by setting n_workers > 1, which uses multiprocessing within the dorian callback.
            It does not support the JAX multi-process model where the entire function is replicated across processes
            """
        )
    # 1. Validate on static metadata — safe inside JIT
    if lightcone.status != FieldStatus.LIGHTCONE:
        raise ValueError(f"Expected lightcone with status=LIGHTCONE, got {lightcone.status}")
    if not isinstance(lightcone, SphericalDensity):
        raise TypeError("raytrace() only supports SphericalDensity (HEALPix) lightcones")
    if lightcone.density_width is None:
        raise ValueError("Lightcone must have density_width for raytracing")
    density_width = lightcone.density_width
    assert density_width is not None  # type narrowing for checker + closures

    # 2. Static simulation parameters — from eqx.field(static=True) metadata
    box_size = lightcone.box_size[0]  # Assumes cubic box in Mpc/h
    n_particles = int(np.prod(lightcone.mesh_size))
    nside = lightcone.nside  # static=True field, always concrete at trace time

    # 3. Normalize sources — determines output shape at trace time
    source_kind, sources = _normalize_sources(nz_shear)
    n_sources = len(sources) if source_kind == "distribution" else sources.shape[0]
    npix = 12 * nside**2
    result_shape = jax.ShapeDtypeStruct((n_sources, npix), jnp.float32)

    # 4. Closure + pure_callback — split by source_kind because "distribution"
    #    objects are concrete Python (safe in closures) while "redshift" sources
    #    are JAX arrays (tracers) and must be passed as explicit positional args.
    if source_kind == "distribution":

        def _callback_dist(density_maps, scale_factors, density_widths, omega_m_val, h_val):
            omega_m = float(omega_m_val)
            h = float(h_val)
            omega_l = 1.0 - omega_m
            shell_redshifts = 1.0 / np.atleast_1d(np.asarray(scale_factors)) - 1.0
            density_maps_np = np.asarray(density_maps)
            density_widths_np = np.atleast_1d(np.asarray(density_widths))
            z_grid = np.linspace(min_z, max_z, n_integrate + 1)
            kappa_grid = _raytrace_z_grid(
                density_maps_np,
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
                shell_widths=density_widths_np,
                n_workers=n_workers,
            )
            # sources is a list of jc.redshift distributions — concrete Python, safe in closure
            kappa_maps = _integrate_nz(kappa_grid, z_grid, sources)
            return kappa_maps.astype(np.float32)

        kappa = jax.pure_callback(
            _callback_dist,
            result_shape,
            lightcone.array,
            lightcone.scale_factors,
            density_width,
            cosmo.Omega_m,
            cosmo.h,
        )
    else:

        def _callback_z(density_maps, scale_factors, density_widths, omega_m_val, h_val, z_sources):
            omega_m = float(omega_m_val)
            h = float(h_val)
            omega_l = 1.0 - omega_m
            shell_redshifts = 1.0 / np.atleast_1d(np.asarray(scale_factors)) - 1.0
            density_maps_np = np.asarray(density_maps)
            density_widths_np = np.atleast_1d(np.asarray(density_widths))
            # z_sources is passed as a positional arg — JAX materializes it as concrete numpy
            z_sources_np = np.atleast_1d(np.asarray(z_sources))
            kappa_maps = _raytrace_z_grid(
                density_maps_np,
                shell_redshifts,
                z_sources_np,
                box_size,
                n_particles,
                omega_m,
                h,
                omega_l,
                nside,
                interp,
                parallel_transport,
                born=born,
                shell_widths=density_widths_np,
            )
            return kappa_maps.astype(np.float32)

        kappa = jax.pure_callback(
            _callback_z,
            result_shape,
            lightcone.array,
            lightcone.scale_factors,
            density_width,
            cosmo.Omega_m,
            cosmo.h,
            sources,
        )

    # Handle single source case - squeeze to 1D
    if n_sources == 1:
        kappa = kappa[0]

    # 7. Return SphericalKappaField
    base_field = lightcone.replace(status=FieldStatus.KAPPA)
    kappa_field = SphericalKappaField.FromDensityMetadata(
        array=kappa,
        field=base_field,
        status=FieldStatus.KAPPA,
        unit=ConvergenceUnit.DIMENSIONLESS,
        z_sources=nz_shear,
    )

    # Replace shell-level metadata (inherited from lightcone) with per-source-bin metadata
    kappa_field = _attach_source_metadata(kappa_field, cosmo, source_kind, sources, min_z, max_z, n_integrate)
    return kappa_field
