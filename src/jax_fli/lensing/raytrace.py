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
        msg = "Missing optional library 'dorian'. Install with: pip install jax-fli[raytrace]"
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
    raytrace: bool = True,
    shell_widths: np.ndarray | None = None,
    nufft_threads: int = 4,
    n_workers=None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        Whether to include Born approximation convergence in the output.
    raytrace : bool, optional
        Whether to include full ray-traced convergence in the output. Default: True.
        When both ``born`` and ``raytrace`` are True, returns a tuple.
    shell_widths : np.ndarray or None, optional
        Shell thickness d_R per shell in Mpc/h.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Convergence maps, shape (n_sources, npix). When both ``born`` and
        ``raytrace`` are True, returns ``(raytraced_maps, born_maps)``.
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
        shell_widths=list(shell_widths),
        parallel_transport=parallel_transport,
        n_workers=n_workers,
    )
    if raytrace and born:
        return result["convergence_raytraced"], result["convergence_born"]
    elif raytrace:
        return result["convergence_raytraced"]
    elif born:
        return result["convergence_born"]
    else:
        raise ValueError("At least one of raytrace or born must be True to return results.")


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
    max_z=1.5,
    n_integrate=32,
    # Dorian options
    interp="bilinear",
    parallel_transport=True,
    born=False,
    raytrace=True,
    n_workers=None,
) -> tuple[SphericalKappaField | None, SphericalKappaField | None]:
    """Multi-plane ray-tracing using dorian.

    Computes weak lensing convergence maps by propagating light rays through
    a series of mass shells, including post-Born corrections from the full
    distortion matrix.

    This function wraps dorian's ray-tracing and provides the same API as
    :func:`jax_fli.lensing.born`. This function must be called **outside**
    :func:`jax.jit`. A :exc:`ValueError` is raised if any input is a JAX tracer.

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
        Default: 1.5.
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
    born : bool, optional
        Whether to compute the Born approximation convergence. Default: False.
    raytrace : bool, optional
        Whether to compute the full ray-traced convergence. Default: True.

    Returns
    -------
    tuple[SphericalKappaField | None, SphericalKappaField | None]
        A ``(raytraced, born)`` tuple. The flag combination determines what is
        returned:

        - ``raytrace=True, born=False`` (default): ``(SphericalKappaField, None)``
        - ``raytrace=False, born=True``: ``(None, SphericalKappaField)``
        - ``raytrace=True, born=True``: ``(SphericalKappaField, SphericalKappaField)``

        Each non-None field has shape ``(n_sources, npix)`` if multiple sources,
        or ``(npix,)`` if a single source.

    Raises
    ------
    ValueError
        If lightcone status is not LIGHTCONE.
    ValueError
        If both ``born`` and ``raytrace`` are False (nothing to compute).
    TypeError
        If lightcone is not SphericalDensity (flat-sky not supported).

    Notes
    -----
    The convergence is computed from the full ray-traced distortion matrix A:
    ``kappa = 1 - 0.5 * (A[0,0] + A[1,1])``, which includes all post-Born
    corrections.

    Examples
    --------
    >>> import jax_cosmo as jc
    >>> from jax_fli.lensing import raytrace
    >>> # Default: ray-traced only
    >>> kappa_rt, _ = raytrace(cosmo, lightcone, nz_shear=[0.5, 1.0, 1.5])
    >>> kappa_rt.array.shape  # (3, npix)
    >>> # Both outputs simultaneously (single dorian call)
    >>> kappa_rt, kappa_born = raytrace(cosmo, lightcone, [1.0], born=True, raytrace=True)
    >>> # Born approximation only
    >>> _, kappa_born = raytrace(cosmo, lightcone, [1.0], born=True, raytrace=False)
    >>> # With redshift distribution
    >>> nz = jc.redshift.smail_nz(1.0, 2.0, 1.0)
    >>> kappa_rt, _ = raytrace(cosmo, lightcone, nz_shear=nz)

    See Also
    --------
    born : Born approximation (faster, fully-JAX, no post-Born corrections).
    """
    # 1. Validate on static metadata
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
    nside = lightcone.nside  # static=True field, always concrete at trace
    assert nside is not None  # type narrowing for checker + closures

    # 2b. Validate flag combination — must request at least one output
    if not born and not raytrace:
        raise ValueError("At least one of `born` or `raytrace` must be True.")

    # Tracer lock — raytrace() must not be called inside jax.jit
    _jax_inputs = (lightcone.array, lightcone.scale_factors, cosmo.Omega_m, cosmo.h)
    if not jax.tree.all(jax.tree.map(jax.core.is_concrete, _jax_inputs)):
        raise ValueError(
            "raytrace() cannot be called inside jax.jit. "
            "All inputs must be concrete arrays. Use outside of jit context."
        )

    # 3. Normalize sources
    source_kind, sources = _normalize_sources(nz_shear)
    n_sources = len(sources) if source_kind == "distribution" else sources.shape[0]

    # 4. Extract concrete numpy arrays and run dorian
    omega_m = float(np.asarray(cosmo.Omega_m))
    h = float(np.asarray(cosmo.h))
    omega_l = 1.0 - omega_m
    shell_redshifts = 1.0 / np.atleast_1d(np.asarray(lightcone.scale_factors)) - 1.0
    density_maps_np = np.asarray(lightcone.array)
    density_widths_np = np.atleast_1d(np.asarray(density_width))

    if source_kind == "distribution":
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
            raytrace=raytrace,
            shell_widths=density_widths_np,
            n_workers=n_workers,
        )
        if born and raytrace:
            kappa_grid_rt, kappa_grid_born = kappa_grid
            kappa = (
                _integrate_nz(kappa_grid_rt, z_grid, sources).astype(np.float32),
                _integrate_nz(kappa_grid_born, z_grid, sources).astype(np.float32),
            )
        else:
            kappa = _integrate_nz(kappa_grid, z_grid, sources).astype(np.float32)
    else:
        z_sources_np = np.atleast_1d(np.asarray(sources))
        kappa = _raytrace_z_grid(
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
            raytrace=raytrace,
            shell_widths=density_widths_np,
            n_workers=n_workers,
        )

    # 5. Build output fields — helper avoids repetition across the three return paths
    def _build_field(arr):
        arr_jax = jnp.array(arr)  # numpy → JAX for SphericalKappaField
        if n_sources == 1:
            arr_jax = arr_jax[0]
        base_field = lightcone.replace(status=FieldStatus.KAPPA)
        field = SphericalKappaField.FromDensityMetadata(
            array=arr_jax,
            field=base_field,
            status=FieldStatus.KAPPA,
            unit=ConvergenceUnit.DIMENSIONLESS,
            z_sources=nz_shear,
        )
        return _attach_source_metadata(field, cosmo, source_kind, sources, min_z, max_z, n_integrate)

    if born and raytrace:
        kappa_rt_raw, kappa_born_raw = kappa
        return _build_field(kappa_rt_raw), _build_field(kappa_born_raw)
    elif raytrace:
        return _build_field(kappa), None
    else:  # born only
        return None, _build_field(kappa)
