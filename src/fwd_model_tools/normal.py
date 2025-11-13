from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.kernels import fftk, interpolate_power_spectrum
from numpyro.distributions import Normal, constraints
from numpyro.distributions.util import promote_shapes
from numpyro.util import is_prng_key
from functools import partial
from .field import DensityField, FieldStatus

@partial(jax.jit, static_argnames=['mesh_size', 'box_size', 'cosmo', 'pk_fn', 'observer_position', 'flatsky_npix', 'nside', 'halo_size', 'sharding'])
def gaussian_initial_conditions(
    key: jax.random.KeyArray,
    mesh_size: Tuple[int, int, int],
    box_size: Tuple[float, float, float],
    *,
    cosmo: jc.Cosmology | None = None,
    pk_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    flatsky_npix: Tuple[int, int] | None = None,
    nside: int | None = None,
    halo_size: int | Tuple[int, int] = 0,
    sharding: Any | None = None,
) -> DensityField:
    """
    Sample Gaussian initial conditions and package them as a Field PyTree.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator.
    mesh_size : tuple[int, int, int]
        Discretization of the simulation volume.
    box_size : tuple[float, float, float]
        Physical box side lengths (Mpc/h).
    pk_fn : callable
        Function mapping |k| to the linear matter power spectrum.
    observer_position : tuple[float, float, float], optional
        Fractional observer coordinates; defaults to box center.
    flatsky_npix : tuple[int, int], optional
        Requested flat-sky pixel resolution for downstream projections.
    nside : int, optional
        HEALPix resolution for spherical projections.
    halo_size : int | tuple[int, int], optional
        Halo exchange depth for distributed painting.
    sharding : Any, optional
        JAX sharding descriptor for distributed arrays.

    Returns
    -------
    DensityField
        Field PyTree populated with the linear density field.
    """
    if not is_prng_key(key):
        raise ValueError("key must be a jax.random.PRNGKey")

    field = normal_field(seed=key, shape=mesh_size, sharding=sharding)
    return interpolate_initial_conditions(
        initial_field=field,
        mesh_size=mesh_size,
        box_size=box_size,
        cosmo=cosmo,
        pk_fn=pk_fn,
        observer_position=observer_position,
        flatsky_npix=flatsky_npix,
        nside=nside,
        halo_size=halo_size,
        sharding=sharding,
    )

@partial(jax.jit, static_argnames=['mesh_size', 'box_size', 'cosmo', 'pk_fn', 'observer_position', 'flatsky_npix', 'nside', 'halo_size', 'sharding'])
def interpolate_initial_conditions(
    initial_field: jax.Array,
    mesh_size: Tuple[int, int, int],
    box_size: Tuple[float, float, float],
    *,
    cosmo: jc.Cosmology | None = None,
    pk_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    flatsky_npix: Tuple[int, int] | None = None,
    nside: int | None = None,
    halo_size: int | Tuple[int, int] = 0,
    sharding: Any | None = None,
) -> DensityField:
    """
    Sample Gaussian initial conditions and package them as a Field PyTree.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator.
    mesh_size : tuple[int, int, int]
        Discretization of the simulation volume.
    box_size : tuple[float, float, float]
        Physical box side lengths (Mpc/h).
    pk_fn : callable
        Function mapping |k| to the linear matter power spectrum.
    observer_position : tuple[float, float, float], optional
        Fractional observer coordinates; defaults to box center.
    flatsky_npix : tuple[int, int], optional
        Requested flat-sky pixel resolution for downstream projections.
    nside : int, optional
        HEALPix resolution for spherical projections.
    halo_size : int | tuple[int, int], optional
        Halo exchange depth for distributed painting.
    sharding : Any, optional
        JAX sharding descriptor for distributed arrays.

    Returns
    -------
    DensityField
        Field PyTree populated with the linear density field.
    """
    if pk_fn is None:
        if cosmo is None:
            raise ValueError(
                "Either pk_fn or cosmo must be provided to compute the power spectrum."
            )
        else:
            k = jnp.logspace(-4, 1, 256)
            pk = jc.power.linear_matter_power(cosmo, k)
            cosmo._workspace = {}
            pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)

    field = fft3d(initial_field)
    kvec = fftk(field)
    kmesh = sum(
        (kk / box_size[i] * mesh_size[i])**2 for i, kk in enumerate(kvec))**0.5
    pkmesh = pk_fn(kmesh) * (mesh_size[0] * mesh_size[1] * mesh_size[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    field = field * jnp.sqrt(pkmesh)
    field = ifft3d(field)

    return DensityField(
        array=field,
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position=observer_position,
        sharding=sharding,
        nside=nside,
        flatsky_npix=flatsky_npix,
        halo_size=halo_size,
        status=FieldStatus.INITIAL_FIELD,
        scale_factors=0.0,
    )
