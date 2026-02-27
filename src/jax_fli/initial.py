from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.kernels import fftk, interpolate_power_spectrum
from jaxtyping import Array, PRNGKeyArray
from numpyro.util import is_prng_key

from .fields import DensityField, DensityUnit, FieldStatus


@partial(
    jax.jit,
    static_argnames=[
        "mesh_size",
        "box_size",
        "cosmo",
        "pk_fn",
        "observer_position",
        "flatsky_npix",
        "nside",
        "field_size",
        "halo_size",
        "sharding",
    ],
)
def gaussian_initial_conditions(
    key: PRNGKeyArray,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    *,
    cosmo: jc.Cosmology | None = None,
    pk_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5),
    flatsky_npix: tuple[int, int] | None = None,
    nside: int | None = None,
    field_size: tuple[int, int] | None = None,
    halo_size: int | tuple[int, int] = (0, 0),
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

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> pk = lambda k: 1.0 / (1.0 + k**2)
    >>> field = gaussian_initial_conditions(
    ...     jr.PRNGKey(0),
    ...     mesh_size=(16, 16, 16),
    ...     box_size=(200.0, 200.0, 200.0),
    ...     pk_fn=pk,
    ... )
    >>> field.array.shape
    (16, 16, 16)
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
        field_size=field_size,
        halo_size=halo_size,
        sharding=sharding,
    )


@partial(
    jax.jit,
    static_argnames=[
        "mesh_size",
        "box_size",
        "cosmo",
        "pk_fn",
        "observer_position",
        "flatsky_npix",
        "nside",
        "field_size",
        "halo_size",
        "sharding",
    ],
)
def interpolate_initial_conditions(
    initial_field: Array,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    *,
    cosmo: jc.Cosmology | None = None,
    pk_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5),
    flatsky_npix: tuple[int, int] | None = None,
    nside: int | None = None,
    field_size: tuple[int, int] | None = None,
    halo_size: int | tuple[int, int] = (0, 0),
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
            raise ValueError("Either pk_fn or cosmo must be provided to compute the power spectrum.")
        else:
            k = jnp.logspace(-4, 1, 128)
            pk = jc.power.linear_matter_power(cosmo, k)
            pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)

    field = fft3d(initial_field)
    kvec = fftk(field)
    kmesh = sum((kk / box_size[i] * mesh_size[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    factor = (mesh_size[0] * mesh_size[1] * mesh_size[2]) / (box_size[0] * box_size[1] * box_size[2])
    pkmesh = pk_fn(kmesh) * factor

    field = field * jnp.sqrt(pkmesh)
    field = ifft3d(field)

    return DensityField(
        array=field,
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position=observer_position,
        sharding=sharding,
        halo_size=halo_size,
        #
        nside=nside,
        field_size=field_size,
        flatsky_npix=flatsky_npix,
        #
        scale_factors=0.0,
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )
