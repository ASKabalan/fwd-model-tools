from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp

from .._src.power import _coherence, _cross_spherical_cl, _flat_cl, _power, _spherical_cl, _transfer


@partial(jax.jit, static_argnames=["box_shape", "multipoles", "los"])
def power(
    mesh,
    mesh2=None,
    *,
    box_shape: tuple[float, float, float],
    kedges: int | float | Iterable[float] | None = None,
    multipoles: int | Iterable[int] = 0,
    los: jnp.ndarray | Iterable[float] = jnp.array([0.0, 0.0, 1.0]),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Array-only 3D power spectrum. Returns (wavenumber, spectra)."""
    box_shape = tuple(box_shape)
    los_array = None if multipoles == 0 else tuple(jnp.asarray(los))
    wavenumber, spectra = _power(
        mesh,
        mesh2,
        box_shape=box_shape,
        kedges=kedges,
        multipoles=multipoles,
        los=los_array,
    )
    return wavenumber, spectra


def transfer(
    mesh0,
    mesh1,
    *,
    box_shape: tuple[float, float, float],
    kedges: int | float | Iterable[float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Monopole transfer function sqrt(P1/P0)."""
    wavenumber, tr = _transfer(
        mesh0,
        mesh1,
        box_shape=box_shape,
        kedges=kedges,
    )
    return wavenumber, tr


def coherence(
    mesh0,
    mesh1,
    *,
    box_shape: tuple[float, float, float],
    kedges: int | float | Iterable[float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Monopole coherence pk01 / sqrt(pk0 pk1)."""
    wavenumber, coh = _coherence(
        mesh0,
        mesh1,
        box_shape=box_shape,
        kedges=kedges,
    )
    return wavenumber, coh


def angular_cl_flat(
    map1,
    map2=None,
    *,
    pixel_size: float | None = None,
    field_size: float | None = None,
    ell_edges: Iterable[float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Flat-sky angular Cl. Returns (ell, spectra)."""
    ell, spectra = _flat_cl(
        map1,
        map2,
        pixel_size=pixel_size,
        field_size=field_size,
        ell_edges=ell_edges,
    )
    return ell, spectra


def angular_cl_spherical(
    map_sphere,
    map_sphere2=None,
    *,
    lmax: int | None = None,
    method: str = "jax",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Spherical (HEALPix) angular Cl. Returns (ell, spectra)."""
    ell_out, spectra = _spherical_cl(map_sphere, map_sphere2, lmax=lmax, method=method)
    return ell_out, spectra


def cross_angular_cl_spherical(
    maps,
    *,
    lmax: int | None = None,
    method: str = "healpy",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cross-spherical (HEALPix) angular Cl for all pairs. Returns (ell, spectra)."""
    ell_out, spectra = _cross_spherical_cl(maps, lmax=lmax, method=method)
    return ell_out, spectra


__all__ = ["power", "transfer", "coherence", "angular_cl_flat", "angular_cl_spherical", "cross_angular_cl_spherical"]
