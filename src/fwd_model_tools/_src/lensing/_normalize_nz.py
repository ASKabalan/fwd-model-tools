from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax_cosmo as jc

__all__ = ["_normalize_sources"]


def _normalize_sources(nz_shear: Any) -> tuple[str, list[Any]]:
    """Accept either scalar redshifts or jc.redshift distributions (not both)."""

    if isinstance(nz_shear, (list | tuple)):
        entries = list(nz_shear)
    else:
        entries = [nz_shear]

    if not entries:
        raise ValueError("nz_shear must contain at least one entry")

    first = entries[0]
    first_is_distribution = isinstance(first | jc.redshift.redshift_distribution)

    if first_is_distribution:
        for entry in entries:
            if not isinstance(entry, jc.redshift.redshift_distribution):
                raise ValueError("Cannot mix redshift distributions with scalar sources")
        return "distribution", entries

    z_array = jnp.array(entries).squeeze()
    if z_array.ndim == 0:
        z_array = jnp.array([z_array])
    if z_array.ndim != 1:
        raise ValueError("Scalar redshift sources must be 1D array-like")

    return "redshift", z_array
