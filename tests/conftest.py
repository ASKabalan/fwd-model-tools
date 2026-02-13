"""Shared fixtures for fwd_model_tools tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import pytest

from fwd_model_tools.parameters import Planck18


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    """Enable float64 for all tests (required for lossless parquet round-trips)."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="session")
def cosmology():
    """Planck18 cosmology instance."""
    return Planck18()


@pytest.fixture(scope="session")
def kde_nz_obj():
    """Unbatched ``jax_cosmo.redshift.kde_nz`` with 100-point redshift distribution."""
    zcat = jnp.linspace(0.01, 2.0, 100)
    weight = jnp.exp(-0.5 * ((zcat - 1.0) / 0.3) ** 2)
    return jc.redshift.kde_nz(zcat, weight, bw=0.05, zmax=2.5, gals_per_arcmin2=30.0)
