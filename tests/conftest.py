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
