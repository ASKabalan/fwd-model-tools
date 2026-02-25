"""Tests for extract_catalog streaming analysis utility."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

from fwd_model_tools._src.base._enums import DensityUnit, FieldStatus
from fwd_model_tools.fields import DensityField
from fwd_model_tools.io.catalog import Catalog
from fwd_model_tools.io.extract import CatalogExtract, extract_catalog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MESH_SIZE = (4, 4, 4)
BOX_SIZE = (100.0, 100.0, 100.0)
N_CHAINS = 2
N_FILES_PER_CHAIN = 2
N_SAMPLES_PER_FILE = 2
N_SAMPLES_PER_CHAIN = N_FILES_PER_CHAIN * N_SAMPLES_PER_FILE  # 4
COSMO_KEYS = ["Omega_c", "sigma8"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_field(seed: int = 0) -> DensityField:
    rng = np.random.RandomState(seed)
    array = jnp.asarray(rng.randn(*MESH_SIZE), dtype=jnp.float64)
    return DensityField(
        array=array,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        status=FieldStatus.DENSITY_FIELD,
        unit=DensityUnit.DENSITY,
        z_sources=jnp.array(1.0, dtype=jnp.float64),
        scale_factors=jnp.array(0.5, dtype=jnp.float64),
        comoving_centers=jnp.array(300.0, dtype=jnp.float64),
        density_width=jnp.array(25.0, dtype=jnp.float64),
    )


def _make_cosmo() -> jc.Cosmology:
    return jc.Planck18()


def _write_catalog(path: Path, seed_start: int) -> None:
    """Write a 2-sample Catalog parquet file starting at given seed."""
    field1 = _make_field(seed=seed_start)
    field2 = _make_field(seed=seed_start + 1)
    cosmo = _make_cosmo()
    catalog = Catalog(field=[field1, field2], cosmology=[cosmo, cosmo])
    catalog.to_parquet(str(path))


def _build_multichain_dir(tmp_path: Path) -> Path:
    """Create 2 chains × 2 files × 2 samples layout under tmp_path."""
    seed = 0
    for chain_idx in range(N_CHAINS):
        samples_dir = tmp_path / f"chain_{chain_idx}" / "samples"
        samples_dir.mkdir(parents=True)
        for file_idx in range(N_FILES_PER_CHAIN):
            _write_catalog(samples_dir / f"samples_{file_idx}.parquet", seed_start=seed)
            seed += N_SAMPLES_PER_FILE
    return tmp_path


def _build_singlechain_dir(tmp_path: Path) -> Path:
    """Create single-chain layout: tmp_path/samples/*.parquet."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir(parents=True)
    seed = 0
    for file_idx in range(N_FILES_PER_CHAIN):
        _write_catalog(samples_dir / f"samples_{file_idx}.parquet", seed_start=seed)
        seed += N_SAMPLES_PER_FILE
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_extract_cosmo_shape(tmp_path):
    """Cosmo dict has shape (n_chains, n_samples) for multi-chain layout."""
    root = _build_multichain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS)

    assert isinstance(result, CatalogExtract)
    assert result.mean_field is None
    assert result.std_field is None
    assert result.power_spectra is None
    assert result.true_ic is None

    cosmo = result.cosmo
    for key in COSMO_KEYS:
        assert key in cosmo
        assert cosmo[key].shape == (
            N_CHAINS,
            N_SAMPLES_PER_CHAIN,
        ), f"cosmo['{key}'].shape = {cosmo[key].shape}, expected ({N_CHAINS}, {N_SAMPLES_PER_CHAIN})"


def test_extract_field_statistic_shapes(tmp_path):
    """mean_field and std_field have shape (n_chains, X, Y, Z) with float64 dtype."""
    root = _build_multichain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS, field_statistic=True)

    mean_field = result.mean_field
    std_field = result.std_field

    assert mean_field is not None
    assert std_field is not None

    expected_shape = (N_CHAINS, *MESH_SIZE)
    assert mean_field.array.shape == expected_shape, f"mean_field.shape = {mean_field.array.shape}"
    assert std_field.array.shape == expected_shape, f"std_field.shape = {std_field.array.shape}"

    assert mean_field.array.dtype == np.float64, f"dtype = {mean_field.array.dtype}"
    assert std_field.array.dtype == np.float64, f"dtype = {std_field.array.dtype}"


def test_extract_field_statistic_false_gives_none(tmp_path):
    """When field_statistic=False, mean_field and std_field are None."""
    root = _build_multichain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS, field_statistic=False)

    assert result.mean_field is None
    assert result.std_field is None


def test_extract_power_statistic_shapes(tmp_path):
    """Power spectra tuple has 4 elements; each has n_chains as leading dim."""
    root = _build_multichain_dir(tmp_path)
    true_ic = _make_field(seed=99)
    result = extract_catalog(
        str(root),
        cosmo_keys=COSMO_KEYS,
        true_ic=true_ic,
        field_statistic=True,
        power_statistic=True,
    )

    assert result.power_spectra is not None
    mean_transfer, std_transfer, mean_coherence, std_coherence = result.power_spectra

    for ps in (mean_transfer, std_transfer, mean_coherence, std_coherence):
        assert ps.array.shape[0] == N_CHAINS, f"Leading dim should be n_chains={N_CHAINS}, got {ps.array.shape}"


def test_extract_power_statistic_requires_true_ic(tmp_path):
    """extract_catalog raises ValueError when power_statistic=True but true_ic=None."""
    root = _build_multichain_dir(tmp_path)
    with pytest.raises(ValueError, match="true_ic"):
        extract_catalog(str(root), cosmo_keys=COSMO_KEYS, power_statistic=True)


def test_extract_singlechain_layout(tmp_path):
    """Single-chain layout (path/samples/*.parquet) returns cosmo with shape (1, n_samples)."""
    root = _build_singlechain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS, field_statistic=True)

    for key in COSMO_KEYS:
        assert result.cosmo[key].shape == (1, N_SAMPLES_PER_CHAIN), f"cosmo['{key}'].shape = {result.cosmo[key].shape}"

    assert result.mean_field.array.shape == (1, *MESH_SIZE)
    assert result.std_field.array.shape == (1, *MESH_SIZE)


def test_extract_field_metadata_preserved(tmp_path):
    """mean_field carries the correct mesh_size and box_size metadata."""
    root = _build_multichain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS, field_statistic=True)

    mean_field = result.mean_field
    assert mean_field.mesh_size == MESH_SIZE
    assert mean_field.box_size == BOX_SIZE


def test_catalog_extract_getitem_integer(tmp_path):
    """Integer index returns a 1-chain CatalogExtract with 2-D cosmo arrays."""
    root = _build_multichain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS)

    chain0 = result[0]
    assert chain0.n_chains == 1
    assert chain0.cosmo["Omega_c"].shape == (1, N_SAMPLES_PER_CHAIN)
    assert chain0.true_ic is None


def test_catalog_extract_getitem_slice(tmp_path):
    """Slice index preserves 2-D shape on cosmo and field arrays."""
    root = _build_multichain_dir(tmp_path)
    result = extract_catalog(str(root), cosmo_keys=COSMO_KEYS, field_statistic=True)

    sub = result[0:1]
    assert sub.n_chains == 1
    assert sub.cosmo["Omega_c"].shape == (1, N_SAMPLES_PER_CHAIN)
    assert sub.mean_field.array.shape == (1, *MESH_SIZE)


def test_requires_datasets_without_library(monkeypatch):
    """requires_datasets raises ImportError with install hint when datasets is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "datasets", None)
    from importlib import reload

    import fwd_model_tools.io.extract as m

    reload(m)
    with pytest.raises(ImportError, match="fwd-model-tools\\[catalog\\]"):
        m.extract_catalog("dummy", cosmo_keys=["Omega_c"])
