"""Round-trip tests for Catalog v2 format (list-based storage, 4-method API).

Test matrix: 6 field types x 2 batching modes x 2 n_entries = 24 round-trip tests,
each testing both parquet and dataset round-trips.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

import jax_cosmo as jc

import fwd_model_tools as ffi
from fwd_model_tools._src.base._enums import ConvergenceUnit
from fwd_model_tools.io.catalog import Catalog

jax.config.update("jax_enable_x64", False)  # Use float64 for better precision in tests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 3
MESH_SIZE = (8, 8, 8)
BOX_SIZE = (100.0, 100.0, 100.0)
NSIDE = 4  # 12 * 4^2 = 192 pixels
FLAT_NPIX = (16, 16)

FIELD_TYPES = [
    "SphericalDensity",
    "FlatDensity",
    "SphericalKappaField",
    "FlatKappaField",
    "ParticleField",
    "DensityField",
]


# ---------------------------------------------------------------------------
# Error function
# ---------------------------------------------------------------------------
def error(tree1, tree2):
    """Compute max squared error across all leaves of two PyTrees."""
    return jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(lambda x, y: jnp.max((x - y)**2), tree1, tree2),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_field(field_type: str, batched: bool, seed: int = 42):
    """Construct a single field with deterministic random data and numeric z_sources."""
    rng = np.random.RandomState(seed)
    B = BATCH_SIZE if batched else 1
    npix_healpix = 12 * NSIDE**2  # 192

    # Build array: always use (B, spatial...) shape so field[0] strips the batch dim for unbatched
    if field_type in ("SphericalDensity", "SphericalKappaField"):
        shape = (B, npix_healpix)
    elif field_type in ("FlatDensity", "FlatKappaField"):
        shape = (B, *FLAT_NPIX)
    elif field_type == "DensityField":
        shape = (B, *MESH_SIZE)
    elif field_type == "ParticleField":
        shape = (B, *MESH_SIZE, 3)
    else:
        raise ValueError(f"Unknown field_type: {field_type}")

    array = jnp.asarray(rng.randn(*shape), dtype=jnp.float64)

    # Dynamic metadata (numeric z_sources)
    if batched:
        z_sources = jnp.asarray(rng.uniform(0.5, 2.0, size=B), dtype=jnp.float64)
        scale_factors = jnp.asarray(rng.uniform(0.5, 1.0, size=B), dtype=jnp.float64)
        comoving_centers = jnp.asarray(rng.uniform(100, 500, size=B), dtype=jnp.float64)
        density_width = jnp.asarray(rng.uniform(10, 50, size=B), dtype=jnp.float64)
    else:
        z_sources = jnp.array(rng.uniform(0.5, 2.0), dtype=jnp.float64)
        scale_factors = jnp.array(0.8, dtype=jnp.float64)
        comoving_centers = jnp.array(300.0, dtype=jnp.float64)
        density_width = jnp.array(25.0, dtype=jnp.float64)

    # Choose unit and status based on field type
    if field_type in ("SphericalKappaField", "FlatKappaField"):
        unit = ConvergenceUnit.DIMENSIONLESS
        status = ffi.FieldStatus.KAPPA
    elif field_type == "ParticleField":
        unit = ffi.PositionUnit.GRID_RELATIVE
        status = ffi.FieldStatus.PARTICLES
    else:
        unit = ffi.DensityUnit.DENSITY
        status = ffi.FieldStatus.LIGHTCONE

    # Choose geometry keys
    nside = NSIDE if field_type in ("SphericalDensity", "SphericalKappaField") else None
    flatsky_npix = FLAT_NPIX if field_type in ("FlatDensity", "FlatKappaField") else None

    field_cls = {
        "SphericalDensity": ffi.SphericalDensity,
        "FlatDensity": ffi.FlatDensity,
        "SphericalKappaField": ffi.SphericalKappaField,
        "FlatKappaField": ffi.FlatKappaField,
        "ParticleField": ffi.ParticleField,
        "DensityField": ffi.DensityField,
    }[field_type]

    field = field_cls(
        array=array,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=None,
        z_sources=z_sources,
        scale_factors=scale_factors,
        comoving_centers=comoving_centers,
        density_width=density_width,
        status=status,
        unit=unit,
    )
    if not batched:
        field = field[0]
    return field


def make_catalog(field_type: str, n_entries: int, batched: bool) -> Catalog:
    """Build a Catalog with n_entries fields + Planck18 cosmologies."""
    fields = [make_field(field_type, batched, seed=42 + i) for i in range(n_entries)]
    cosmologies = [jc.Planck18() for _ in range(n_entries)]
    return Catalog(field=fields, cosmology=cosmologies)


# ---------------------------------------------------------------------------
# Main parametrized test: 6 x 2 x 2 = 24 tests, each testing parquet + dataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_entries", [1, 3], ids=["single", "multi"])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("field_type", FIELD_TYPES)
def test_catalog_roundtrip(tmp_path, field_type, batched, n_entries):
    """Round-trip: Catalog -> write -> read -> compare."""
    catalog = make_catalog(field_type, n_entries=n_entries, batched=batched)

    print(f"Testing field_type={field_type}, batched={batched}, n_entries={n_entries}")

    # Parquet round-trip
    path = str(tmp_path / "test.parquet")
    catalog.to_parquet(path)
    reloaded = Catalog.from_parquet(path)
    assert isinstance(reloaded, Catalog)
    assert len(reloaded) == n_entries
    assert error(catalog, reloaded) < 1e-8

    # Dataset round-trip
    ds = catalog.to_dataset()
    reloaded_ds = Catalog.from_dataset(ds)
    assert isinstance(reloaded_ds, Catalog)
    assert len(reloaded_ds) == n_entries
    assert error(catalog, reloaded_ds) < 1e-8


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------
def test_catalog_single_field_normalization():
    """Single field/cosmology should be auto-wrapped to list."""
    field = make_field("SphericalDensity", batched=False)
    cosmo = jc.Planck18()
    cat = Catalog(field=field, cosmology=cosmo)
    assert isinstance(cat.field, list)
    assert isinstance(cat.cosmology, list)
    assert len(cat) == 1


def test_catalog_length_mismatch():
    """Mismatched list lengths should raise ValueError."""
    f1 = make_field("SphericalDensity", batched=False, seed=1)
    f2 = make_field("SphericalDensity", batched=False, seed=2)
    cosmo = jc.Planck18()
    with pytest.raises(ValueError, match="same length"):
        Catalog(field=[f1, f2], cosmology=[cosmo])


def test_catalog_getitem():
    """Indexing a multi-entry Catalog returns a sub-Catalog."""
    catalog = make_catalog("FlatDensity", n_entries=3, batched=False)
    assert len(catalog) == 3

    sub = catalog[0]
    assert isinstance(sub, Catalog)
    assert len(sub) == 1

    sliced = catalog[0:2]
    assert isinstance(sliced, Catalog)
    assert len(sliced) == 2


def test_spherical_density_ndim_validation():
    """4D array (too many dims) should be rejected by SphericalDensity."""
    # (2, 3, 192) is now valid as (N, S, npix). A 4D array is still invalid.
    array_4d = jnp.ones((2, 3, 4, 192))
    with pytest.raises(ValueError, match="SphericalDensity array must have shape"):
        ffi.SphericalDensity(
            array=array_4d,
            mesh_size=MESH_SIZE,
            box_size=BOX_SIZE,
            observer_position=(0.5, 0.5, 0.5),
            sharding=None,
            halo_size=(0, 0),
            nside=NSIDE,
            flatsky_npix=None,
            field_size=None,
            z_sources=jnp.array(1.0),
            scale_factors=jnp.array(0.8),
            comoving_centers=jnp.array(300.0),
            density_width=jnp.array(25.0),
            status=ffi.FieldStatus.LIGHTCONE,
            unit=ffi.DensityUnit.DENSITY,
        )


def test_spherical_density_wrong_npix_validation():
    """3D array with wrong last dimension should be rejected by SphericalDensity."""
    # npix for nside=4 is 192; use 100 to trigger the npix mismatch error
    array_wrong_npix = jnp.ones((2, 3, 100))
    with pytest.raises(ValueError, match="does not match"):
        ffi.SphericalDensity(
            array=array_wrong_npix,
            mesh_size=MESH_SIZE,
            box_size=BOX_SIZE,
            observer_position=(0.5, 0.5, 0.5),
            sharding=None,
            halo_size=(0, 0),
            nside=NSIDE,
            flatsky_npix=None,
            field_size=None,
            z_sources=jnp.array(1.0),
            scale_factors=jnp.array(0.8),
            comoving_centers=jnp.array(300.0),
            density_width=jnp.array(25.0),
            status=ffi.FieldStatus.LIGHTCONE,
            unit=ffi.DensityUnit.DENSITY,
        )


# ---------------------------------------------------------------------------
# Helpers for multi-batched tests
# ---------------------------------------------------------------------------


def _make_batched_cosmo(n: int) -> jc.Cosmology:
    """Build a batched jc.Cosmology with n identical Planck18-like entries."""
    return jc.Cosmology(
        Omega_c=jnp.full(n, 0.3),
        Omega_b=jnp.full(n, 0.05),
        h=jnp.full(n, 0.7),
        n_s=jnp.full(n, 0.96),
        sigma8=jnp.full(n, 0.8),
        w0=jnp.full(n, -1.0),
        wa=jnp.full(n, 0.0),
        Omega_k=jnp.full(n, 0.0),
        Omega_nu=jnp.full(n, 0.0),
    )


def _make_multi_batched_density(N: int, S: int, seed: int = 0) -> ffi.DensityField:
    """Create an (N, S, *MESH_SIZE) DensityField with metadata shaped (N, S)."""
    rng = np.random.RandomState(seed)
    array = jnp.asarray(rng.randn(N, S, *MESH_SIZE))
    # metadata shaped (N, S) so field[i].z_sources has shape (S,)
    z_sources = jnp.asarray(rng.uniform(0.5, 2.0, size=(N, S)))
    scale_factors = jnp.asarray(rng.uniform(0.5, 1.0, size=(N, S)))
    comoving_centers = jnp.asarray(rng.uniform(100, 500, size=(N, S)))
    density_width = jnp.asarray(rng.uniform(10, 50, size=(N, S)))
    return ffi.DensityField(
        array=array,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        nside=None,
        flatsky_npix=None,
        field_size=None,
        z_sources=z_sources,
        scale_factors=scale_factors,
        comoving_centers=comoving_centers,
        density_width=density_width,
        status=ffi.FieldStatus.DENSITY_FIELD,
        unit=ffi.DensityUnit.DENSITY,
    )


# ---------------------------------------------------------------------------
# New multi-batched tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_type", FIELD_TYPES)
def test_multi_batched_field_shapes(field_type):
    """(N, S, spatial...) arrays should be accepted by all field types."""
    N, S = 2, 4
    npix_healpix = 12 * NSIDE**2  # 192

    if field_type in ("SphericalDensity", "SphericalKappaField"):
        array = jnp.zeros((N, S, npix_healpix))
        nside = NSIDE
        flatsky_npix = None
    elif field_type in ("FlatDensity", "FlatKappaField"):
        array = jnp.zeros((N, S, *FLAT_NPIX))
        nside = None
        flatsky_npix = FLAT_NPIX
    elif field_type == "DensityField":
        array = jnp.zeros((N, S, *MESH_SIZE))
        nside = None
        flatsky_npix = None
    elif field_type == "ParticleField":
        array = jnp.zeros((N, S, *MESH_SIZE, 3))
        nside = None
        flatsky_npix = None

    if field_type in ("SphericalKappaField", "FlatKappaField"):
        unit = ConvergenceUnit.DIMENSIONLESS
        status = ffi.FieldStatus.KAPPA
    elif field_type == "ParticleField":
        unit = ffi.PositionUnit.GRID_RELATIVE
        status = ffi.FieldStatus.PARTICLES
    else:
        unit = ffi.DensityUnit.DENSITY
        status = ffi.FieldStatus.LIGHTCONE

    field_cls = {
        "SphericalDensity": ffi.SphericalDensity,
        "FlatDensity": ffi.FlatDensity,
        "SphericalKappaField": ffi.SphericalKappaField,
        "FlatKappaField": ffi.FlatKappaField,
        "ParticleField": ffi.ParticleField,
        "DensityField": ffi.DensityField,
    }[field_type]

    field = field_cls(
        array=array,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=None,
        z_sources=jnp.array(1.0),
        scale_factors=jnp.array(0.8),
        comoving_centers=jnp.array(300.0),
        density_width=jnp.array(25.0),
        status=status,
        unit=unit,
    )
    assert field.is_multi_batched(), f"{field_type}: expected is_multi_batched() == True"
    assert field.is_batched(), f"{field_type}: expected is_batched() == True"


def test_stack_snapshot_to_multi_batched():
    """Stacking two (S, *MESH_SIZE) fields yields an (2, S, *MESH_SIZE) multi-batched field."""
    S = 4
    f1 = ffi.DensityField(
        array=jnp.zeros((S, *MESH_SIZE)),
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        z_sources=jnp.array(1.0),
        scale_factors=jnp.array(0.8),
        comoving_centers=jnp.array(300.0),
        density_width=jnp.array(25.0),
        status=ffi.FieldStatus.DENSITY_FIELD,
        unit=ffi.DensityUnit.DENSITY,
    )
    f2 = ffi.DensityField(
        array=jnp.ones((S, *MESH_SIZE)),
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        z_sources=jnp.array(1.2),
        scale_factors=jnp.array(0.75),
        comoving_centers=jnp.array(350.0),
        density_width=jnp.array(20.0),
        status=ffi.FieldStatus.DENSITY_FIELD,
        unit=ffi.DensityUnit.DENSITY,
    )
    stacked = ffi.DensityField.stack([f1, f2])
    assert stacked.array.shape == (2, S, *MESH_SIZE)
    assert stacked.is_multi_batched()


def test_stack_multi_batched_raises():
    """Stacking already-multi-batched (N, S, ...) fields should raise ValueError mentioning concat."""
    N, S = 2, 4
    f = _make_multi_batched_density(N, S)
    with pytest.raises(ValueError, match="concat"):
        ffi.DensityField.stack([f, f])


def test_concat_multi_batched():
    """DensityField.concat joins two (N, S, ...) fields along axis 0."""
    N, S = 2, 4
    f1 = _make_multi_batched_density(N, S, seed=1)
    f2 = _make_multi_batched_density(N, S, seed=2)
    result = ffi.DensityField.concat([f1, f2])
    assert result.array.shape[0] == 2 * N
    assert result.array.shape[1:] == (S, *MESH_SIZE)
    assert result.is_multi_batched()


def test_concat_snapshot_raises():
    """DensityField.concat on a plain snapshot (S, ...) field should raise ValueError mentioning stack."""
    S = 4
    f = ffi.DensityField(
        array=jnp.zeros((S, *MESH_SIZE)),
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        z_sources=jnp.array(1.0),
        scale_factors=jnp.array(0.8),
        comoving_centers=jnp.array(300.0),
        density_width=jnp.array(25.0),
        status=ffi.FieldStatus.DENSITY_FIELD,
        unit=ffi.DensityUnit.DENSITY,
    )
    with pytest.raises(ValueError, match="stack"):
        ffi.DensityField.concat([f])


def test_catalog_auto_expand_multi_batched():
    """(N=2, S=4, *MESH_SIZE) field + (N=2,) cosmo → Catalog with 2 entries of shape (S, *MESH_SIZE)."""
    N, S = 2, 4
    field = _make_multi_batched_density(N, S)
    cosmo = _make_batched_cosmo(N)
    cat = Catalog(field=field, cosmology=cosmo)
    assert len(cat) == N
    for entry in cat.field:
        assert entry.array.shape == (S, *MESH_SIZE)
        assert not entry.is_multi_batched()
        assert entry.is_batched()


def test_catalog_auto_expand_snapshot():
    """(S=3, *MESH_SIZE) field + (S=3,) cosmo → Catalog with 3 entries of shape MESH_SIZE."""
    S = 3
    rng = np.random.RandomState(7)
    array = jnp.asarray(rng.randn(S, *MESH_SIZE))
    z_sources = jnp.asarray(rng.uniform(0.5, 2.0, size=S))
    scale_factors = jnp.asarray(rng.uniform(0.5, 1.0, size=S))
    comoving_centers = jnp.asarray(rng.uniform(100, 500, size=S))
    density_width = jnp.asarray(rng.uniform(10, 50, size=S))
    field = ffi.DensityField(
        array=array,
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        halo_size=(0, 0),
        z_sources=z_sources,
        scale_factors=scale_factors,
        comoving_centers=comoving_centers,
        density_width=density_width,
        status=ffi.FieldStatus.DENSITY_FIELD,
        unit=ffi.DensityUnit.DENSITY,
    )
    cosmo = _make_batched_cosmo(S)
    cat = Catalog(field=field, cosmology=cosmo)
    assert len(cat) == S
    for entry in cat.field:
        assert entry.array.shape == MESH_SIZE
        assert not entry.is_batched()
        assert not entry.is_multi_batched()


def test_catalog_scalar_cosmo_multi_batched_raises():
    """Scalar cosmo + (N=2, S=3, ...) field → ValueError."""
    N, S = 2, 3
    field = _make_multi_batched_density(N, S)
    cosmo = jc.Planck18()
    with pytest.raises(ValueError, match="Scalar cosmology"):
        Catalog(field=field, cosmology=cosmo)


def test_catalog_multi_batched_round_trip(tmp_path):
    """(N=2, S=4, *MESH_SIZE) + (N=2,) cosmo → Catalog → parquet → reload, shape (S, *MESH_SIZE)."""
    N, S = 2, 4
    field = _make_multi_batched_density(N, S)
    cosmo = _make_batched_cosmo(N)
    cat = Catalog(field=field, cosmology=cosmo)
    assert len(cat) == N

    path = str(tmp_path / "multi_batched.parquet")
    cat.to_parquet(path)
    reloaded = Catalog.from_parquet(path)

    assert len(reloaded) == N
    for entry in reloaded.field:
        assert entry.array.shape == (S, *MESH_SIZE)
        assert entry.is_batched()
        assert not entry.is_multi_batched()
