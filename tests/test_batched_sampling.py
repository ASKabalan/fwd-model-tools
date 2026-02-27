"""Tests for batched_sampling with mock_probmodel + sample2catalog callback.

Validates all 5 sampler configurations:
  NUTS/HMC × numpyro/blackjax  +  MCLMC/blackjax

Each test runs a tiny 4³ mesh with 2 samples and 1 batch, then checks that
the IC parquet is written and survives a Catalog round-trip.
"""

from __future__ import annotations

import jax
import jax_cosmo as jc
import pytest

datasets = pytest.importorskip("datasets")

import jax_fli as jfli
from jax_fli.fields import FieldStatus
from jax_fli.io.catalog import Catalog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MESH_SIZE = (4, 4, 4)
BOX_SIZE = (250.0, 250.0, 250.0)
NSIDE = 4

SAMPLER_BACKENDS = [
    ("NUTS", "numpyro"),
    ("HMC", "numpyro"),
    ("NUTS", "blackjax"),
    ("HMC", "blackjax"),
    ("MCLMC", "blackjax"),
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return jfli.ppl.Configurations(
        mesh_size=MESH_SIZE,
        box_size=BOX_SIZE,
        nside=NSIDE,
        fiducial_cosmology=jc.Planck18,
        nz_shear=[],  # no kappa bins → sample2catalog skips kappa saving
        priors={
            "Omega_c": jfli.sampling.PreconditionnedUniform(0.1, 0.5),
            "sigma8": jfli.sampling.PreconditionnedUniform(0.6, 1.0),
        },
        sigma_e=0.26,
        geometry="spherical",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler,backend", SAMPLER_BACKENDS)
def test_mock_model_catalog_saving(tmp_path, config, sampler, backend):
    """End-to-end: mock_probmodel → batched_sampling → sample2catalog → parquet round-trip."""
    model = jfli.ppl.mock_probmodel(config)

    # Obtain valid init_params by forward-tracing the model once.
    # This gives constrained sample values that init_to_value will inverse-transform
    # to the unconstrained space used by the samplers.
    from numpyro.handlers import seed
    from numpyro.handlers import trace as numpyro_trace

    model_trace = numpyro_trace(seed(model, 0)).get_trace()
    init_params = {k: v["value"] for k, v in model_trace.items() if v["type"] == "sample"}

    jfli.sampling.batched_sampling(
        model,
        path=str(tmp_path),
        rng_key=jax.random.PRNGKey(42),
        num_warmup=2,
        num_samples=2,
        batch_count=1,
        sampler=sampler,
        backend=backend,
        progress_bar=False,
        save_callback=jfli.ppl.sample2catalog(config),
        init_params=init_params,
    )

    # batched_sampling passes f"{path}/samples" as the path arg to save_callback;
    # sample2catalog then creates {path}/samples/samples/samples_{batch_id}.parquet
    ic_parquet = tmp_path / "samples" / "samples" / "samples_0.parquet"
    assert ic_parquet.exists(), f"IC parquet not saved at {ic_parquet}"

    # Round-trip: reload and verify field metadata
    catalog = Catalog.from_parquet(str(ic_parquet))
    ic_field = catalog.field[0]
    assert ic_field.mesh_size == config.mesh_size
    assert ic_field.status == FieldStatus.INITIAL_FIELD

    # No kappa parquet expected (nz_shear=[])
    kappa_dir = tmp_path / "samples" / "kappa_fields"
    assert not kappa_dir.exists(), "Kappa directory should not be created for empty nz_shear"
