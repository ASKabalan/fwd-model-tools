"""Tests for the fli-simulate CLI entry point.

Strategy: inject sys.argv, call main(), verify parquet output exists,
reload via Catalog.from_parquet(), check shapes and metadata consistency.

Uses small meshes (16^3, nside=16, order=1) for speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
    from dorian.lensing import raytrace_from_density  # noqa: F401

    HAS_DORIAN = True
except ImportError:
    HAS_DORIAN = False

COMMON = "--mesh-size 16 16 16 --box-size 100 100 100 --nside 16 --order 1 --t0 0.1"
NBODY_EXTRA = "--t1 1.0 --dt0 0.1"


def _run(args_str: str, tmp_path: Path):
    from jax_fli.io import Catalog
    from jax_fli.scripts.fli_simulate import main

    out = tmp_path / "out.parquet"
    sys.argv = ["fli-simulate"] + args_str.split() + ["-o", str(out)]
    main()
    assert out.exists(), f"Output file not created: {out}"
    cat = Catalog.from_parquet(str(out))
    assert len(cat) == 1, f"Expected 1 catalog entry, got {len(cat)}"
    return cat


def _check_metadata(cat):
    """Verify all dynamic metadata arrays have the same length as batch_size."""
    field = cat.field[0]
    arr = field.array
    batch = arr.shape[0] if arr.ndim > 1 else 1
    for name in ("scale_factors", "comoving_centers", "density_width"):
        val = np.atleast_1d(np.asarray(getattr(field, name, np.zeros(1))))
        assert val.shape[0] == batch, f"{name} has length {val.shape[0]}, expected batch_size={batch}"


# ---------------------------------------------------------------------------
# LPT subcommand
# ---------------------------------------------------------------------------


def test_lpt_nb_shells(tmp_path):
    """lpt with --nb-shells: automatic shell spacing."""
    cat = _run(f"lpt {COMMON} --nb-shells 4", tmp_path)
    _check_metadata(cat)


def test_lpt_ts(tmp_path):
    """lpt with explicit --ts scale factors."""
    cat = _run(f"lpt {COMMON} --ts 0.3 0.5 0.7", tmp_path)
    _check_metadata(cat)
    assert cat.field[0].array.shape[0] == 3


def test_lpt_ts_near_far(tmp_path):
    """lpt with --ts-near/--ts-far pairs (2 shell ranges)."""
    cat = _run(f"lpt {COMMON} --ts-near 0.3 0.5 --ts-far 0.5 0.9", tmp_path)
    _check_metadata(cat)
    assert cat.field[0].array.shape[0] == 2


# ---------------------------------------------------------------------------
# NBody subcommand
# ---------------------------------------------------------------------------


def test_nbody_nb_shells(tmp_path):
    """nbody with --nb-shells: lightcone output."""
    cat = _run(f"nbody {COMMON} {NBODY_EXTRA} --nb-shells 4", tmp_path)
    _check_metadata(cat)


def test_nbody_ts(tmp_path):
    """nbody with explicit --ts scale factors."""
    cat = _run(f"nbody {COMMON} {NBODY_EXTRA} --ts 0.5 0.8 1.0", tmp_path)
    _check_metadata(cat)
    assert cat.field[0].array.shape[0] == 3


def test_nbody_ts_near_far(tmp_path):
    """nbody with --ts-near/--ts-far pairs (2 shell ranges)."""
    cat = _run(f"nbody {COMMON} {NBODY_EXTRA} --ts-near 0.3 0.5 --ts-far 0.5 0.9", tmp_path)
    _check_metadata(cat)
    assert cat.field[0].array.shape[0] == 2


# ---------------------------------------------------------------------------
# Lensing (Born approximation) subcommand
# ---------------------------------------------------------------------------


def test_lensing_born_scalar(tmp_path):
    """lensing born with single scalar source redshift."""
    cat = _run(f"lensing {COMMON} {NBODY_EXTRA} --nb-shells 4 --nz-shear 1.0 --born", tmp_path)
    _check_metadata(cat)


def test_lensing_born_multi(tmp_path):
    """lensing born with multiple scalar source redshifts."""
    cat = _run(f"lensing {COMMON} {NBODY_EXTRA} --ts 0.5 0.8 --nz-shear 0.5 1.0 1.5 --born", tmp_path)
    field = cat.field[0]
    assert field.array.shape[0] == 3, f"Expected 3 source bins, got {field.array.shape[0]}"
    _check_metadata(cat)


def test_lensing_born_stage3(tmp_path):
    """lensing born with Stage 3 (4-bin) n(z) distributions."""
    cat = _run(
        f"lensing {COMMON} {NBODY_EXTRA} --ts-near 0.3 0.5 --ts-far 0.5 0.9 --nz-shear stage3 --born",
        tmp_path,
    )
    field = cat.field[0]
    assert field.array.shape[0] == 4, f"Expected 4 Stage-3 source bins, got {field.array.shape[0]}"
    _check_metadata(cat)


# ---------------------------------------------------------------------------
# Lensing (raytrace via dorian) subcommand
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DORIAN, reason="dorian not installed")
def test_lensing_raytrace_scalar(tmp_path):
    """raytrace with multiple scalar source redshifts."""
    cat = _run(f"lensing {COMMON} {NBODY_EXTRA} --nb-shells 4 --nz-shear 0.01 0.02 --raytrace", tmp_path)
    field = cat.field[0]
    assert field.array.shape[0] == 2, f"Expected 2 source bins, got {field.array.shape[0]}"
    _check_metadata(cat)


@pytest.mark.skipif(not HAS_DORIAN, reason="dorian not installed")
def test_lensing_raytrace_single(tmp_path):
    """raytrace with a single scalar source redshift."""
    cat = _run(f"lensing {COMMON} {NBODY_EXTRA} --nb-shells 4 --nz-shear 1.0 --raytrace", tmp_path)
    _check_metadata(cat)
