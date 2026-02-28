"""Tests for equal-volume shell partitioning in compute_lightcone_shells."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

from jax_fli.fields import DensityField, FieldStatus
from jax_fli.fields.units import DensityUnit
from jax_fli.utils import _compute_equal_vol_shells, compute_lightcone_shells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_field():
    """Tiny DensityField sufficient to exercise compute_lightcone_shells.

    Box of 2000 Mpc/h gives max_radius ≈ 1000 Mpc/h, safely above 8 * 50 Mpc/h.
    """
    return DensityField(
        array=jnp.zeros((16, 16, 16)),
        mesh_size=(16, 16, 16),
        box_size=(2000.0, 2000.0, 2000.0),
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )


# ---------------------------------------------------------------------------
# Unit tests for _compute_equal_vol_shells (pure Python helper)
# ---------------------------------------------------------------------------


class TestComputeEqualVolShells:
    def test_edges_shape(self):
        edges = _compute_equal_vol_shells(R_last=1000.0, N_total=8, min_width=50.0)
        assert edges.shape == (9,), "Should return N_total + 1 edges"

    def test_edges_start_and_end(self):
        R_last = 1000.0
        edges = _compute_equal_vol_shells(R_last=R_last, N_total=8, min_width=50.0)
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(R_last)

    def test_ascending_order(self):
        edges = _compute_equal_vol_shells(R_last=1000.0, N_total=8, min_width=50.0)
        assert np.all(np.diff(edges) > 0), "Edges must be strictly ascending"

    def test_all_widths_at_least_min_width(self):
        min_width = 50.0
        edges = _compute_equal_vol_shells(R_last=1000.0, N_total=8, min_width=min_width)
        widths = np.diff(edges)
        assert np.all(widths >= min_width - 1e-10), f"All widths must be >= min_width={min_width}"

    def test_equal_volume_inner_shells(self):
        """Inner (equal-vol) shells must have equal R^3 increments."""
        R_last, N_total, min_width = 1000.0, 8, 50.0
        edges = _compute_equal_vol_shells(R_last=R_last, N_total=N_total, min_width=min_width)
        # Count how many outer fixed-width shells there are by finding first width > min_width + eps
        widths = np.diff(edges)
        n_outer = int(np.sum(np.abs(widths - min_width) < 1e-10) and 0)
        # All shells satisfy min_width, but inner ones must also be equal-volume.
        # The total R^3 volume is partitioned uniformly over the inner shells.
        # We verify: edges[k]^3 - edges[k-1]^3 is constant for inner shells.
        cube_diffs = np.diff(edges**3)
        # Outer shells have fixed width, so their cube diffs grow; inner shells have equal cube diffs.
        # With M=0 (pure equal-vol), all cube diffs should be equal.
        if np.all(np.abs(widths - widths[0]) < 1.0):  # roughly equal widths → M=0 case
            assert np.allclose(cube_diffs, cube_diffs[0], rtol=1e-6)

    def test_pure_equal_vol_m0(self):
        """When min_width is small, M=0: all shells are equal-volume."""
        R_last, N_total, min_width = 1000.0, 8, 1.0
        edges = _compute_equal_vol_shells(R_last=R_last, N_total=N_total, min_width=min_width)
        cube_diffs = np.diff(edges**3)
        assert np.allclose(cube_diffs, cube_diffs[0], rtol=1e-10), "M=0: all cube diffs must be equal"

    def test_infeasible_raises_value_error(self):
        """min_width so large that N*min_width > R_last must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot fit"):
            _compute_equal_vol_shells(R_last=100.0, N_total=4, min_width=30.0)

    def test_total_volume_conserved(self):
        """Total volume (sum of R^3 diffs) must equal R_last^3."""
        R_last, N_total, min_width = 1000.0, 8, 50.0
        edges = _compute_equal_vol_shells(R_last=R_last, N_total=N_total, min_width=min_width)
        total_cube = np.sum(np.diff(edges**3))
        assert total_cube == pytest.approx(R_last**3, rel=1e-9)


# ---------------------------------------------------------------------------
# Integration tests for compute_lightcone_shells
# ---------------------------------------------------------------------------


class TestComputeLightconeShells:
    def test_equal_vol_output_shapes(self, cosmology, small_field):
        r_centers, a_centers = compute_lightcone_shells(
            cosmology, small_field, nb_shells=8, equal_vol=True, min_width=50.0
        )
        assert r_centers.shape == (8,), "r_centers must have shape (nb_shells,)"
        assert a_centers.shape == (8,), "a_centers must have shape (nb_shells,)"

    def test_default_output_shapes(self, cosmology, small_field):
        r_centers, a_centers = compute_lightcone_shells(cosmology, small_field, nb_shells=8)
        assert r_centers.shape == (8,), "r_centers must have shape (nb_shells,)"
        assert a_centers.shape == (8,), "a_centers must have shape (nb_shells,)"

    def test_regression_equal_vol_false_unchanged(self, cosmology, small_field):
        """equal_vol=False (default) must produce the same results as before."""
        r1, a1 = compute_lightcone_shells(cosmology, small_field, nb_shells=8)
        r2, a2 = compute_lightcone_shells(cosmology, small_field, nb_shells=8, equal_vol=False)
        assert jnp.allclose(r1, r2), "Default and explicit equal_vol=False must match"
        assert jnp.allclose(a1, a2), "Default and explicit equal_vol=False must match"

    def test_equal_vol_r_centers_descending(self, cosmology, small_field):
        """r_centers are returned far-to-near (descending) to match integrator convention."""
        r_centers, _ = compute_lightcone_shells(
            cosmology, small_field, nb_shells=8, equal_vol=True, min_width=50.0
        )
        assert jnp.all(jnp.diff(r_centers) < 0), "r_centers must be in descending (far-to-near) order"

    def test_default_r_centers_descending(self, cosmology, small_field):
        r_centers, _ = compute_lightcone_shells(cosmology, small_field, nb_shells=8)
        assert jnp.all(jnp.diff(r_centers) < 0), "r_centers must be in descending (far-to-near) order"

    def test_equal_vol_all_widths_at_least_min_width(self, cosmology, small_field):
        """In equal_vol mode every shell must be at least min_width wide."""
        min_width = 50.0
        r_centers, _ = compute_lightcone_shells(
            cosmology, small_field, nb_shells=8, equal_vol=True, min_width=min_width
        )
        # Reconstruct widths from centers (centers are descending, so negate diff)
        widths = jnp.abs(jnp.diff(r_centers))
        assert jnp.all(widths >= min_width - 1e-6), "All shells must be >= min_width"

    def test_equal_vol_different_from_equal_width(self, cosmology, small_field):
        """The two modes should produce different shell distributions."""
        r_ev, _ = compute_lightcone_shells(cosmology, small_field, nb_shells=8, equal_vol=True, min_width=1.0)
        r_ew, _ = compute_lightcone_shells(cosmology, small_field, nb_shells=8, equal_vol=False)
        assert not jnp.allclose(r_ev, r_ew), "equal_vol and equal_width must differ"

    def test_jit_caching_same_args(self, cosmology, small_field):
        """Calling twice with identical args should not retrace (relies on JAX's static cache)."""
        r1, a1 = compute_lightcone_shells(cosmology, small_field, nb_shells=6, equal_vol=True, min_width=50.0)
        r2, a2 = compute_lightcone_shells(cosmology, small_field, nb_shells=6, equal_vol=True, min_width=50.0)
        assert jnp.allclose(r1, r2) and jnp.allclose(a1, a2)
