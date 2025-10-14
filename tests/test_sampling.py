import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fwd_model_tools.sampling import load_samples


class TestLoadSamples:

    def test_load_samples_with_scalar_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz",
                     num_steps=100,
                     acceptance_rate=0.85)
            np.savez(f"{tmpdir}/samples_1.npz",
                     num_steps=105,
                     acceptance_rate=0.82)

            samples = load_samples(tmpdir)

            assert "num_steps" in samples
            assert "acceptance_rate" in samples
            assert samples["num_steps"].shape == (2, )
            assert samples["acceptance_rate"].shape == (2, )
            assert jnp.allclose(samples["num_steps"], jnp.array([100, 105]))
            assert jnp.allclose(samples["acceptance_rate"],
                                jnp.array([0.85, 0.82]))

    def test_load_samples_with_1d_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz",
                     Omega_c=np.array([0.26, 0.27, 0.28]),
                     sigma8=np.array([0.81, 0.82, 0.80]))
            np.savez(f"{tmpdir}/samples_1.npz",
                     Omega_c=np.array([0.265, 0.275]),
                     sigma8=np.array([0.815, 0.805]))

            samples = load_samples(tmpdir)

            assert samples["Omega_c"].shape == (5, )
            assert samples["sigma8"].shape == (5, )
            assert jnp.allclose(samples["Omega_c"],
                                jnp.array([0.26, 0.27, 0.28, 0.265, 0.275]))
            assert jnp.allclose(samples["sigma8"],
                                jnp.array([0.81, 0.82, 0.80, 0.815, 0.805]))

    def test_load_samples_with_multidimensional_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_conds_0 = np.random.randn(10, 8, 8, 8)
            initial_conds_1 = np.random.randn(5, 8, 8, 8)

            np.savez(f"{tmpdir}/samples_0.npz",
                     initial_conditions=initial_conds_0)
            np.savez(f"{tmpdir}/samples_1.npz",
                     initial_conditions=initial_conds_1)

            samples = load_samples(tmpdir)

            assert samples["initial_conditions"].shape == (15, 8, 8, 8)
            assert jnp.allclose(samples["initial_conditions"][:10],
                                initial_conds_0)
            assert jnp.allclose(samples["initial_conditions"][10:],
                                initial_conds_1)

    def test_load_samples_with_mixed_scalar_and_array_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz",
                     Omega_c=np.array([0.26, 0.27]),
                     num_steps=100)
            np.savez(f"{tmpdir}/samples_1.npz",
                     Omega_c=np.array([0.28, 0.29]),
                     num_steps=105)

            samples = load_samples(tmpdir)

            assert samples["Omega_c"].shape == (4, )
            assert samples["num_steps"].shape == (2, )
            assert jnp.allclose(samples["Omega_c"],
                                jnp.array([0.26, 0.27, 0.28, 0.29]))
            assert jnp.allclose(samples["num_steps"], jnp.array([100, 105]))

    def test_load_samples_with_param_names_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz",
                     Omega_c=np.array([0.26, 0.27]),
                     sigma8=np.array([0.81, 0.82]),
                     num_steps=100)

            samples = load_samples(tmpdir, param_names=["Omega_c"])

            assert "Omega_c" in samples
            assert "sigma8" not in samples
            assert "num_steps" not in samples
            assert samples["Omega_c"].shape == (2, )

    def test_load_samples_with_multiple_param_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz",
                     Omega_c=np.array([0.26, 0.27]),
                     sigma8=np.array([0.81, 0.82]),
                     h=np.array([0.67, 0.68]),
                     num_steps=100)

            samples = load_samples(tmpdir, param_names=["Omega_c", "sigma8"])

            assert "Omega_c" in samples
            assert "sigma8" in samples
            assert "h" not in samples
            assert "num_steps" not in samples

    def test_load_samples_missing_param_in_some_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz",
                     Omega_c=np.array([0.26, 0.27]),
                     sigma8=np.array([0.81, 0.82]))
            np.savez(f"{tmpdir}/samples_1.npz", Omega_c=np.array([0.28, 0.29]))

            samples = load_samples(tmpdir)

            assert "Omega_c" in samples
            assert samples["Omega_c"].shape == (4, )
            assert "sigma8" in samples
            assert samples["sigma8"].shape == (2, )

    def test_load_samples_no_files_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError,
                               match="No sample files found"):
                load_samples(tmpdir)

    def test_load_samples_empty_npz_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz")

            samples = load_samples(tmpdir)

            assert samples == {}

    def test_load_samples_sorts_files_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz", batch_id=np.array([0, 0, 0]))
            np.savez(f"{tmpdir}/samples_2.npz", batch_id=np.array([2, 2, 2]))
            np.savez(f"{tmpdir}/samples_1.npz", batch_id=np.array([1, 1, 1]))

            samples = load_samples(tmpdir)

            expected = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
            assert jnp.allclose(samples["batch_id"], expected)
