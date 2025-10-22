import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from fwd_model_tools.sampling import batched_sampling, load_samples


class TestLoadSamples:
    def test_load_samples_with_scalar_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz", num_steps=100, acceptance_rate=0.85)
            np.savez(f"{tmpdir}/samples_1.npz", num_steps=105, acceptance_rate=0.82)

            samples = load_samples(tmpdir)

            assert "num_steps" in samples
            assert "acceptance_rate" in samples
            assert samples["num_steps"].shape == (2,)
            assert samples["acceptance_rate"].shape == (2,)
            assert jnp.allclose(samples["num_steps"], jnp.array([100, 105]))
            assert jnp.allclose(samples["acceptance_rate"], jnp.array([0.85, 0.82]))

    def test_load_samples_with_1d_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(
                f"{tmpdir}/samples_0.npz", Omega_c=np.array([0.26, 0.27, 0.28]), sigma8=np.array([0.81, 0.82, 0.80])
            )
            np.savez(f"{tmpdir}/samples_1.npz", Omega_c=np.array([0.265, 0.275]), sigma8=np.array([0.815, 0.805]))

            samples = load_samples(tmpdir)

            assert samples["Omega_c"].shape == (5,)
            assert samples["sigma8"].shape == (5,)
            assert jnp.allclose(samples["Omega_c"], jnp.array([0.26, 0.27, 0.28, 0.265, 0.275]))
            assert jnp.allclose(samples["sigma8"], jnp.array([0.81, 0.82, 0.80, 0.815, 0.805]))

    def test_load_samples_with_multidimensional_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_conds_0 = np.random.randn(10, 8, 8, 8)
            initial_conds_1 = np.random.randn(5, 8, 8, 8)

            np.savez(f"{tmpdir}/samples_0.npz", initial_conditions=initial_conds_0)
            np.savez(f"{tmpdir}/samples_1.npz", initial_conditions=initial_conds_1)

            samples = load_samples(tmpdir)

            assert samples["initial_conditions"].shape == (15, 8, 8, 8)
            assert jnp.allclose(samples["initial_conditions"][:10], initial_conds_0)
            assert jnp.allclose(samples["initial_conditions"][10:], initial_conds_1)

    def test_load_samples_with_mixed_scalar_and_array_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz", Omega_c=np.array([0.26, 0.27]), num_steps=100)
            np.savez(f"{tmpdir}/samples_1.npz", Omega_c=np.array([0.28, 0.29]), num_steps=105)

            samples = load_samples(tmpdir)

            assert samples["Omega_c"].shape == (4,)
            assert samples["num_steps"].shape == (2,)
            assert jnp.allclose(samples["Omega_c"], jnp.array([0.26, 0.27, 0.28, 0.29]))
            assert jnp.allclose(samples["num_steps"], jnp.array([100, 105]))

    def test_load_samples_with_param_names_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(
                f"{tmpdir}/samples_0.npz", Omega_c=np.array([0.26, 0.27]), sigma8=np.array([0.81, 0.82]), num_steps=100
            )

            samples = load_samples(tmpdir, param_names=["Omega_c"])

            assert "Omega_c" in samples
            assert "sigma8" not in samples
            assert "num_steps" not in samples
            assert samples["Omega_c"].shape == (2,)

    def test_load_samples_with_multiple_param_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(
                f"{tmpdir}/samples_0.npz",
                Omega_c=np.array([0.26, 0.27]),
                sigma8=np.array([0.81, 0.82]),
                h=np.array([0.67, 0.68]),
                num_steps=100,
            )

            samples = load_samples(tmpdir, param_names=["Omega_c", "sigma8"])

            assert "Omega_c" in samples
            assert "sigma8" in samples
            assert "h" not in samples
            assert "num_steps" not in samples

    def test_load_samples_missing_param_in_some_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(f"{tmpdir}/samples_0.npz", Omega_c=np.array([0.26, 0.27]), sigma8=np.array([0.81, 0.82]))
            np.savez(f"{tmpdir}/samples_1.npz", Omega_c=np.array([0.28, 0.29]))

            samples = load_samples(tmpdir)

            assert "Omega_c" in samples
            assert samples["Omega_c"].shape == (4,)
            assert "sigma8" in samples
            assert samples["sigma8"].shape == (2,)

    def test_load_samples_no_files_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No sample files found"):
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


class TestBatchedSampling:
    @pytest.fixture
    def simple_model(self):
        def model():
            mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
            sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.3))
            numpyro.sample("x", dist.Normal(mu, sigma))

        return model

    @pytest.mark.parametrize(
        "backend,sampler",
        [
            ("numpyro", "NUTS"),
            ("blackjax", "NUTS"),
            # ("blackjax", "MCLMC"),
        ],
    )
    def test_batched_vs_nonbatched_equivalence(self, simple_model, backend, sampler):
        with tempfile.TemporaryDirectory() as tmpdir_1batch:
            with tempfile.TemporaryDirectory() as tmpdir_10batch:
                seed = 42
                num_warmup = 100
                total_samples = 1000
                samples_per_batch = 100

                batched_sampling(
                    model=simple_model,
                    path=tmpdir_1batch,
                    rng_key=jax.random.PRNGKey(seed),
                    num_warmup=num_warmup,
                    num_samples=total_samples,
                    batch_count=1,
                    sampler=sampler,
                    backend=backend,
                    save=True,
                )

                batched_sampling(
                    model=simple_model,
                    path=tmpdir_10batch,
                    rng_key=jax.random.PRNGKey(seed),
                    num_warmup=num_warmup,
                    num_samples=samples_per_batch,
                    batch_count=10,
                    sampler=sampler,
                    backend=backend,
                    save=True,
                )

                samples_1batch = load_samples(tmpdir_1batch, param_names=["mu", "sigma", "x"])
                samples_10batch = load_samples(tmpdir_10batch, param_names=["mu", "sigma", "x"])
                print(f"num samples 1batch: {samples_1batch['mu'].shape[0]}")
                print(f"num samples 10batch: {samples_10batch['mu'].shape[0]}")

                assert samples_1batch["mu"].shape[0] == total_samples
                assert samples_10batch["mu"].shape[0] == total_samples

                mean_1batch = {k: float(v.mean()) for k, v in samples_1batch.items()}
                mean_10batch = {k: float(v.mean()) for k, v in samples_10batch.items()}
                std_1batch = {k: float(v.std()) for k, v in samples_1batch.items()}
                std_10batch = {k: float(v.std()) for k, v in samples_10batch.items()}

                print(f"Means 1 batch: {mean_1batch} vs 10 batch: {mean_10batch}")
                print(f"Stds 1 batch: {std_1batch} vs 10 batch: {std_10batch}")

                for param in ["mu", "sigma", "x"]:
                    mean_diff = abs(mean_1batch[param] - mean_10batch[param])
                    std_diff = abs(std_1batch[param] - std_10batch[param])
                    print(f"{param}: mean diff = {mean_diff}, std diff = {std_diff}")

                    assert mean_diff < 0.2, f"{param}: mean difference {mean_diff} too large"
                    assert std_diff < 0.2, f"{param}: std difference {std_diff} too large"

    @pytest.mark.parametrize(
        "backend,sampler",
        [
            ("numpyro", "NUTS"),
            ("blackjax", "NUTS"),
        ],
    )
    def test_batched_sampling_produces_valid_samples(self, simple_model, backend, sampler):
        with tempfile.TemporaryDirectory() as tmpdir:
            seed = 42
            num_warmup = 50
            samples_per_batch = 50
            batch_count = 4
            total_samples = samples_per_batch * batch_count

            batched_sampling(
                model=simple_model,
                path=tmpdir,
                rng_key=jax.random.PRNGKey(seed),
                num_warmup=num_warmup,
                num_samples=samples_per_batch,
                batch_count=batch_count,
                sampler=sampler,
                backend=backend,
                save=True,
            )

            samples = load_samples(tmpdir, param_names=["mu", "sigma", "x"])

            assert samples["mu"].shape[0] == total_samples
            assert samples["sigma"].shape[0] == total_samples
            assert samples["x"].shape[0] == total_samples

            assert jnp.isfinite(samples["mu"]).all()
            assert jnp.isfinite(samples["sigma"]).all()
            assert jnp.isfinite(samples["x"]).all()

            assert -3.0 < float(samples["mu"].mean()) < 3.0
            assert 0.5 < float(samples["sigma"].mean()) < 2.0

    def test_batched_sampling_saves_correct_number_of_files(self, simple_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_count = 5
            batched_sampling(
                model=simple_model,
                path=tmpdir,
                rng_key=jax.random.PRNGKey(0),
                num_warmup=20,
                num_samples=10,
                batch_count=batch_count,
                sampler="NUTS",
                backend="numpyro",
                save=True,
            )

            sample_files = sorted(Path(tmpdir).glob("samples_*.npz"))
            assert len(sample_files) == batch_count

            state_file = Path(tmpdir) / "sampling_state.pkl"
            assert state_file.exists()

    def test_batched_sampling_resume_from_checkpoint(self, simple_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            batched_sampling(
                model=simple_model,
                path=tmpdir,
                rng_key=jax.random.PRNGKey(42),
                num_warmup=20,
                num_samples=10,
                batch_count=3,
                sampler="NUTS",
                backend="numpyro",
                save=True,
            )

            sample_files = sorted(Path(tmpdir).glob("samples_*.npz"))
            assert len(sample_files) == 3

            batched_sampling(
                model=simple_model,
                path=tmpdir,
                rng_key=jax.random.PRNGKey(42),
                num_warmup=20,
                num_samples=10,
                batch_count=5,
                sampler="NUTS",
                backend="numpyro",
                save=True,
            )

            sample_files_after = sorted(Path(tmpdir).glob("samples_*.npz"))
            assert len(sample_files_after) == 5

            samples = load_samples(tmpdir, param_names=["mu"])
            assert samples["mu"].shape[0] == 50

    def test_batched_sampling_concatenates_correctly(self, simple_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_count = 4
            samples_per_batch = 25

            batched_sampling(
                model=simple_model,
                path=tmpdir,
                rng_key=jax.random.PRNGKey(0),
                num_warmup=30,
                num_samples=samples_per_batch,
                batch_count=batch_count,
                sampler="NUTS",
                backend="numpyro",
                save=True,
            )

            samples = load_samples(tmpdir, param_names=["mu", "sigma", "x"])

            assert samples["mu"].shape[0] == batch_count * samples_per_batch
            assert samples["sigma"].shape[0] == batch_count * samples_per_batch
            assert samples["x"].shape[0] == batch_count * samples_per_batch

            assert jnp.isfinite(samples["mu"]).all()
            assert jnp.isfinite(samples["sigma"]).all()
            assert jnp.isfinite(samples["x"]).all()

    def test_init_params_with_multidimensional_field(self):
        def model_with_field():
            Omega_c = numpyro.sample("Omega_c", dist.Uniform(0.2, 0.4))
            sigma8 = numpyro.sample("sigma8", dist.Uniform(0.6, 1.0))
            ic = numpyro.sample("initial_conditions", dist.Normal(jnp.zeros((8, 8, 8)), jnp.ones((8, 8, 8))))
            numpyro.deterministic("sum_params", Omega_c + sigma8 + ic.mean())

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = 42
            num_warmup = 30
            samples_per_batch = 50
            batch_count = 1

            ic_init = jax.random.normal(jax.random.PRNGKey(123), (8, 8, 8))
            init_params = {
                "Omega_c": 0.26,
                "sigma8": 0.81,
                "initial_conditions": ic_init,
            }

            batched_sampling(
                model=model_with_field,
                path=tmpdir,
                rng_key=jax.random.PRNGKey(seed),
                num_warmup=num_warmup,
                num_samples=samples_per_batch,
                batch_count=batch_count,
                sampler="NUTS",
                backend="numpyro",
                save=True,
                init_params=init_params,
            )

            samples = load_samples(tmpdir, param_names=["Omega_c", "sigma8", "initial_conditions"])

            assert samples["Omega_c"].shape[0] == samples_per_batch
            assert samples["sigma8"].shape[0] == samples_per_batch
            assert samples["initial_conditions"].shape == (samples_per_batch, 8, 8, 8)

            assert jnp.isfinite(samples["Omega_c"]).all()
            assert jnp.isfinite(samples["sigma8"]).all()
            assert jnp.isfinite(samples["initial_conditions"]).all()

            mean_omega = float(samples["Omega_c"].mean())
            mean_sigma8 = float(samples["sigma8"].mean())

            assert 0.2 < mean_omega < 0.4, f"Omega_c mean {mean_omega} within prior bounds"
            assert 0.6 < mean_sigma8 < 1.0, f"sigma8 mean {mean_sigma8} within prior bounds"
