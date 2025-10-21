#!/usr/bin/env python
"""
Simple field-based Bayesian inference workflow.

Complete workflow:
1. Define model: 16x16 initial condition field + 2 parameters (alpha, beta)
2. Forward model: produce linear, quadratic, and combined observables of the field
3. Generate synthetic observations by conditioning on true IC and parameters
4. Run MCMC sampling to infer IC and parameters from noisy observable
5. Analyze results: plot posteriors and IC comparison

Usage:
  python scripts/run_simple_sampling.py --output-dir output/simple --num-warmup 500 --num-samples 1000
  python scripts/run_simple_sampling.py --output-dir output/simple --plot-only  # analyze existing results
"""

import argparse
import os
import time
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition, seed, trace

from fwd_model_tools.plotting import plot_posterior
from fwd_model_tools.sampling import batched_sampling, load_samples

FIELD_SHAPE = (4, 4)
TRUE_ALPHA = 2.0
TRUE_BETA = 0.5
NOISE_STD = 0.1


def setup_output_dir(output_dir):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    samples_dir = output_dir / "samples"
    data_dir = output_dir / "data"

    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plots_dir, samples_dir, data_dir


def normalize_field(field, eps=1e-6):
    """Center and rescale a field to unit standard deviation to remove scale degeneracy."""
    centered = field - jnp.mean(field)
    std = jnp.std(centered)
    std = jnp.maximum(std, eps)
    return centered / std, std


def forward_model_components(ic, alpha, beta):
    """Compute separate forward-model contributions to make parameters identifiable."""
    linear_term = alpha * ic
    quadratic_term = beta * ic**2
    return linear_term, quadratic_term, linear_term + quadratic_term


def field_model(log_ic=False):
    ic_raw = numpyro.sample("initial_conditions", dist.Normal(jnp.zeros(FIELD_SHAPE), jnp.ones(FIELD_SHAPE)))
    ic, ic_scale = normalize_field(ic_raw)

    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 3.5))
    beta = numpyro.sample("beta", dist.Uniform(-1.0, 1.5))

    linear_term, quadratic_term, combined_term = forward_model_components(ic, alpha, beta)

    if log_ic:
        numpyro.deterministic("ic", ic)
        numpyro.deterministic("ic_scale", ic_scale)

    numpyro.sample("obs_linear", dist.Normal(linear_term, NOISE_STD))
    numpyro.sample("obs_quadratic", dist.Normal(quadratic_term, NOISE_STD))
    numpyro.sample("obs_combined", dist.Normal(combined_term, NOISE_STD))


def plot_field_comparison(true_field, samples_field, plots_dir):
    true_field = np.asarray(true_field)
    samples_field = np.asarray(samples_field)

    mean_field = samples_field.mean(axis=0)
    std_field = samples_field.std(axis=0)
    error_field = np.abs(mean_field - true_field)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    vmin_ic, vmax_ic = np.percentile(true_field, [2, 98])

    im0 = axes[0].imshow(true_field, origin="lower", cmap="viridis", vmin=vmin_ic, vmax=vmax_ic)
    axes[0].set_title("True IC")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mean_field, origin="lower", cmap="viridis", vmin=vmin_ic, vmax=vmax_ic)
    axes[1].set_title("Posterior Mean IC")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(std_field, origin="lower", cmap="plasma")
    axes[2].set_title("Posterior Std IC")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(error_field, origin="lower", cmap="hot")
    axes[3].set_title("|Mean - True|")
    plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.savefig(plots_dir / "ic_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("IC Error Statistics:")
    print(f"  Mean absolute error: {error_field.mean():.4f}")
    print(f"  Max absolute error: {error_field.max():.4f}")
    print(f"  Mean std: {std_field.mean():.4f}")


def generate_synthetic_observations(data_dir, plots_dir, magic_seed=42):
    print("\n" + "=" * 60)
    print("Step 1: Generating synthetic observations")
    print("=" * 60)

    key = jax.random.PRNGKey(magic_seed)
    true_ic_raw = jax.random.normal(key, FIELD_SHAPE)
    true_ic, true_ic_scale = normalize_field(true_ic_raw)

    def model_with_logging():
        return field_model(log_ic=True)

    conditioned_model = condition(
        model_with_logging, {"initial_conditions": true_ic_raw, "alpha": TRUE_ALPHA, "beta": TRUE_BETA}
    )

    print("Tracing model to generate synthetic observable...")
    start_time = time.time()
    model_trace = trace(seed(conditioned_model, magic_seed)).get_trace()
    elapsed = time.time() - start_time
    print(f"✓ Model traced in {elapsed:.2f}s")

    true_obs_linear = model_trace["obs_linear"]["value"]
    true_obs_quadratic = model_trace["obs_quadratic"]["value"]
    true_obs_combined = true_obs_linear + true_obs_quadratic

    np.savez(
        data_dir / "true_data.npz",
        ic=np.asarray(true_ic),
        ic_raw=np.asarray(true_ic_raw),
        ic_scale=float(true_ic_scale),
        alpha=TRUE_ALPHA,
        beta=TRUE_BETA,
        obs_linear=np.asarray(true_obs_linear),
        obs_quadratic=np.asarray(true_obs_quadratic),
        obs_combined=np.asarray(true_obs_combined),
    )
    print(f"✓ Saved true data to {data_dir / 'true_data.npz'}")
    print(f"  True IC scale (std before normalization): {float(true_ic_scale):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(np.asarray(true_ic), origin="lower", cmap="viridis")
    axes[0].set_title("True IC (unit std)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.asarray(true_obs_linear), origin="lower", cmap="magma")
    axes[1].set_title("Linear observable")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(np.asarray(true_obs_quadratic), origin="lower", cmap="magma")
    axes[2].set_title("Quadratic observable")
    plt.colorbar(im2, ax=axes[2])

    fig.text(
        0.5,
        0.02,
        f"True values: α = {TRUE_ALPHA:.2f}, β = {TRUE_BETA:.2f}",
        ha="center",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(plots_dir / "true_data.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plotted true data to {plots_dir / 'true_data.png'}")

    return {
        "obs_linear": true_obs_linear,
        "obs_quadratic": true_obs_quadratic,
        "obs_combined": true_obs_combined,
    }


def run_mcmc_inference(true_obs, samples_dir, args):
    print("\n" + "=" * 60)
    print("Step 2: Running MCMC inference")
    print("=" * 60)

    def model_for_inference():
        return field_model(log_ic=True)

    observed_model = condition(
        model_for_inference,
        {
            "obs_linear": true_obs["obs_linear"],
            "obs_quadratic": true_obs["obs_quadratic"],
            "obs_combined": true_obs["obs_combined"],
        },
    )

    print(f"Sampling with {args.sampler} using {args.backend} backend")
    print(f"Warmup: {args.num_warmup}, Samples: {args.num_samples}, Batches: {args.batch_count}")

    start_time = time.time()
    batched_sampling(
        model=observed_model,
        path=str(samples_dir),
        rng_key=jax.random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
        save=True,
    )
    elapsed = time.time() - start_time
    print(f"✓ MCMC sampling completed in {elapsed:.2f}s")


def analyze_results(samples_dir, data_dir, plots_dir):
    print("\n" + "=" * 60)
    print("Step 3: Loading samples and plotting results")
    print("=" * 60)

    samples = load_samples(str(samples_dir))
    print(f"Loaded parameters: {list(samples.keys())}")

    true_data = np.load(data_dir / "true_data.npz")
    true_alpha = float(true_data["alpha"])
    true_beta = float(true_data["beta"])
    true_ic = true_data["ic"]

    print("\nPosterior Statistics:")
    print(f"True alpha: {true_alpha:.4f}")
    print(f"Inferred alpha: {samples['alpha'].mean():.4f} ± {samples['alpha'].std():.4f}")
    print(f"True beta: {true_beta:.4f}")
    print(f"Inferred beta: {samples['beta'].mean():.4f} ± {samples['beta'].std():.4f}")

    if "ic" in samples:
        print("\nPlotting IC comparison...")
        plot_field_comparison(true_ic, samples["ic"], plots_dir)
        print(f"✓ Plotted IC comparison to {plots_dir / 'ic_comparison.png'}")

    param_samples = {"alpha": samples["alpha"], "beta": samples["beta"]}
    true_param_values = {"alpha": true_alpha, "beta": true_beta}
    plot_posterior(param_samples, plots_dir, params=("alpha", "beta"), true_values=true_param_values)
    print(f"✓ Plotted posteriors to {plots_dir / 'posterior_trace.png'} and {plots_dir / 'posterior_pair.png'}")


def main():
    parser = argparse.ArgumentParser(description="Run simple field-based Bayesian inference workflow")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/simple",
        help="Output directory for plots and samples",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=500,
        help="Number of warmup steps for MCMC",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples per batch",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=2,
        help="Number of batches to run",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["NUTS", "HMC", "MCLMC"],
        default="NUTS",
        help="MCMC sampler to use",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["numpyro", "blackjax"],
        default="blackjax",
        help="Sampling backend",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only analyze existing samples (skip data generation and sampling)",
    )

    args = parser.parse_args()

    output_dir, plots_dir, samples_dir, data_dir = setup_output_dir(args.output_dir)

    print("=" * 60)
    print("Simple Field-Based Bayesian Inference Workflow")
    print("=" * 60)
    print(f"Field shape: {FIELD_SHAPE}")
    print(f"True alpha: {TRUE_ALPHA}")
    print(f"True beta: {TRUE_BETA}")
    print(f"Noise std: {NOISE_STD}")
    print(f"Output directory: {output_dir}")

    if args.plot_only:
        print("\n⚠ Plot-only mode: skipping data generation and sampling")
        analyze_results(samples_dir, data_dir, plots_dir)
    else:
        true_obs = generate_synthetic_observations(data_dir, plots_dir, magic_seed=args.seed)
        run_mcmc_inference(true_obs, samples_dir, args)
        analyze_results(samples_dir, data_dir, plots_dir)

    print("\n" + "=" * 60)
    print("✓ Workflow completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
