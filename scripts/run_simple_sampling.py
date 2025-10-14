#!/usr/bin/env python
"""
Minimal example using batched_sampling on a toy NumPyro model.

The model has three parameters and one deterministic site:
  - mu ~ Normal(0, 1)
  - sigma ~ LogNormal(0, 0.3)
  - x ~ Normal(mu, sigma)
  - y = deterministic = x**2

You can run with either backend (numpyro or blackjax) and collect the
deterministic site "y" into the saved npz batch files.

Usage examples:
  python -m fwd_model_tools.scripts.run_simple_sampling --output-dir output/simple \
         --backend blackjax --sampler NUTS --include-deterministic

  python -m fwd_model_tools.scripts.run_simple_sampling --output-dir output/simple_np \
         --backend numpyro --sampler NUTS --include-deterministic
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from fwd_model_tools.sampling import batched_sampling, load_samples


def toy_model():
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.3))
    x = numpyro.sample("x", dist.Normal(mu, sigma))
    numpyro.deterministic("y", x**2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simple batched sampling example")
    parser.add_argument("--output-dir",
                        type=str,
                        default="output/simple",
                        help="Directory to store samples and state")
    parser.add_argument("--backend",
                        type=str,
                        choices=["numpyro", "blackjax"],
                        default="numpyro",
                        help="Sampling backend")
    parser.add_argument("--sampler",
                        type=str,
                        choices=["NUTS", "HMC", "MCLMC"],
                        default="NUTS",
                        help="Sampler (MCLMC only available with blackjax)")
    parser.add_argument("--num-warmup",
                        type=int,
                        default=200,
                        help="Warmup steps")
    parser.add_argument("--num-samples",
                        type=int,
                        default=500,
                        help="Samples per batch")
    parser.add_argument("--batch-count",
                        type=int,
                        default=1,
                        help="Number of batches")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--include-deterministic",
                        action="store_true",
                        help="Also save deterministic sites (e.g., 'y')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Backend: {args.backend}, Sampler: {args.sampler}")
    print(f"Saving to: {args.output_dir}")

    batched_sampling(
        model=toy_model,
        path=args.output_dir,
        rng_key=jax.random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
    )

    print("\nLoading samples back...")
    samples = load_samples(args.output_dir)
    print("Available keys:", list(samples.keys()))
    for k in samples:
        arr = np.asarray(samples[k])
        mean = arr.mean()
        std = arr.std()
        print(f"  {k:>8s}: shape={arr.shape}, mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    main()
