"""fli-extract: stream MCMC catalogs and compute per-chain statistics."""

from __future__ import annotations

import argparse

import jax

from jax_fli.io.catalog import Catalog
from jax_fli.io.extract import extract_catalog
from jax_fli.scripts._common import _build_sharding


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-extract."""
    p = argparse.ArgumentParser(
        prog="fli-extract",
        description="Stream MCMC catalog parquet files and compute per-chain statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Source (mutually exclusive: local path or HF Hub) ---
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--path",
        type=str,
        metavar="PATH",
        help="Local root dir containing chain_N/samples/*.parquet (or samples/*.parquet for single-chain).",
    )
    source.add_argument(
        "--repo-id",
        type=str,
        metavar="REPO_ID",
        help="HuggingFace Hub repository ID (e.g. 'user/repo').",
    )

    # --- HF Hub config names (one per chain) ---
    p.add_argument(
        "--config",
        nargs="+",
        metavar="NAME",
        help="HF Hub dataset config names, one per chain (required with --repo-id).",
    )

    # --- Required ---
    p.add_argument(
        "--set-name",
        type=str,
        required=True,
        metavar="NAME",
        help="Name label for the CatalogExtract.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="PATH",
        help="Output parquet file path.",
    )
    p.add_argument(
        "--cosmo-keys",
        nargs="+",
        required=True,
        metavar="KEY",
        help="Cosmological parameter names to extract (e.g. Omega_c sigma8).",
    )

    # --- Optional ---
    p.add_argument(
        "--truth",
        type=str,
        default=None,
        metavar="PATH",
        help="Parquet path of a truth Catalog (used for truth_cosmo and power spectra reference).",
    )
    p.add_argument(
        "--field-statistic",
        action="store_true",
        help="Compute per-chain mean and std of the density fields.",
    )
    p.add_argument(
        "--power-statistic",
        action="store_true",
        help="Compute per-chain transfer function and coherence spectra (requires --truth).",
    )
    p.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Delta degrees of freedom for std computation (0 = population, 1 = sample).",
    )

    # --- Device mesh / distributed ---
    p.add_argument(
        "--pdim",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("PX", "PY"),
        help="Device mesh dimensions.",
    )
    p.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes.",
    )
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    return p


def main() -> None:
    """CLI entry point registered as fli-extract."""
    p = parser()
    args = p.parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    # Validate: --config required when --repo-id is set
    if args.repo_id is not None and args.config is None:
        p.error("--config is required when --repo-id is set.")

    # Validate: --truth required when --power-statistic is set
    if args.power_statistic and args.truth is None:
        p.error("--truth is required when --power-statistic is set.")

    # Build device sharding
    sharding = _build_sharding(args)

    # Load truth Catalog if provided
    truth = None
    if args.truth is not None:
        truth = Catalog.from_parquet(args.truth, sharding=sharding)

    # Run streaming extraction
    ce = extract_catalog(
        cosmo_keys=args.cosmo_keys,
        set_name=args.set_name,
        path=args.path,
        repo_id=args.repo_id,
        config=args.config,
        truth=truth,
        field_statistic=args.field_statistic,
        power_statistic=args.power_statistic,
        ddof=args.ddof,
        sharding=sharding,
    )
    if args.field_statistic and ce.mean_field is not None and jax.process_index() == 0:
        print(f"sharding of mean fields {ce.mean_field.array.sharding}")

    # Save to parquet
    ce.to_parquet(args.output)
    if jax.process_index() == 0:
        print(f"Saved CatalogExtract '{ce.name}' with {ce.n_chains} chains to {args.output}")


if __name__ == "__main__":
    main()
