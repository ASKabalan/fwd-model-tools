"""fli-raytrace: post-process existing lightcone parquet files with Born/raytrace lensing.

For each input parquet file, each row (cosmology run) is processed independently
and saved to BORN_<stem>.parquet or RAYTRACE_<stem>.parquet in the output directory.
Memory is freed after each row to stay frugal on large grids.
"""

from __future__ import annotations

import glob as glob_module
import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def parser() -> ArgumentParser:
    """Build the argument parser for fli-raytrace."""
    p = ArgumentParser(
        prog="fli-raytrace",
        description="Post-process lightcone parquet files with Born/raytrace lensing.",
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="FILE_OR_GLOB",
        help="Input parquet file(s) — single path or shell glob (e.g. 'results/*.parquet')",
    )
    p.add_argument("--output", "-o", default=".", metavar="DIR", help="Output directory (default: .)")
    p.add_argument(
        "--lensing",
        choices=["born", "raytrace", "both"],
        default="born",
        help="Lensing method: 'born' (default), 'raytrace', or 'both' (saves two files per row)",
    )
    p.add_argument(
        "--nz-shear",
        nargs="+",
        required=True,
        metavar="Z",
        help="Source redshifts or 's3'/'s3[i]'/'s3[start:stop]' for Stage-3 distributions",
    )
    p.add_argument("--min-z", type=float, default=0.01, help="Minimum redshift for nz integration (default: 0.01)")
    p.add_argument("--max-z", type=float, default=3.0, help="Maximum redshift for nz integration (default: 3.0)")
    p.add_argument(
        "--n-integrate", type=int, default=32, help="Number of integration points for nz distributions (default: 32)"
    )
    p.add_argument(
        "--rt-interp",
        choices=["bilinear", "ngp", "nufft"],
        default="bilinear",
        help="Interpolation method for raytrace (default: bilinear)",
    )
    p.add_argument("--no-parallel-transport", action="store_true", help="Disable parallel transport in raytrace")
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")
    return p


def _resolve_input_files(pattern: str) -> list[str]:
    """Expand a path or glob pattern to a sorted list of parquet files."""
    files = sorted(glob_module.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def main() -> None:
    """CLI entry point registered as fli-raytrace."""
    import jax

    from jax_fli.io import Catalog
    from jax_fli.scripts.fli_simulate import _resolve_nz_shear

    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)

    try:
        input_files = _resolve_input_files(args.input)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve nz_shear once (shared across all files/rows)
    nz_shear = _resolve_nz_shear(args)

    import datasets

    import jax_fli as jfli

    lensing = args.lensing
    min_z = args.min_z
    max_z = args.max_z
    n_integrate = args.n_integrate

    for file_path in input_files:
        stem = Path(file_path).stem
        print(f"\nProcessing {file_path} ...")

        # Stream row-by-row to stay memory-frugal
        ds = datasets.load_dataset("parquet", data_files=file_path, split="train", streaming=True)

        row_count = 0
        for i, row in enumerate(ds):
            catalog = Catalog.from_dataset(row)
            field = catalog.field[0]
            cosmo = catalog.cosmology[0]

            # For single-row files omit the row index suffix; add it from row 1 onward
            suffix = f"_row{i:04d}" if i > 0 else ""

            print(f"  row {i}: field={type(field).__name__} cosmo=Oc={float(cosmo.Omega_c):.4f}")

            if lensing in ("born", "both"):
                born_result = jax.block_until_ready(
                    jfli.born(cosmo, field, nz_shear, min_z=min_z, max_z=max_z, n_integrate=n_integrate)
                )
                out_path = output_dir / f"BORN_{stem}{suffix}.parquet"
                os.makedirs(out_path.parent, exist_ok=True)
                Catalog(field=born_result, cosmology=cosmo).to_parquet(str(out_path))
                print(f"    Saved Born kappa → {out_path}")
                del born_result

            if lensing in ("raytrace", "both"):
                kappa_rt, kappa_born = jax.block_until_ready(
                    jfli.raytrace(cosmo, field, nz_shear, born=(lensing == "both"), raytrace=True)
                )
                out_path_rt = output_dir / f"RAYTRACE_{stem}{suffix}.parquet"
                out_path_born = output_dir / f"RAYTRACE_BORN_{stem}{suffix}.parquet"
                os.makedirs(out_path_rt.parent, exist_ok=True)
                Catalog(field=kappa_rt, cosmology=cosmo).to_parquet(str(out_path_rt))
                print(f"    Saved raytrace kappa → {out_path_rt}")
                if lensing == "both":
                    Catalog(field=kappa_born, cosmology=cosmo).to_parquet(str(out_path_born))
                    print(f"    Saved raytrace-born kappa → {out_path_born}")
                del kappa_rt, kappa_born

            del field, cosmo, catalog
            row_count += 1

        print(f"  Done: {row_count} row(s) processed from {Path(file_path).name}")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
