"""fli-dorian-rt: post-process existing lightcone parquet files with dorian ray-tracing via MPI.

For each input parquet file, each row (cosmology run) is processed independently.
Only MPI rank 0 saves output. Memory is freed after each row.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def parser() -> ArgumentParser:
    """Build the argument parser for fli-dorian-rt."""
    p = ArgumentParser(
        prog="fli-dorian-rt",
        description="Post-process lightcone parquet files with dorian ray-tracing (MPI).",
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="FILE_OR_GLOB",
        help="Input parquet file(s) — single path or shell glob (e.g. 'results/*.parquet')",
    )
    p.add_argument("--output", "-o", default=".", metavar="DIR", help="Output directory (default: .)")
    p.add_argument(
        "--nz-shear",
        nargs="+",
        required=True,
        metavar="Z",
        help="Source redshifts or 's3'/'s3[i]'/'s3[start:stop]' for Stage-3 distributions",
    )
    p.add_argument("--min-z", type=float, default=0.01, help="Minimum redshift for nz integration (default: 0.01)")
    p.add_argument("--max-z", type=float, default=1.5, help="Maximum redshift for nz integration (default: 1.5)")
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
    return p


def main() -> None:
    """CLI entry point registered as fli-dorian-rt."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"MPI rank={rank}, size={size}")

    from jax_fli.io import Catalog
    from jax_fli.scripts._common import _resolve_nz_shear

    p = parser()
    args = p.parse_args()

    nz_shear = _resolve_nz_shear(args)

    import datasets

    import jax_fli as jfli

    if rank == 0:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output)

    min_z = args.min_z
    max_z = args.max_z
    n_integrate = args.n_integrate
    interp = args.rt_interp
    parallel_transport = not args.no_parallel_transport

    ds = datasets.load_dataset("parquet", data_files=args.input, split="train", streaming=True).with_format("numpy")
    row_count = 0
    for i, row in enumerate(ds):
        catalog = Catalog.from_dataset(row, sharding=None)
        field = catalog.field[0]
        cosmo = catalog.cosmology[0]

        if rank == 0:
            print(f"  row {i}: field={type(field).__name__} cosmo=Oc={float(cosmo.Omega_c):.4f}")

        kappa_rt, _ = jfli.raytrace(
            cosmo,
            field,
            nz_shear,
            born=False,
            raytrace=True,
            min_z=min_z,
            max_z=max_z,
            n_integrate=n_integrate,
            interp=interp,
            parallel_transport=parallel_transport,
            comm=comm,
        )

        if rank == 0:
            out_path = (
                output_dir
                / f"RAYTRACE_M_{field.mesh_size[0]}_B_{int(field.box_size[0])}_N_{field.nside}_row{i:04d}.parquet"
            )
            Catalog(field=kappa_rt, cosmology=cosmo).to_parquet(str(out_path))
            print(f"    Saved raytrace kappa → {out_path}")

        del kappa_rt, field, cosmo, catalog
        row_count += 1

        if comm is not None:
            comm.Barrier()  # Ensure all ranks have finished processing the current row before moving on

    if rank == 0:
        print(f"  Done: {row_count} row(s)")
        print("\nAll files processed.")


if __name__ == "__main__":
    main()
