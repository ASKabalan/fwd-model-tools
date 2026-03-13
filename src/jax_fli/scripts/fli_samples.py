"""fli-samples: generate prior-predictive samples from a probabilistic model."""

from __future__ import annotations

import argparse

import jax
import jax_cosmo as jc
from numpyro.infer import Predictive

import jax_fli as jfli
from jax_fli.scripts._common import _build_sharding, _resolve_nz_shear

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fli-samples."""
    p = argparse.ArgumentParser(
        prog="fli-samples",
        description="Generate prior-predictive samples from a probabilistic model.",
    )

    p.add_argument(
        "--model",
        choices=["full", "mock"],
        default="full",
        help="Probabilistic model to sample from: 'full' (full_field_probmodel) or 'mock' (mock_probmodel). (default: full)",
    )

    # Mesh / box
    p.add_argument(
        "--mesh-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        metavar=("NX", "NY", "NZ"),
        help="Mesh resolution (default: 64 64 64)",
    )
    p.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=[200.0, 200.0, 200.0],
        metavar=("LX", "LY", "LZ"),
        help="Box side lengths in Mpc/h (default: 200 200 200)",
    )
    p.add_argument(
        "--pdim",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("PX", "PY"),
        help="Process mesh dimensions (default: 1 1 = single device)",
    )
    p.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    p.add_argument(
        "--halo-fraction",
        type=int,
        default=8,
        metavar="F",
        help="Halo size as mesh // fraction for distributed painting (default: 8)",
    )
    p.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        metavar=("OX", "OY", "OZ"),
        help="Observer position in box coordinates (default: 0.5 0.5 0.5)",
    )

    # Geometry (mutually exclusive)
    geom_group = p.add_mutually_exclusive_group()
    geom_group.add_argument(
        "--nside",
        type=int,
        default=None,
        help="HEALPix NSIDE for spherical painting",
    )
    geom_group.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Flat-sky pixel resolution (height width)",
    )

    # Simulation parameters
    p.add_argument("--lpt-order", type=int, choices=[1, 2], default=2, help="LPT order (default: 2)")
    p.add_argument("--t0", type=float, default=0.01, help="Start scale factor (default: 0.01)")
    p.add_argument("--t1", type=float, default=1.0, help="End scale factor (default: 1.0)")
    p.add_argument(
        "--nb-steps",
        type=int,
        default=100,
        dest="nb_steps",
        help="Number of integration steps (>= 2); dt0 = (t1 - t0) / (nb_steps - 1). (default: 100)",
    )
    p.add_argument("--nb-shells", type=int, default=8, help="Number of lightcone shells (default: 8)")
    p.add_argument(
        "--density-widths", type=float, nargs="+", default=None, metavar="W", help="Override shell widths (Mpc/h)"
    )
    # Shell spec alternatives
    ts_group = p.add_mutually_exclusive_group()
    ts_group.add_argument(
        "--ts", type=float, nargs="+", default=None, metavar="A", help="Scale factors for snapshot/shell output"
    )
    ts_group.add_argument(
        "--ts-near",
        type=float,
        nargs="+",
        default=None,
        metavar="A_NEAR",
        help="Near scale factor edge(s) (use with --ts-far)",
    )
    p.add_argument(
        "--ts-far",
        type=float,
        nargs="+",
        default=None,
        metavar="A_FAR",
        help="Far scale factor edge(s) (use with --ts-near)",
    )
    p.add_argument(
        "--interp",
        choices=["none", "onion", "telephoto"],
        default="none",
        help="Interpolation kernel (default: none)",
    )
    p.add_argument(
        "--scheme",
        choices=["ngp", "bilinear", "rbf_neighbor"],
        default="bilinear",
        help="Spherical painting interpolation scheme (default: bilinear)",
    )
    p.add_argument(
        "--paint-nside",
        type=int,
        default=None,
        dest="paint_nside",
        help="Override nside used for painting (default: same as --nside)",
    )
    p.add_argument(
        "--drift-on-lightcone", action="store_true", help="Apply drift correction when painting lightcone shells"
    )
    p.add_argument("--equal-vol", action="store_true", default=False, help="Use equal-volume shell partitioning")
    p.add_argument(
        "--min-width", type=float, default=50.0, dest="min_width", help="Minimum shell width in Mpc/h (default: 50.0)"
    )

    # Lensing / noise
    p.add_argument(
        "--nz-shear",
        nargs="+",
        default=["s3"],
        metavar="Z",
        help="Source redshift bins: 's3'/'s3[i]'/'s3[start:stop]' or floats (default: s3)",
    )
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape-noise dispersion (default: 0.26)")
    p.add_argument(
        "--density-plane-smoothing",
        type=float,
        default=0.0,
        help="Density plane smoothing scale (default: 0.0)",
    )
    p.add_argument("--min-z", type=float, default=0.01, help="Minimum redshift for nz integration (default: 0.01)")
    p.add_argument("--max-z", type=float, default=1.5, help="Maximum redshift for nz integration (default: 1.5)")
    p.add_argument(
        "--n-integrate", type=int, default=32, help="Number of integration points for nz distributions (default: 32)"
    )

    # Sampling
    p.add_argument("--num-samples", type=int, default=100, help="Number of prior-predictive samples (default: 100)")
    p.add_argument("--seed", type=int, default=0, help="JAX PRNGKey seed (default: 0)")

    # Output
    p.add_argument("--path", type=str, required=True, help="Output directory")
    p.add_argument("--batch-id", type=int, default=0, help="Batch index written into output filenames (default: 0)")

    # Precision
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as fli-samples."""
    args = parser().parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    # --- validate geometry ---
    if args.nside is None and args.flatsky_npix is None:
        parser().error("One of --nside or --flatsky-npix is required.")
    if args.nside is not None and args.flatsky_npix is not None:
        parser().error("Only one of --nside or --flatsky-npix can be specified.")

    # --- resolve nz_shear ---
    nz_shear = _resolve_nz_shear(args)

    priors = {
        "Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5),
        "sigma8": jfli.infer.PreconditionnedUniform(0.6, 1.0),
    }

    # --- determine geometry ---
    if args.flatsky_npix is not None:
        geometry, nside, flatsky_npix = "flat", None, tuple(args.flatsky_npix)
    else:
        geometry, nside, flatsky_npix = "spherical", args.nside, None

    # -- determine sharding ---
    sharding = _build_sharding(args)

    # --- compute halo_size ---
    mesh = tuple(args.mesh_size)
    halo_size = (mesh[0] // args.halo_fraction, mesh[1] // args.halo_fraction)

    # --- build Configurations ---
    config = jfli.ppl.Configurations(
        mesh_size=mesh,
        box_size=tuple(args.box_size),
        nside=nside,
        flatsky_npix=flatsky_npix,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        density_plane_smoothing=args.density_plane_smoothing,
        halo_size=halo_size,
        observer_position=tuple(args.observer_position),
        t0=args.t0,
        nb_steps=args.nb_steps,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        geometry=geometry,
        scheme=args.scheme,
        paint_nside=args.paint_nside,
        sharding=sharding,
        lensing="born",
        drift_on_lightcone=args.drift_on_lightcone,
        equal_vol=args.equal_vol,
        min_width=args.min_width,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
    )

    # --- select model ---
    if args.model == "full":
        model = jfli.ppl.full_field_probmodel(config)
    else:
        model = jfli.ppl.mock_probmodel(config)

    # --- sample with NumPyro Predictive ---
    rng_key = jax.random.PRNGKey(args.seed)
    pred = Predictive(model, num_samples=args.num_samples)
    samples = pred(rng_key)

    print(f"sharding {samples['initial_conditions'].array.sharding} samples with {config.sharding}...")
    # --- save via sample2catalog ---
    saving_fn = jfli.infer.sample2catalog(config)
    saving_fn(samples, args.path, args.batch_id)


if __name__ == "__main__":
    main()
