"""ffi-samples: generate prior-predictive samples from a probabilistic model."""

from __future__ import annotations

import argparse

import jax
import jax_cosmo as jc
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpyro.infer import Predictive

import fwd_model_tools as ffi

# ---------------------------------------------------------------------------
# Sharding setup
# ---------------------------------------------------------------------------


def _build_sharding(args: argparse.Namespace):
    """Return sharding or None for single-device runs."""

    print(f"jax devices: {jax.devices()}")
    pdim = tuple(args.pdim)
    if pdim == (1, 1):
        return None

    mesh = jax.make_mesh(pdim, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    return sharding


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ffi-samples."""
    p = argparse.ArgumentParser(
        prog="ffi-samples",
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

    # Lensing / noise
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape-noise dispersion (default: 0.26)")
    p.add_argument(
        "--density-plane-smoothing",
        type=float,
        default=0.0,
        help="Density plane smoothing scale (default: 0.0)",
    )

    # Time-stepping
    p.add_argument("--t0", type=float, default=0.01, help="Start scale factor (default: 0.01)")
    p.add_argument("--t1", type=float, default=1.0, help="End scale factor (default: 1.0)")
    p.add_argument("--dt0", type=float, default=0.01, help="Integration time step (default: 0.01)")
    p.add_argument("--lpt-order", type=int, choices=[1, 2], default=2, help="LPT order (default: 2)")
    p.add_argument("--nb-shells", type=int, default=8, help="Number of lightcone shells (default: 8)")

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
    """CLI entry point registered as ffi-samples."""
    args = parser().parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    # --- validate geometry ---
    if args.nside is None and args.flatsky_npix is None:
        parser().error("One of --nside or --flatsky-npix is required.")
    if args.nside is not None and args.flatsky_npix is not None:
        parser().error("Only one of --nside or --flatsky-npix can be specified.")

    # --- hardcoded priors and nz_shear ---
    nz_shear = ffi.io.get_stage3_nz_shear()
    priors = {
        "Omega_c": ffi.infer.PreconditionnedUniform(0.1, 0.5),
        "sigma8": ffi.infer.PreconditionnedUniform(0.6, 1.0),
    }

    # --- determine geometry ---
    if args.flatsky_npix is not None:
        geometry, nside, flatsky_npix = "flat", None, tuple(args.flatsky_npix)
    else:
        geometry, nside, flatsky_npix = "spherical", args.nside, None

    # -- determine sharding ---
    sharding = _build_sharding(args)

    # --- build Configurations ---
    config = ffi.ppl.Configurations(
        mesh_size=tuple(args.mesh_size),
        box_size=tuple(args.box_size),
        nside=nside,
        flatsky_npix=flatsky_npix,
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        density_plane_smoothing=args.density_plane_smoothing,
        t0=args.t0,
        dt0=args.dt0,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        geometry=geometry,
        sharding=sharding,
    )

    # --- select model ---
    if args.model == "full":
        model = ffi.ppl.full_field_probmodel(config)
    else:
        model = ffi.ppl.mock_probmodel(config)

    # --- sample with NumPyro Predictive ---
    rng_key = jax.random.PRNGKey(args.seed)
    pred = Predictive(model, num_samples=args.num_samples)
    samples = pred(rng_key)

    # --- save via sample2catalog ---
    saving_fn = ffi.ppl.sample2catalog(config)
    saving_fn(samples, args.path, args.batch_id)


if __name__ == "__main__":
    main()
