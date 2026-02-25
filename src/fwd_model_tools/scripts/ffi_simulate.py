"""ffi-simulate: CLI entry point for the fwd_model_tools pipeline.

Provides subcommands:
  lpt      — IC → LPT → lightcone / particles
  nbody    — IC → LPT(particles) → NBody → lightcone / particles
  lensing  — IC → LPT → NBody → Born/raytrace → kappa

JAX is imported lazily (after argument parsing) so --help is instantaneous.
"""

import sys
import time
from argparse import ArgumentParser, Namespace
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import os
import fwd_model_tools as ffi
from jax.experimental.multihost_utils import sync_global_devices

# ---------------------------------------------------------------------------
# Cosmology builder
# ---------------------------------------------------------------------------


def _build_cosmo(args: Namespace):
    """Construct a jax_cosmo.Cosmology from CLI flags."""
    return jc.Cosmology(
        Omega_c=args.Omega_c,
        Omega_b=args.Omega_b,
        h=args.h,
        n_s=args.n_s,
        sigma8=args.sigma8,
        Omega_k=args.Omega_k,
        w0=args.w0,
        wa=args.wa,
        Omega_nu=args.Omega_nu,
    )


# ---------------------------------------------------------------------------
# Painting options builder
# ---------------------------------------------------------------------------


def _build_painting(args: Namespace):
    """Return a PaintingOptions and (nside, flatsky_npix) for IC generation."""
    from fwd_model_tools.fields import PaintingOptions

    nside = getattr(args, "nside", None)
    flatsky_npix = getattr(args, "flatsky_npix", None)
    density = getattr(args, "density", False)

    if nside is not None:
        return PaintingOptions(target="spherical"), nside, None
    elif flatsky_npix is not None:
        h, w = flatsky_npix
        return PaintingOptions(target="flat"), None, (h, w)
    elif density:
        return PaintingOptions(target="density"), None, None
    else:
        return PaintingOptions(target="particles"), None, None


# ---------------------------------------------------------------------------
# Snapshot/lightcone time-step resolver
# ---------------------------------------------------------------------------


def _resolve_ts(args: Namespace):
    """Return ts (jnp array or None) from --ts / --ts-near+far."""
    if args.ts is not None:
        return jnp.array(args.ts)
    if args.ts_near is not None and args.ts_far is not None:
        # Build (2, N) near/far array — convert lists from nargs="+" to JAX arrays first
        return jnp.stack([jnp.array(args.ts_near), jnp.array(args.ts_far)], axis=0)

    return None


# ---------------------------------------------------------------------------
# nz_shear resolver
# ---------------------------------------------------------------------------


def _resolve_nz_shear(args: Namespace):
    """Return nz_shear list from CLI --nz-shear values."""
    nz_shear = getattr(args, "nz_shear", None)
    if nz_shear is None:
        return None
    values = nz_shear
    if len(values) == 1 and values[0].lower() in ("stage3", "s3"):
        return ffi.io.get_stage3_nz_shear()
    # Otherwise parse as floats
    try:
        return jnp.array(values, dtype=jnp.float32)
    except ValueError as exc:
        raise ValueError(f"--nz-shear values must be floats or 'stage3': {values}") from exc


# ---------------------------------------------------------------------------
# Solver builder
# ---------------------------------------------------------------------------


def _build_solver(args: Namespace, painting):
    """Build an AbstractNBodySolver from CLI flags."""
    inter = getattr(args, "interp", "none")
    drift_on_lightcone = getattr(args, "drift_on_lightcone", False)
    if inter == "none":
        if drift_on_lightcone:
            interp_kernel = ffi.DriftInterp(painting=painting)
        else:
            interp_kernel = ffi.NoInterp(painting=painting)
    elif inter == "onion":
        if painting.target != "spherical":
            raise ValueError("--interp onion requires --nside (spherical painting target)")
        interp_kernel = ffi.OnionTiler(painting=painting, drift_on_lightcone=drift_on_lightcone)
    elif inter == "telephoto":
        interp_kernel = ffi.TelephotoInterp(painting=painting, drift_on_lightcone=drift_on_lightcone)
    else:
        raise ValueError(f"Unknown --interp value: {inter}")

    return ffi.ReversibleDoubleKickDrift(interp_kernel=interp_kernel)


# ---------------------------------------------------------------------------
# Sharding setup
# ---------------------------------------------------------------------------


def _build_sharding(args: Namespace):
    """Return sharding or None for single-device runs."""

    print(f"jax devices: {jax.devices()}")
    pdim = tuple(args.pdim)
    if pdim == (1, 1):
        return None

    mesh = jax.make_mesh(pdim, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    return sharding


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def _save_result(result, cosmo, args: Namespace) -> None:
    """Save result to parquet (process 0 only)."""
    # Create folder if it doesn't exist
    # If file has parent folders create them, otherwise do nothing
    parent_folder = os.path.dirname(args.output)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)
    catalog = ffi.io.Catalog(field=result, cosmology=cosmo)
    catalog.to_parquet(args.output)
    print(f"Saved to {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> ArgumentParser:
    """Build the full argparse tree with subcommands."""

    # ------------------------------------------------------------------
    # Common parent (shared by all subcommands)
    # ------------------------------------------------------------------
    common = ArgumentParser(add_help=False)

    # Mesh / box
    common.add_argument(
        "--mesh-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        metavar=("NX", "NY", "NZ"),
        help="Mesh resolution (default: 64 64 64)",
    )
    common.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=[200.0, 200.0, 200.0],
        metavar=("LX", "LY", "LZ"),
        help="Box side lengths in Mpc/h (default: 200 200 200)",
    )
    common.add_argument(
        "--pdim",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("PX", "PY"),
        help="Process mesh dimensions (default: 1 1 = single device)",
    )
    common.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    common.add_argument(
        "--halo-size",
        type=int,
        nargs=2,
        default=[0, 0],
        metavar=("H0", "H1"),
        help="Halo exchange depth for distributed painting (default: 0 0)",
    )
    common.add_argument('--observer-position',
                        type=float,
                        nargs=3,
                        default=[0.5, 0.5, 0.5],
                        metavar=('OX', 'OY', 'OZ'),
                        help='Observer position in box coordinates (default: 0.5 0.5 0.5, i.e. center of the box)')

    # Random seed and output
    common.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    common.add_argument("--output",
                        "-o",
                        default="sim_output.parquet",
                        help="Output file path (default: sim_output.parquet)")

    # Performance
    common.add_argument("--perf", action="store_true", help="Benchmark: warmup + N timed iterations")
    common.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=5,
        metavar="N",
        help="Number of timed iterations for --perf (default: 5)",
    )
    common.add_argument("--trace", action="store_true", help="Run with JAX profiler trace")
    common.add_argument("--trace-dir",
                        default="/tmp/jax_trace",
                        help="Directory for profiler trace (default: /tmp/jax_trace)")
    common.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision (default: False)")

    # Cosmology
    cosmo_group = common.add_argument_group("cosmology")
    cosmo_group.add_argument("--Omega-c", type=float, default=0.2589, help="Cold dark matter density (default: 0.2589)")
    cosmo_group.add_argument("--Omega-b", type=float, default=0.0486, help="Baryon density (default: 0.0486)")
    cosmo_group.add_argument("--h", type=float, default=0.6774, help="Dimensionless Hubble parameter (default: 0.6774)")
    cosmo_group.add_argument("--n-s", type=float, default=0.9667, dest="n_s", help="Spectral index (default: 0.9667)")
    cosmo_group.add_argument("--sigma8", type=float, default=0.8159, help="sigma8 (default: 0.8159)")
    cosmo_group.add_argument("--Omega-k", type=float, default=0.0, help="Curvature density (default: 0.0)")
    cosmo_group.add_argument("--w0", type=float, default=-1.0, help="Dark energy EOS w0 (default: -1.0)")
    cosmo_group.add_argument("--wa", type=float, default=0.0, help="Dark energy EOS wa (default: 0.0)")
    cosmo_group.add_argument("--Omega-nu", type=float, default=0.0, help="Neutrino density (default: 0.0)")

    # Painting target (mutually exclusive)
    paint_group = common.add_mutually_exclusive_group()
    paint_group.add_argument("--nside", type=int, default=None, help="HEALPix NSIDE for spherical painting")
    paint_group.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Flat-sky pixel resolution (height width)",
    )
    paint_group.add_argument("--field-size",
                             type=int,
                             nargs=2,
                             default=None,
                             metavar=("H", "W"),
                             help="2D field pixel resolution (alternative to --flatsky-npix)")
    paint_group.add_argument("--density", action="store_true", default=False, help="3D density field painting")

    # ------------------------------------------------------------------
    # LPT parent (shared by lpt, nbody, lensing)
    # ------------------------------------------------------------------
    lpt_parent = ArgumentParser(add_help=False)
    lpt_parent.add_argument("--t0", type=float, default=0.1, help="LPT starting scale factor (default: 0.1)")
    lpt_parent.add_argument("--order", type=int, default=2, choices=[1, 2], help="LPT order (default: 2)")
    lpt_parent.add_argument("--nb-shells", type=int, default=None, help="Number of lightcone shells")
    lpt_parent.add_argument("--density-widths",
                            type=float,
                            nargs="+",
                            default=None,
                            metavar="W",
                            help="Override shell widths (Mpc/h)")
    # Mutually exclusive ts group
    ts_group = lpt_parent.add_mutually_exclusive_group()
    ts_group.add_argument("--ts",
                          type=float,
                          nargs="+",
                          default=None,
                          metavar="A",
                          help="Scale factors for snapshot/shell output")
    ts_group.add_argument(
        "--ts-near",
        type=float,
        nargs="+",
        default=None,
        metavar="A_NEAR",
        help="Near scale factor edge(s) (use with --ts-far; one value per shell pair)",
    )
    lpt_parent.add_argument(
        "--ts-far",
        type=float,
        nargs="+",
        default=None,
        metavar="A_FAR",
        help="Far scale factor edge(s) (use with --ts-near; one value per shell pair)",
    )

    # ------------------------------------------------------------------
    # NBody parent (adds t1, dt0, solver, interp)
    # ------------------------------------------------------------------
    nbody_parent = ArgumentParser(add_help=False)
    nbody_parent.add_argument("--t1", type=float, default=1.0, help="NBody final scale factor (default: 1.0)")
    nbody_parent.add_argument("--dt0", type=float, default=0.05, help="Integration time step (default: 0.05)")
    nbody_parent.add_argument(
        "--interp",
        choices=["none", "onion", "telephoto"],
        default="none",
        help="Interpolation kernel (default: none)",
    )
    nbody_parent.add_argument("--drift-on-lightcone",
                              action="store_true",
                              help="Apply drift correction when painting lightcone shells")

    # ------------------------------------------------------------------
    # Top-level parser
    # ------------------------------------------------------------------
    parser = ArgumentParser(
        prog="ffi-simulate",
        description="fwd_model_tools simulation pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # lpt subcommand
    subparsers.add_parser(
        "lpt",
        parents=[common, lpt_parent],
        help="Run IC → LPT only",
        description="Generate initial conditions and apply LPT displacements.",
    )

    # nbody subcommand
    subparsers.add_parser(
        "nbody",
        parents=[common, lpt_parent, nbody_parent],
        help="Run IC → LPT(particles) → NBody",
        description="Run full N-body integration from LPT initial conditions.",
    )

    # lensing subcommand
    lensing_p = subparsers.add_parser(
        "lensing",
        parents=[common, lpt_parent, nbody_parent],
        help="Run IC → LPT → NBody → lensing (Born or raytrace)",
        description="Run full pipeline including weak lensing convergence maps.",
    )
    lensing_p.add_argument(
        "--nz-shear",
        nargs="+",
        required=True,
        metavar="Z",
        help="Source redshifts or 'stage3'/'s3' for 4-bin Stage 3 distributions",
    )
    lensing_method = lensing_p.add_mutually_exclusive_group()
    lensing_method.add_argument("--born", action="store_true", default=True, help="Use Born approximation (default)")
    lensing_method.add_argument("--raytrace",
                                action="store_true",
                                default=False,
                                help="Use multi-plane ray-tracing via dorian")
    lensing_p.add_argument("--min-z",
                           type=float,
                           default=0.01,
                           help="Minimum redshift for nz integration (default: 0.01)")
    lensing_p.add_argument("--max-z",
                           type=float,
                           default=3.0,
                           help="Maximum redshift for nz integration (default: 3.0)")
    lensing_p.add_argument("--n-integrate",
                           type=int,
                           default=32,
                           help="Number of integration points for nz distributions (default: 32)")
    lensing_p.add_argument(
        "--rt-interp",
        choices=["bilinear", "ngp", "nufft"],
        default="bilinear",
        help="Interpolation method for raytrace (default: bilinear)",
    )
    lensing_p.add_argument("--no-parallel-transport",
                           action="store_true",
                           help="Disable parallel transport in raytrace")

    return parser


# ---------------------------------------------------------------------------
# Pre-JAX argument validation
# ---------------------------------------------------------------------------


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    """Validate argument combinations before importing JAX."""
    # --ts-near requires --ts-far and --nb-shells
    ts_near = getattr(args, "ts_near", None)
    ts_far = getattr(args, "ts_far", None)
    nb_shells = getattr(args, "nb_shells", None)

    if ts_near is not None:
        if ts_far is None:
            parser.error("--ts-near requires --ts-far")
        if len(ts_near) != len(ts_far):
            parser.error(
                f"--ts-near and --ts-far must have the same number of values ({len(ts_near)} vs {len(ts_far)})")

    # lensing requires a projection target
    if args.subcommand == "lensing":
        nside = getattr(args, "nside", None)
        flatsky_npix = getattr(args, "flatsky_npix", None)
        use_raytrace = getattr(args, "raytrace", False)
        if nside is None and flatsky_npix is None:
            parser.error("lensing subcommand requires --nside or --flatsky-npix")
        if use_raytrace and nside is None:
            parser.error("--raytrace requires --nside (spherical painting)")

    # --perf and --trace are mutually exclusive (perf wins)
    if getattr(args, "perf", False) and getattr(args, "trace", False):
        print("Warning: --perf and --trace both specified; --perf takes precedence, --trace ignored", file=sys.stderr)
        args.trace = False

    # --interp onion requires --nside
    interp = getattr(args, "interp", "none")
    nside = getattr(args, "nside", None)
    if interp == "onion" and nside is None:
        parser.error("--interp onion requires --nside")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(3, 4, 6, 7, 10, 12))
def run_simulations(
    cosmo,
    initial_conditions,
    solver,
    t0,
    t1,
    ts,
    dt0,
    nb_shells,
    painting,
    interp_kernel,
    lpt_order,
    nz_shear,
    sim_type,
) -> ffi.io.Catalog:
    jax.config.update("jax_enable_x64", False)

    if sim_type == "lpt":
        # lightcone mode — forward ts + nb_shells from CLI
        dx, p = ffi.lpt(cosmo, initial_conditions, ts=ts, nb_shells=nb_shells, order=lpt_order, painting=painting)
        return dx

    # All other modes: LPT to particles snapshot at t0, then run NBody
    dx, p = ffi.lpt(cosmo, initial_conditions, ts=t0, order=lpt_order, painting=ffi.PaintingOptions(target="particles"))

    # Run NBody
    lightcone = ffi.nbody(
        cosmo,
        dx,
        p,
        t0=t0,
        t1=t1,
        dt0=dt0,
        ts=ts,
        nb_shells=nb_shells,
        solver=solver,
    )
    if sim_type == "nbody":
        return lightcone

    # Run lensing
    if sim_type == "born":
        kappa = ffi.born(cosmo, lightcone, nz_shear)
    elif sim_type == "raytrace":
        kappa = ffi.raytrace(cosmo, lightcone, nz_shear)
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")

    return kappa


def main() -> None:
    """CLI entry point registered as ffi-simulate."""
    p = parser()
    args, unknown = p.parse_known_args()
    if unknown:
        print(
            f"Warning: the following arguments are not recognized by the "
            f"'{args.subcommand}' subcommand and will be ignored: {unknown}",
            file=sys.stderr,
        )
    jax.config.update("jax_enable_x64", args.enable_x64)
    _validate_args(args, p)

    # Prepare arguments

    cosmo = _build_cosmo(args)
    cosmo._workspace = {}

    painting, nside, flatsky_npix = _build_painting(args)
    sharding = _build_sharding(args)

    ts = _resolve_ts(args)
    nz_shear = _resolve_nz_shear(args)
    solver = _build_solver(args, painting)
    t1 = getattr(args, "t1", 1.0)
    dt0 = getattr(args, "dt0", 0.05)

    key = jax.random.key(args.seed)

    initial_field = ffi.gaussian_initial_conditions(
        key,
        tuple(args.mesh_size),
        tuple(args.box_size),
        observer_position=tuple(args.observer_position),
        cosmo=cosmo,
        nside=args.nside,
        flatsky_npix=tuple(args.flatsky_npix) if args.flatsky_npix is not None else None,
        field_size=tuple(args.field_size) if args.field_size is not None else None,
        sharding=sharding)

    sim_type = args.subcommand
    lpt_order = args.order
    if args.subcommand == "lensing":
        sim_type = "raytrace" if args.raytrace else "born"

    run_kwargs = {
        "cosmo": cosmo,
        "initial_conditions": initial_field,
        "solver": solver,
        "t0": args.t0,
        "t1": t1,
        "dt0": dt0,
        "ts": ts,
        "nb_shells": args.nb_shells,
        "painting": painting,
        "interp_kernel": solver.interp_kernel,
        "lpt_order": lpt_order,
        "nz_shear": nz_shear,
        "sim_type": sim_type,
    }

    if args.perf:
        try:
            from jax_hpc_profiler import JaxTimer
        except ImportError:
            print("Error: jax-hpc-profiler not found. Please install it to use --perf.", file=sys.stderr)
            sys.exit(1)

        timer = JaxTimer(save_jaxpr=False, static_argnums=(3, 4, 6, 7, 10, 12))
        print("Compiling and running first iteration...")
        # chrono_jit measures compilation + first run
        result = timer.chrono_jit(run_simulations, **run_kwargs)

        print(f"Running {args.iterations} timed iterations...")
        for i in range(args.iterations):
            # chrono_fun measures execution time
            result = timer.chrono_fun(run_simulations, **run_kwargs)

        metadata = {
            'precision': 'float64' if jax.config.jax_enable_x64 else 'float32',
            'x': str(args.mesh_size[0]),
            'y': str(args.mesh_size[1]),
            'z': str(args.mesh_size[2]),
            'px': str(args.pdim[0]),
            'py': str(args.pdim[1]),
            'nodes': str(args.nodes)
        }
        extra_info = {
            'halo_size': str(args.halo_size),
            'painting_target': painting.target,
            'ts': str(args.ts) if args.ts is not None else f"near={args.ts_near}, far={args.ts_far}",
            'nb_shells': str(args.nb_shells),
            'lpt_order': str(args.order),
        }

        report_file = f"perf_{sim_type}.csv"
        timer.report(report_file, function=sim_type, extra_info=extra_info, **metadata)
        print(f"Performance report saved to {report_file}")
    else:
        result = run_simulations(**run_kwargs).block_until_ready()

    print("Simulation completed... saving results.")
    sync_global_devices("Done")
    # --- Save output ---
    _save_result(result, cosmo, args)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
