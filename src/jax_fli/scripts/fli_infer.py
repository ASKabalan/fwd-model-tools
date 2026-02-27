"""fli-infer: run full-field MCMC inference conditioned on observed kappa maps."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from argparse import Namespace

import jax
import jax_cosmo as jc
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpyro.handlers import condition

import jax_fli as jfli
from jax_fli.fields import FlatKappaField, SphericalKappaField

# ---------------------------------------------------------------------------
# Sharding setup
# ---------------------------------------------------------------------------


def _build_sharding(args: Namespace):
    """Return sharding or None for single-device runs.

    Warns if the product of pdim dimensions does not match the available device count.
    """
    print(f"jax devices: {jax.devices()}")
    pdim = tuple(args.pdim)

    n_devices = jax.device_count()
    if math.prod(pdim) != n_devices:
        warnings.warn(
            f"--pdim {pdim} implies {math.prod(pdim)} devices but jax.device_count() == {n_devices}. "
            "Results may be incorrect on a misconfigured device mesh.",
            stacklevel=2,
        )

    if pdim == (1, 1):
        return None

    mesh = jax.make_mesh(pdim, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    return sharding


# ---------------------------------------------------------------------------
# Observable loading
# ---------------------------------------------------------------------------


def _load_observable(path: str):
    """Load a kappa Catalog from parquet and extract per-bin arrays + metadata.

    Parameters
    ----------
    path : str
        Path to a parquet Catalog saved by ``sample2catalog`` (or equivalent).
        Must contain a single ``SphericalKappaField`` or ``FlatKappaField`` entry
        whose leading array dimension encodes the tomographic bins.

    Returns
    -------
    kappa_arrays : list[jnp.ndarray]
        One array per tomographic bin.  Shape (npix,) for spherical or (ny, nx) for flat.
    obs_cosmo : jc.Cosmology
        The cosmology stored alongside the observable field.
    obs_box_size : tuple[float, float, float]
        Box size (Mpc/h) recorded in the observable catalog.
    geometry : str
        ``"spherical"`` or ``"flat"``.
    nside : int | None
        HEALPix NSIDE for spherical geometry, else None.
    flatsky_npix : tuple[int, int] | None
        ``(ny, nx)`` pixel resolution for flat geometry, else None.
    field_size : tuple[float, float] | None
        Physical field size in degrees ``(size_y, size_x)`` for flat, else None.
    n_kappas : int
        Number of tomographic kappa bins.
    """
    catalog = jfli.io.Catalog.from_parquet(path)
    obs_field = catalog.field[0]
    obs_cosmo = catalog.cosmology[0]

    if isinstance(obs_field, SphericalKappaField):
        geometry = "spherical"
        nside = obs_field.nside
        flatsky_npix = None
        field_size = None
    elif isinstance(obs_field, FlatKappaField):
        geometry = "flat"
        nside = None
        flatsky_npix = obs_field.flatsky_npix
        field_size = obs_field.field_size
    else:
        raise ValueError(
            f"Observable must be a SphericalKappaField or FlatKappaField, got {type(obs_field).__name__}. "
            "Generate observables with ffi-samples or ffi-simulate lensing."
        )

    # The catalog stores all tomographic bins stacked along axis 0: (n_kappas, npix) or (n_kappas, ny, nx)
    n_kappas = obs_field.array.shape[0]
    kappa_arrays = [obs_field.array[i] for i in range(n_kappas)]

    return kappa_arrays, obs_cosmo, obs_field.box_size, geometry, nside, flatsky_npix, field_size, n_kappas


# ---------------------------------------------------------------------------
# Initial condition loading
# ---------------------------------------------------------------------------


def _load_initial_condition(path: str):
    """Load an IC DensityField from a parquet Catalog.

    Parameters
    ----------
    path : str
        Path to a parquet Catalog containing a ``DensityField`` (IC).

    Returns
    -------
    ic_field : DensityField
        The initial condition field (first catalog entry).
    """
    catalog = jfli.io.Catalog.from_parquet(path)
    return catalog.field[0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ffi-infer."""
    p = argparse.ArgumentParser(
        prog="ffi-infer",
        description="Run full-field MCMC inference conditioned on observed kappa maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required positional-style required args ---
    p.add_argument(
        "--observable",
        type=str,
        required=True,
        metavar="PATH",
        help="Parquet Catalog with batched kappa field (n_bins, npix) and cosmology.",
    )
    p.add_argument(
        "--path",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory for MCMC checkpoints and parquet catalogs.",
    )
    p.add_argument("--nb-shells", type=int, required=True, metavar="INT", help="Number of lightcone shells.")
    p.add_argument(
        "--mesh-size", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"), help="Inference mesh resolution."
    )
    p.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        required=True,
        metavar=("LX", "LY", "LZ"),
        help="Box side lengths in Mpc/h. Warns if different from the observable's stored box_size.",
    )

    # --- Optional: IC and sampling targets ---
    p.add_argument(
        "--initial-condition",
        type=str,
        default=None,
        metavar="PATH",
        help="Parquet Catalog with IC DensityField for initialization or fixing IC.",
    )
    p.add_argument(
        "--sample",
        nargs="+",
        default=["cosmo", "ic"],
        choices=["cosmo", "ic"],
        metavar="WHAT",
        help="Space-separated subset of {cosmo, ic} to sample (default: cosmo ic).",
    )

    # --- Device mesh / distributed ---
    p.add_argument("--pdim", type=int, nargs=2, default=[1, 1], metavar=("PX", "PY"), help="Device mesh dimensions.")
    p.add_argument(
        "--halo-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H0", "H1"),
        help="Halo exchange depth; required when --pdim != 1 1 and jax.device_count() > 1.",
    )

    # --- Observer / physics ---
    p.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        metavar=("OX", "OY", "OZ"),
        help="Observer position in box coordinates.",
    )
    p.add_argument("--sigma-e", type=float, default=0.26, help="Shape noise dispersion.")
    p.add_argument("--density-plane-smoothing", type=float, default=0.0, help="Density plane smoothing scale.")

    # --- Time-stepping ---
    p.add_argument("--t0", type=float, default=0.01, help="LPT start scale factor.")
    p.add_argument("--t1", type=float, default=1.0, help="NBody end scale factor.")
    p.add_argument("--dt0", type=float, default=0.01, help="Integration time step.")
    p.add_argument("--lpt-order", type=int, choices=[1, 2], default=2, help="LPT order.")

    # --- Gradient strategy ---
    p.add_argument(
        "--adjoint", choices=["checkpointed", "recursive"], default="checkpointed", help="Gradient strategy for NUTS."
    )
    p.add_argument(
        "--checkpoints", type=int, default=10, help="Number of gradient checkpoints (used when --adjoint checkpointed)."
    )

    # --- MCMC settings ---
    p.add_argument("--num-warmup", type=int, default=500, help="MCMC warmup iterations.")
    p.add_argument("--num-samples", type=int, default=1000, help="Samples per batch.")
    p.add_argument("--batch-count", type=int, default=5, help="Number of sequential batches.")
    p.add_argument("--sampler", choices=["NUTS", "HMC", "MCLMC"], default="NUTS", help="MCMC sampler.")
    p.add_argument("--backend", choices=["numpyro", "blackjax"], default="numpyro", help="Sampling backend.")

    # --- Misc ---
    p.add_argument("--seed", type=int, default=0, help="JAX PRNGKey seed.")
    p.add_argument("--no-progress-bar", action="store_true", help="Suppress tqdm progress bars.")
    p.add_argument("--enable-x64", action="store_true", help="Enable JAX 64-bit precision.")

    return p


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _validate_args(args: Namespace, p: argparse.ArgumentParser) -> None:
    """Validate argument combinations before running JAX.

    Raises SystemExit (via parser.error) for hard errors; issues warnings for soft mismatches.
    """
    # 1. --sample must contain at least one of cosmo or ic
    if not args.sample:
        p.error("--sample must contain at least one of 'cosmo' or 'ic'.")

    sample_set = set(args.sample)
    if not sample_set & {"cosmo", "ic"}:
        p.error(f"--sample must contain at least one of 'cosmo' or 'ic', got: {args.sample}")

    # 2. If IC is NOT being sampled, --initial-condition is required to fix the IC value
    if "ic" not in sample_set and args.initial_condition is None:
        p.error("--initial-condition is required when 'ic' is not in --sample (IC must be fixed).")

    # 3. MCLMC sampler requires --backend blackjax
    if args.sampler == "MCLMC" and args.backend != "blackjax":
        p.error("--sampler MCLMC requires --backend blackjax.")

    # 4. If --pdim != 1 1 AND multiple devices: --halo-size is required
    pdim = tuple(args.pdim)
    if pdim != (1, 1) and jax.device_count() > 1 and args.halo_size is None:
        p.error("--halo-size is required when --pdim != 1 1 and jax.device_count() > 1.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as ffi-infer."""
    p = parser()
    args = p.parse_args()

    jax.config.update("jax_enable_x64", args.enable_x64)

    _validate_args(args, p)

    # 1. Load observable → kappa arrays and geometry metadata
    kappa_arrays, obs_cosmo, obs_box_size, geometry, nside, flatsky_npix, field_size, n_kappas = _load_observable(
        args.observable
    )

    # 2. Warn if CLI box_size differs from observable's stored box_size
    cli_box = tuple(args.box_size)
    if cli_box != tuple(obs_box_size):
        warnings.warn(
            f"--box-size {cli_box} differs from the observable's stored box_size {tuple(obs_box_size)}. "
            "Make sure this is intentional.",
            stacklevel=2,
        )

    # 3. Optionally load initial conditions
    ic_field = None
    if args.initial_condition is not None:
        ic_field = _load_initial_condition(args.initial_condition)

    sample_set = set(args.sample)

    # 4. Build condition_data: fix observed kappa maps and optionally cosmo / IC
    condition_data = {f"kappa_{i}": kappa_arrays[i] for i in range(n_kappas)}

    if "cosmo" not in sample_set:
        condition_data["Omega_c"] = float(obs_cosmo.Omega_c)
        condition_data["sigma8"] = float(obs_cosmo.sigma8)

    if "ic" not in sample_set:
        assert ic_field is not None  # guaranteed by _validate_args
        condition_data["initial_conditions"] = ic_field.array

    # 5. Build init_params: warm-start IC chain from a provided field
    init_params = None
    if "ic" in sample_set and ic_field is not None:
        init_params = {"initial_conditions": ic_field.array}

    # 6. Build sharding (warns if pdim product != device count)
    sharding = _build_sharding(args)

    # 7. Assemble Configurations, probabilistic model, and conditioned model
    nz_shear = jfli.io.get_stage3_nz_shear()

    if len(nz_shear) != n_kappas:
        print(
            f"Warning: observable has {n_kappas} kappa maps but Stage-3 nz_shear has {len(nz_shear)} bins. "
            "Inference may fail if the numbers don't match.",
            file=sys.stderr,
        )

    priors = {
        "Omega_c": jfli.infer.PreconditionnedUniform(0.1, 0.5),
        "sigma8": jfli.infer.PreconditionnedUniform(0.6, 1.0),
    }

    config = jfli.ppl.Configurations(
        mesh_size=tuple(args.mesh_size),
        box_size=tuple(args.box_size),
        nside=nside,
        flatsky_npix=flatsky_npix,
        field_size=field_size,
        geometry=geometry,
        observer_position=tuple(args.observer_position),
        fiducial_cosmology=jc.Planck18,
        nz_shear=nz_shear,
        priors=priors,
        sigma_e=args.sigma_e,
        density_plane_smoothing=args.density_plane_smoothing,
        halo_size=tuple(args.halo_size) if args.halo_size is not None else (0, 0),
        t0=args.t0,
        dt0=args.dt0,
        t1=args.t1,
        lpt_order=args.lpt_order,
        number_of_shells=args.nb_shells,
        lensing="born",
        adjoint=args.adjoint,
        checkpoints=args.checkpoints,
        sharding=sharding,
    )

    prob_model = jfli.ppl.full_field_probmodel(config)
    conditioned_model = condition(prob_model, data=condition_data)

    # 8. Run batched MCMC
    jfli.infer.batched_sampling(
        conditioned_model,
        path=args.path,
        rng_key=jax.random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
        init_params=init_params,
        progress_bar=not args.no_progress_bar,
        save_callback=jfli.ppl.sample2catalog(config),
    )


if __name__ == "__main__":
    main()
