"""fli-grid: single-process Cartesian grid runner for fli-simulate.

Accepts list-valued parameters for the griddable dimensions and runs all
Cartesian combinations sequentially, reusing the same JAX JIT cache across
runs that share the same static shapes.

Range notation
--------------
Scalar griddable parameters (--Omega-c, --sigma8, --seed, --nb-shells,
--dt0, --nb-steps) accept either explicit values or a compact
``start:stop:step`` range token (stop is inclusive).  Both styles can be
mixed freely in the same argument.

    --Omega-c 0.2:0.4:0.01          # 21 values: 0.20, 0.21, …, 0.40
    --seed 0:9:1                     # 10 seeds: 0, 1, …, 9
    --sigma8 0.7 0.8:0.9:0.05       # explicit 0.7 + range 0.80, 0.85, 0.90

Example
-------
    fli-grid nbody \\
        --mesh-size 64 64 64 128 128 128 \\
        --box-size 200 200 200 \\
        --Omega-c 0.2:0.4:0.1 \\
        --sigma8 0.7:0.9:0.1 \\
        --seed 0:2:1 \\
        --nb-shells 10 \\
        --nb-steps 18 \\
        --nside 16 --output-dir /tmp/grid_out --dry-run
"""

from __future__ import annotations

import copy
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_range_values(raw: list, dtype: type) -> list:
    """Expand a mixed list of plain scalars and ``start:stop:step`` range tokens.

    Each element of *raw* is either:

    * A plain scalar (string or already-converted number) → converted to *dtype*.
    * A ``"start:stop:step"`` string → expanded via ``np.arange``; *stop* is
      inclusive (uses a small epsilon to guard against floating-point rounding).

    Examples
    --------
    >>> _expand_range_values(["0.2:0.4:0.1"], float)
    [0.2, 0.3, 0.4]
    >>> _expand_range_values(["0:2:1"], int)
    [0, 1, 2]
    >>> _expand_range_values([0.8159, "0.9:1.0:0.05"], float)
    [0.8159, 0.9, 0.95, 1.0]
    """
    result: list = []
    for v in raw:
        s = str(v)
        if ":" in s:
            parts = s.split(":")
            if len(parts) != 3:
                raise ValueError(f"Range notation must be 'start:stop:step', got {v!r}")
            start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
            arr = np.arange(start, stop + step * 1e-9, step)
            result.extend(dtype(x) for x in arr)
        else:
            result.append(dtype(s))
    return result


def _parse_groups(values: list, group_size: int) -> list[tuple]:
    """Split a flat list into tuples of ``group_size``.

    e.g. [64, 64, 64, 128, 128, 128] with group_size=3
         → [(64, 64, 64), (128, 128, 128)]
    """
    if len(values) % group_size != 0:
        raise ValueError(f"Expected a multiple of {group_size} values, got {len(values)}: {values}")
    return [tuple(values[i : i + group_size]) for i in range(0, len(values), group_size)]


def _make_stem(
    subcommand: str, mesh, box, omega_c, sigma8, seed, nb_shells, dt0_or_steps, is_steps: bool, density_widths=None
) -> str:
    """Build a descriptive filename stem for a single grid combination."""
    mesh_str = "x".join(str(m) for m in mesh)
    box_str = "x".join(str(int(b)) if b == int(b) else str(b) for b in box)
    step_tag = f"Nst{dt0_or_steps}" if is_steps else f"dt{dt0_or_steps}"
    dw_tag = f"_dw{density_widths}" if density_widths is not None else ""
    return f"{subcommand}_M{mesh_str}_B{box_str}_Oc{omega_c}_S8{sigma8}_s{seed}_Ns{nb_shells}_{step_tag}{dw_tag}"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parser() -> ArgumentParser:
    """Build the fli-grid argument parser."""

    p = ArgumentParser(
        prog="fli-grid",
        description="Run a Cartesian grid of fli-simulate combinations.",
    )
    subparsers = p.add_subparsers(dest="subcommand", required=True)

    # ------------------------------------------------------------------
    # Shared grid parent (used by all subcommands)
    # ------------------------------------------------------------------
    grid_parent = ArgumentParser(add_help=False)

    # --- Griddable: mesh / box (flat list, parsed in groups of 3) ---
    grid_parent.add_argument(
        "--mesh-size",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        metavar="N",
        help="Mesh resolution(s) — groups of 3: '64 64 64 128 128 128' → two configs",
    )
    grid_parent.add_argument(
        "--box-size",
        type=float,
        nargs="+",
        default=[200.0, 200.0, 200.0],
        metavar="L",
        help="Box side length(s) in Mpc/h — groups of 3",
    )

    # --- Griddable: cosmology / seed ---
    grid_parent.add_argument(
        "--Omega-c",
        type=str,
        nargs="+",
        default=[0.2589],
        metavar="OC",
        help="Cold dark matter density values; supports start:stop:step range notation",
    )
    grid_parent.add_argument(
        "--sigma8",
        type=str,
        nargs="+",
        default=[0.8159],
        metavar="S8",
        help="sigma8 values; supports start:stop:step range notation",
    )
    grid_parent.add_argument(
        "--seed",
        type=str,
        nargs="+",
        default=[0],
        metavar="S",
        help="Random seed(s); supports start:stop:step range notation (e.g. 0:9:1)",
    )

    # --- Griddable: shells ---
    grid_parent.add_argument(
        "--nb-shells",
        type=str,
        nargs="+",
        default=[10],
        metavar="N",
        help="Number of lightcone shells values; supports start:stop:step range notation",
    )

    # --- Griddable: time-stepping (mutually exclusive) ---
    dt_group = grid_parent.add_mutually_exclusive_group()
    dt_group.add_argument(
        "--dt0",
        type=str,
        nargs="+",
        default=None,
        metavar="DT",
        help="Integration time step(s); supports start:stop:step range notation",
    )
    dt_group.add_argument(
        "--nb-steps",
        type=str,
        nargs="+",
        default=None,
        dest="nb_steps",
        metavar="N",
        help="Number of integration steps (mutually exclusive with --dt0); supports start:stop:step range notation",
    )

    # --- Fixed: other cosmology ---
    cosmo_group = grid_parent.add_argument_group("fixed cosmology")
    cosmo_group.add_argument("--Omega-b", type=float, default=0.0486)
    cosmo_group.add_argument("--h", type=float, default=0.6774)
    cosmo_group.add_argument("--n-s", type=float, default=0.9667, dest="n_s")
    cosmo_group.add_argument("--Omega-k", type=float, default=0.0)
    cosmo_group.add_argument("--w0", type=float, default=-1.0)
    cosmo_group.add_argument("--wa", type=float, default=0.0)
    cosmo_group.add_argument("--Omega-nu", type=float, default=0.0)

    # --- Fixed: integration bounds ---
    grid_parent.add_argument("--t0", type=float, default=0.1)
    grid_parent.add_argument("--t1", type=float, default=1.0)

    # --- Fixed: LPT / solver ---
    grid_parent.add_argument("--order", type=int, default=2, choices=[1, 2])
    grid_parent.add_argument("--interp", choices=["none", "onion", "telephoto"], default="none")
    grid_parent.add_argument("--drift-on-lightcone", action="store_true")

    # --- Fixed: lightcone geometry ---
    grid_parent.add_argument("--equal-vol", action="store_true", default=False)
    grid_parent.add_argument("--min-width", type=float, default=50.0, dest="min_width")
    grid_parent.add_argument("--density-widths", type=float, nargs="+", default=None, metavar="W")
    ts_group = grid_parent.add_mutually_exclusive_group()
    ts_group.add_argument("--ts", type=float, nargs="+", default=None, metavar="A")
    ts_group.add_argument("--ts-near", type=float, nargs="+", default=None, metavar="A_NEAR")
    grid_parent.add_argument("--ts-far", type=float, nargs="+", default=None, metavar="A_FAR")

    # --- Fixed: painting target ---
    paint_group = grid_parent.add_mutually_exclusive_group()
    paint_group.add_argument("--nside", type=int, default=None)
    paint_group.add_argument("--flatsky-npix", type=int, nargs=2, default=None, metavar=("H", "W"))
    paint_group.add_argument("--field-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    paint_group.add_argument("--density", action="store_true", default=False)

    # --- Fixed: distributed / sharding ---
    grid_parent.add_argument("--pdim", type=int, nargs=2, default=[1, 1], metavar=("PX", "PY"))
    grid_parent.add_argument("--nodes", type=int, default=1)
    grid_parent.add_argument("--halo-size", type=int, nargs=2, default=[0, 0], metavar=("H0", "H1"))
    grid_parent.add_argument(
        "--observer-position", type=float, nargs=3, default=[0.5, 0.5, 0.5], metavar=("OX", "OY", "OZ")
    )

    # --- Output / control ---
    grid_parent.add_argument("--output-dir", default=".", metavar="DIR", help="Output directory (default: .)")
    grid_parent.add_argument("--dry-run", action="store_true", help="Print combinations without running")
    grid_parent.add_argument("--enable-x64", action="store_true")

    # --- lpt subcommand ---
    subparsers.add_parser("lpt", parents=[grid_parent], help="Grid over LPT runs")

    # --- nbody subcommand ---
    subparsers.add_parser("nbody", parents=[grid_parent], help="Grid over NBody runs")

    # --- lensing subcommand ---
    lensing_p = subparsers.add_parser("lensing", parents=[grid_parent], help="Grid over full lensing pipeline runs")
    lensing_p.add_argument("--nz-shear", nargs="+", required=True, metavar="Z")
    lensing_p.add_argument("--lensing", choices=["born", "raytrace", "both"], default="born")
    lensing_p.add_argument("--min-z", type=float, default=0.01)
    lensing_p.add_argument("--max-z", type=float, default=1.5)
    lensing_p.add_argument("--n-integrate", type=int, default=32)
    lensing_p.add_argument("--rt-interp", choices=["bilinear", "ngp", "nufft"], default="bilinear")
    lensing_p.add_argument("--no-parallel-transport", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered as fli-grid."""
    import jax
    import jax.numpy as jnp

    from jax_fli.scripts.fli_simulate import (
        _build_cosmo,
        _build_painting,
        _build_sharding,
        _build_solver,
        _resolve_dt0,
        _resolve_nz_shear,
        _resolve_ts,
        _save_result,
        run_simulations,
    )

    p = parser()
    args = p.parse_args()
    jax.config.update("jax_enable_x64", args.enable_x64)

    # --- Expand range notation for griddable scalar parameters ---
    args.Omega_c = _expand_range_values(args.Omega_c, float)
    args.sigma8 = _expand_range_values(args.sigma8, float)
    args.seed = _expand_range_values(args.seed, int)
    args.nb_shells = _expand_range_values(args.nb_shells, int)
    if args.dt0 is not None:
        args.dt0 = _expand_range_values(args.dt0, float)
    if args.nb_steps is not None:
        args.nb_steps = _expand_range_values(args.nb_steps, int)

    # --- Expand griddable parameters ---
    mesh_configs = _parse_groups(args.mesh_size, 3)
    box_configs = _parse_groups(args.box_size, 3)

    # Resolve dt0 values: either a list from --dt0, from --nb-steps, or the default
    if args.nb_steps is not None:
        # Store as negative sentinel so we can distinguish nb_steps vs dt0 later
        dt_values = [("nb_steps", n) for n in args.nb_steps]
    elif args.dt0 is not None:
        dt_values = [("dt0", d) for d in args.dt0]
    else:
        # Default: dt0=0.05
        dt_values = [("dt0", None)]  # None triggers _resolve_dt0 default

    # Each --density-widths value is a scalar broadcast to all shells; None = use defaults
    density_width_values = args.density_widths if args.density_widths is not None else [None]

    # Compute total from dimension sizes, then iterate lazily
    total = (
        len(mesh_configs)
        * len(box_configs)
        * len(args.Omega_c)
        * len(args.sigma8)
        * len(args.seed)
        * len(args.nb_shells)
        * len(dt_values)
        * len(density_width_values)
    )
    grid = product(
        mesh_configs,
        box_configs,
        args.Omega_c,
        args.sigma8,
        args.seed,
        args.nb_shells,
        dt_values,
        density_width_values,
    )
    print(f"Grid: {total} combination(s) — subcommand={args.subcommand}")
    if args.dry_run:
        print("Dry run — combinations:")

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve nz_shear once (shared across all combos; only valid for lensing subcommand)
    nz_shear = _resolve_nz_shear(args) if args.subcommand == "lensing" else None

    # Sharding from fixed args (shared across all combos)
    sharding = _build_sharding(args)

    import jax_fli as jfli

    for idx, (mesh, box, omega_c, sigma8, seed, nb_shells, (dt_kind, dt_val), density_width_val) in enumerate(grid):
        # Build combo Namespace by shallow-copying fixed args and overriding grid dims
        combo = copy.copy(args)
        combo.mesh_size = list(mesh)
        combo.box_size = list(box)
        combo.Omega_c = omega_c
        combo.sigma8 = sigma8
        combo.seed = seed
        combo.nb_shells = nb_shells

        # Resolve dt0 for this combo
        if dt_kind == "nb_steps":
            combo.dt0 = None
            combo.nb_steps = dt_val
            dt0 = _resolve_dt0(combo, combo.t1)
            is_steps = True
            step_label = dt_val
        else:
            combo.dt0 = dt_val  # None triggers default 0.05 inside _resolve_dt0
            combo.nb_steps = None
            dt0 = _resolve_dt0(combo, combo.t1)
            is_steps = False
            step_label = dt0

        stem = _make_stem(
            args.subcommand, mesh, box, omega_c, sigma8, seed, nb_shells, step_label, is_steps, density_width_val
        )

        print(f"[{idx + 1}/{total}] {stem}")
        if args.dry_run:
            continue

        # Build objects for this combo
        cosmo = _build_cosmo(combo)
        painting, nside, flatsky_npix = _build_painting(combo)
        ts = _resolve_ts(combo)
        solver = _build_solver(combo, painting)

        halo_size = (mesh[0] // 8, mesh[1] // 8)

        key = jax.random.key(seed)
        initial_field = jfli.gaussian_initial_conditions(
            key,
            tuple(mesh),
            tuple(box),
            observer_position=tuple(combo.observer_position),
            cosmo=cosmo,
            nside=combo.nside,
            flatsky_npix=tuple(combo.flatsky_npix) if combo.flatsky_npix is not None else None,
            field_size=tuple(combo.field_size) if combo.field_size is not None else None,
            sharding=sharding,
            halo_size=halo_size,
        )

        sim_type = args.subcommand
        lpt_order = combo.order
        if args.subcommand == "lensing":
            sim_type = combo.lensing

        run_kwargs = {
            "cosmo": cosmo,
            "initial_conditions": initial_field,
            "solver": solver,
            "t0": combo.t0,
            "t1": combo.t1,
            "dt0": dt0,
            "ts": ts,
            "nb_shells": nb_shells,
            "painting": painting,
            "interp_kernel": solver.interp_kernel,
            "lpt_order": lpt_order,
            "nz_shear": nz_shear,
            "sim_type": sim_type,
            "equal_vol": combo.equal_vol,
            "min_width": combo.min_width,
            "density_widths": jnp.array(density_width_val) if density_width_val is not None else None,
        }

        result = jax.block_until_ready(run_simulations(**run_kwargs))

        out_path = output_dir / f"{stem}.parquet"
        if sim_type == "both":
            result_rt, result_born = result
            base = str(out_path.with_suffix(""))
            _save_result(result_rt, cosmo, combo, output=base + "_raytraced.parquet")
            _save_result(result_born, cosmo, combo, output=base + "_born.parquet")
            del result_rt, result_born
        else:
            _save_result(result, cosmo, combo, output=str(out_path))
            del result

        del cosmo, initial_field, solver, painting, ts

    if not args.dry_run:
        print(f"\nAll {total} combination(s) completed.")
    else:
        print(f"\n{total} combination(s) listed (dry run — nothing executed).")


if __name__ == "__main__":
    main()
