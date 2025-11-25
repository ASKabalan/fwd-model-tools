#!/usr/bin/env python3
"""
Generate particle mesh densities from N-body simulation.

Saves outputs as Orbax checkpoint containing initial conditions,
lightcones, redshift ranges, and cosmology parameters.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffrax import ODETerm, RecursiveCheckpointAdjoint, SaveAt, diffeqsolve
from jaxpm.lensing import density_plane_fn, spherical_density_fn
from jaxpm.ode import symplectic_ode

from fwd_model_tools.fields import DensityField, FieldStatus
from fwd_model_tools.initial import interpolate_initial_conditions
from fwd_model_tools.pm import lpt
from fwd_model_tools.sampling.persistency import save_sharded
from fwd_model_tools.solvers.reversible_efficient_fastpm import ReversibleEfficientFastPM
from fwd_model_tools.utils import compute_box_size_from_redshift, compute_snapshot_scale_factors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PM densities from N-body simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required simulation parameters
    parser.add_argument(
        "--mesh-size",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Mesh size (nx, ny, nz)",
    )

    # Box size (one of box-size or max-redshift required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        help="Box size in Mpc/h (Lx, Ly, Lz)",
    )
    group.add_argument(
        "--max-redshift",
        type=float,
        metavar="Z",
        help="Maximum redshift (box_size computed automatically)",
    )

    # Cosmology parameters
    parser.add_argument(
        "--omega-m",
        type=float,
        default=0.3097,
        help="Total matter density Omega_m",
    )
    parser.add_argument("--omega-b", type=float, default=0.049, help="Baryon density Omega_b")
    parser.add_argument(
        "--h",
        type=float,
        default=0.6766,
        help="Hubble parameter h = H0 / (100 km/s/Mpc)",
    )
    parser.add_argument(
        "--sigma8",
        type=float,
        default=0.8102,
        help="Amplitude of matter fluctuations sigma_8",
    )
    parser.add_argument("--n-s", type=float, default=0.9665, help="Scalar spectral index n_s")
    parser.add_argument(
        "--w0",
        type=float,
        default=-1.0,
        help="Dark energy equation of state w0",
    )
    parser.add_argument(
        "--wa",
        type=float,
        default=0.0,
        help="Dark energy equation of state evolution wa",
    )

    # Time evolution (choose ONE method)
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--redshifts",
        type=float,
        nargs="+",
        metavar="Z",
        help="Output redshifts for snapshots",
    )
    time_group.add_argument(
        "--scale-factors",
        type=float,
        nargs="+",
        metavar="A",
        help="Output scale factors for snapshots",
    )
    time_group.add_argument(
        "--nb-shells",
        type=int,
        metavar="N",
        help="Number of shells (uses compute_snapshot_scale_factors)",
    )

    parser.add_argument("--t0", type=float, default=0.1, help="Initial scale factor")
    parser.add_argument("--t1", type=float, default=1.0, help="Final scale factor")
    parser.add_argument("--dt0", type=float, default=0.05, help="Initial time step")

    # Geometry (can specify multiple)
    parser.add_argument(
        "--geometry",
        type=str,
        nargs="+",
        required=True,
        choices=["spherical", "flat"],
        help="Output geometries (can specify multiple)",
    )

    # Observer and projections
    parser.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        metavar=("X", "Y", "Z"),
        help="Observer position as fraction of box (x, y, z)",
    )
    parser.add_argument(
        "--flatsky-npix",
        type=int,
        nargs=2,
        default=None,
        metavar=("NX", "NY"),
        help="Flat-sky resolution (nx, ny) - required if 'flat' in geometry",
    )
    parser.add_argument(
        "--field-size",
        type=float,
        default=None,
        metavar="DEG",
        help="Field of view in degrees - required if 'flat' in geometry",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=None,
        metavar="NSIDE",
        help="HEALPix nside - required if 'spherical' in geometry",
    )

    # Initial conditions
    parser.add_argument(
        "--lpt-order",
        type=int,
        default=1,
        choices=[1, 2],
        help="LPT order for initial conditions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initial conditions")

    # Output
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for Orbax checkpoint (e.g., output/simulation)",
    )

    return parser.parse_args()


def compute_snapshot_times(args, cosmo, temp_field):
    """
    Determine snapshot times from arguments.

    Uses compute_snapshot_scale_factors from utils.py for nb_shells case.
    """
    if args.redshifts is not None:
        z = jnp.array(args.redshifts)
        a = 1.0 / (1.0 + z)
        return jnp.sort(a)[::-1]  # Descending for backward integration

    elif args.scale_factors is not None:
        return jnp.sort(jnp.array(args.scale_factors))[::-1]

    elif args.nb_shells is not None:
        # Use the function from utils.py
        a_center = compute_snapshot_scale_factors(cosmo, temp_field, nb_shells=args.nb_shells)
        return a_center[::-1]  # Reverse for backward integration

    else:
        raise ValueError("Must specify one of: redshifts, scale_factors, nb_shells")


def compute_z_near_z_far(scale_factors, cosmo):
    """
    Compute z_near and z_far for each shell from scale factors.

    Returns
    -------
    z_near : jax.Array
        Near redshift for each shell (higher z, earlier time)
    z_far : jax.Array
        Far redshift for each shell (lower z, later time)
    """
    # Sort in ascending order (early to late)
    a_sorted = jnp.sort(scale_factors)

    # Compute bin edges (midpoints between adjacent scale factors)
    if len(a_sorted) > 1:
        a_edges = jnp.concatenate([
            jnp.array([0.5 * (a_sorted[0] + a_sorted[1]) - (a_sorted[1] - a_sorted[0]) / 2]),
            0.5 * (a_sorted[1:] + a_sorted[:-1]),
            jnp.array([0.5 * (a_sorted[-1] + a_sorted[-2]) + (a_sorted[-1] - a_sorted[-2]) / 2]),
        ])
    else:
        # Single shell - use reasonable bounds
        da = 0.1
        a_edges = jnp.array([a_sorted[0] - da, a_sorted[0] + da])

    # Convert to redshifts
    z_edges = 1.0 / a_edges - 1.0
    z_near = z_edges[:-1]  # Near edge (higher z, lower a)
    z_far = z_edges[1:]  # Far edge (lower z, higher a)

    # Match the order of scale_factors input
    # If scale_factors were descending, reverse z_near and z_far
    if scale_factors[0] > scale_factors[-1]:
        z_near = z_near[::-1]
        z_far = z_far[::-1]

    return z_near, z_far


def create_multi_geometry_snapshot_fn(
    geometry,
    mesh_size,
    box_size,
    observer_position_mpc,
    density_plane_width,
    nside=None,
    flatsky_npix=None,
    sharding=None,
):
    """
    Create snapshot function that returns tuple of results based on geometry.

    Parameters
    ----------
    geometry : list[str]
        List of geometries: "spherical" and/or "flat"

    Returns
    -------
    callable
        Snapshot function compatible with SaveAt
    """
    # Normalize to list
    if isinstance(geometry, str):
        geometry = [geometry]

    def snapshot_fn(t, y, args):
        results = []

        if "spherical" in geometry:
            spherical_fn = spherical_density_fn(
                mesh_size,
                box_size,
                nside,
                observer_position_mpc,
                density_plane_width,
                sharding=sharding,
            )
            spherical_map = spherical_fn(t, y[1], args)
            results.append(spherical_map)

        if "flat" in geometry:
            flat_fn = density_plane_fn(
                mesh_size,
                box_size,
                density_plane_width,
                flatsky_npix,
                sharding=sharding,
            )
            flat_map = flat_fn(t, y[1], args)
            results.append(flat_map)

        # Return single value if only one geometry, else tuple
        if len(results) == 1:
            return results[0]
        return tuple(results)

    return snapshot_fn


def main():
    args = parse_args()

    print("=" * 60)
    print("PM Density Generation")
    print("=" * 60)

    # Validate geometry-specific requirements
    if "flat" in args.geometry:
        if args.flatsky_npix is None or args.field_size is None:
            raise ValueError("--flatsky-npix and --field-size required for flat geometry")

    if "spherical" in args.geometry:
        if args.nside is None:
            raise ValueError("--nside required for spherical geometry")

    # Setup
    mesh_size = tuple(args.mesh_size)
    observer_position = tuple(args.observer_position)

    print(f"\nMesh size: {mesh_size}")
    print(f"Observer position: {observer_position}")

    # Create cosmology
    omega_c = args.omega_m - args.omega_b
    cosmo = jc.Cosmology(
        Omega_c=omega_c,
        Omega_b=args.omega_b,
        h=args.h,
        sigma8=args.sigma8,
        n_s=args.n_s,
        w0=args.w0,
        wa=args.wa,
        Omega_k=0.0,  # Flat universe
    )

    print(f"\nCosmology:")
    print(f"  Omega_m: {cosmo.Omega_m:.4f}")
    print(f"  Omega_c: {cosmo.Omega_c:.4f}")
    print(f"  Omega_b: {cosmo.Omega_b:.4f}")
    print(f"  h: {cosmo.h:.4f}")
    print(f"  sigma8: {cosmo.sigma8:.4f}")
    print(f"  n_s: {cosmo.n_s:.4f}")

    # Compute box_size if needed
    if args.box_size is not None:
        box_size = tuple(args.box_size)
    else:
        print(f"\nComputing box size from max_redshift={args.max_redshift}...")
        box_size = compute_box_size_from_redshift(cosmo, args.max_redshift, observer_position)
        box_size = tuple([float(b) for b in box_size])

    print(f"Box size: {box_size} Mpc/h")

    # Create temporary field for compute_snapshot_scale_factors
    temp_field = DensityField(
        array=jnp.zeros(mesh_size),
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position=observer_position,
        nside=args.nside,
        flatsky_npix=tuple(args.flatsky_npix) if args.flatsky_npix else None,
        field_size=args.field_size,
        status=FieldStatus.RAW,
    )

    # Compute snapshot times
    print(f"\nComputing snapshot times...")
    snapshot_times = compute_snapshot_times(args, cosmo, temp_field)
    print(f"Number of snapshots: {len(snapshot_times)}")
    print(f"Scale factor range: [{snapshot_times.min():.4f}, {snapshot_times.max():.4f}]")

    # Compute z_near and z_far
    z_near, z_far = compute_z_near_z_far(snapshot_times, cosmo)
    print(f"Redshift range: [{z_far.min():.4f}, {z_near.max():.4f}]")

    # Generate initial conditions
    print(f"\nGenerating initial conditions (seed={args.seed})...")
    key = jax.random.PRNGKey(args.seed)

    # Create power spectrum function
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    cosmo._workspace = {}

    def pk_fn(x):
        return jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    lin_field = interpolate_initial_conditions(
        jax.random.normal(key, mesh_size),
        mesh_size,
        box_size,
        pk_fn=pk_fn,
    )

    # Wrap in DensityField
    lin_density = DensityField(
        array=lin_field,
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position=observer_position,
        sharding=None,
        nside=args.nside,
        flatsky_npix=tuple(args.flatsky_npix) if args.flatsky_npix else None,
        field_size=args.field_size,
        halo_size=0,
        status=FieldStatus.INITIAL_FIELD,
        scale_factors=args.t0,
    )

    print(f"Initial field shape: {lin_field.shape}")

    # Run LPT
    print(f"\nRunning LPT (order={args.lpt_order})...")
    dx_field, p_field = lpt(cosmo, lin_density, a=args.t0, order=args.lpt_order)
    cosmo._workspace = {}

    print(f"Displacement field shape: {dx_field.array.shape}")
    print(f"Momentum field shape: {p_field.array.shape}")

    # Setup ODE integration
    print(f"\nSetting up N-body integration...")
    drift, kick = symplectic_ode(mesh_size, paint_absolute_pos=False, halo_size=0, sharding=None)
    ode_fn = (ODETerm(kick), ODETerm(drift))

    # Compute observer position in physical coordinates
    observer_position_mpc = np.array([
        observer_position[0] * box_size[0],
        observer_position[1] * box_size[1],
        observer_position[2] * box_size[2],
    ])

    # Compute density_plane_width
    factors = np.clip(observer_position, 0.0, 1.0)
    factors = 1.0 + 2.0 * np.minimum(factors, 1.0 - factors)
    max_radius = box_size[2] / factors[2]
    density_plane_width = max_radius / len(snapshot_times)

    print(f"Max radius: {max_radius:.2f} Mpc/h")
    print(f"Density plane width: {density_plane_width:.2f} Mpc/h")

    # Create multi-geometry snapshot function
    snapshot_fn = create_multi_geometry_snapshot_fn(
        geometry=args.geometry,
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position_mpc=observer_position_mpc,
        density_plane_width=density_plane_width,
        nside=args.nside,
        flatsky_npix=tuple(args.flatsky_npix) if args.flatsky_npix else None,
        sharding=None,
    )

    saveat = SaveAt(ts=snapshot_times, fn=snapshot_fn)

    solver = ReversibleEfficientFastPM()

    y0 = (p_field.array, dx_field.array)

    print(f"\nRunning simulation (t0={args.t0}, t1={args.t1}, dt0={args.dt0})...")
    print(f"Geometries: {', '.join(args.geometry)}")

    solution = diffeqsolve(
        ode_fn,
        solver,
        args.t0,
        args.t1,
        args.dt0,
        y0,
        cosmo,
        saveat=saveat,
        adjoint=RecursiveCheckpointAdjoint(5),
    )

    print(f"Simulation complete!")

    # Build output PyTree
    print(f"\nPreparing output...")

    output = {
        "initial_conditions": lin_field,
        "z_near": z_near,
        "z_far": z_far,
        "scale_factors": snapshot_times,
        "cosmology": {
            "omega_c": jnp.array(cosmo.Omega_c),
            "omega_b": jnp.array(cosmo.Omega_b),
            "omega_m": jnp.array(cosmo.Omega_m),
            "h": jnp.array(cosmo.h),
            "sigma8": jnp.array(cosmo.sigma8),
            "n_s": jnp.array(cosmo.n_s),
            "w0": jnp.array(cosmo.w0),
            "wa": jnp.array(cosmo.wa),
        },
        "simulation": {
            "mesh_size": jnp.array(mesh_size),
            "box_size": jnp.array(box_size),
            "observer_position": jnp.array(observer_position),
            "t0": jnp.array(args.t0),
            "t1": jnp.array(args.t1),
            "lpt_order": jnp.array(args.lpt_order),
            "seed": jnp.array(args.seed),
        },
    }

    # Add geometry-specific outputs (reverse to get chronological order)
    if len(args.geometry) == 1:
        # Single geometry - solution.ys is array
        if "spherical" in args.geometry:
            output["lightcone_spherical"] = solution.ys[::-1]
        elif "flat" in args.geometry:
            output["lightcone_flat"] = solution.ys[::-1]
    else:
        # Multiple geometries - solution.ys is tuple
        results = solution.ys if isinstance(solution.ys, tuple) else (solution.ys, )
        for i, geom in enumerate(args.geometry):
            if geom == "spherical":
                output["lightcone_spherical"] = results[i][::-1]
            elif geom == "flat":
                output["lightcone_flat"] = results[i][::-1]

    # Save using orbax checkpoint
    print(f"\nSaving to {args.output_path}...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_sharded(output, str(output_path), overwrite=True)

    print(f"\n{'=' * 60}")
    print(f"✓ Simulation complete")
    print(f"✓ Output saved to: {args.output_path}")
    print(f"✓ Geometries: {', '.join(args.geometry)}")
    print(f"✓ Number of shells: {len(snapshot_times)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
