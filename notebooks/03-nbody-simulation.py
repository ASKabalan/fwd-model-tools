"""
N-body Simulation and Lightcone Generation

This script demonstrates running N-body simulations with lightcone generation. Key features:

1. Generate Gaussian initial conditions
2. Compute LPT displacements and momenta at initial scale factor
3. Run N-body simulation with different geometry options:
   - geometry='particles': Save particle positions (flexible but memory-intensive)
   - geometry="flat": Compute flat-sky projections on-the-fly (memory-efficient)
   - geometry="spherical": Compute HEALPix projections on-the-fly (memory-efficient)
4. Post-hoc painting of saved density fields to different projections
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax_cosmo as jc

from fwd_model_tools.field import FieldStatus, ParticleField, DensityField, FlatDensity, SphericalDensity
from fwd_model_tools.normal import gaussian_initial_conditions
from fwd_model_tools.pm import lpt, nbody
from fwd_model_tools.lensing_model import Planck18


def main():
    print("=" * 80)
    print("N-body Simulation and Lightcone Generation")
    print("=" * 80)

    # Create output directory for plots
    output_dir = "output/plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Configuration
    print("\n## Configuration")
    print("\nSet up simulation parameters. Using small mesh size for demonstration.")

    # Simulation parameters
    mesh_size = (64, 64, 64)
    box_size = (500.0, 500.0, 500.0)
    observer_position = (0.5, 0.5, 0.5)
    nside = 64
    flatsky_npix = (64, 64)

    # N-body parameters
    t0 = 0.1  # Initial scale factor
    t1 = 1.0  # Final scale factor (present day)
    dt0 = 0.05  # Time step

    # Random seed
    key = jax.random.PRNGKey(42)
    cosmo = Planck18()

    print(f"Mesh size: {mesh_size}")
    print(f"Box size: {box_size} Mpc/h")

    # Generate Initial Conditions
    print("\n## Generate Initial Conditions")
    print("\nCreate Gaussian initial conditions with the specified parameters.")

    gaussian_field = gaussian_initial_conditions(
        key=key,
        cosmo=cosmo,
        mesh_size=mesh_size,
        box_size=box_size,
        observer_position=observer_position,
        nside=nside,
        flatsky_npix=flatsky_npix,
    )

    print(f"Initial field: {gaussian_field}")
    print(f"Status: {gaussian_field.status}")

    # Compute LPT Initial Conditions
    print("\n## Compute LPT Initial Conditions")
    print("\nCompute LPT displacements and momenta at the initial scale factor t0.")

    dx_field, p_field = lpt(cosmo, gaussian_field, a=t0, order=1)

    print(f"Displacement field: {dx_field}")
    print(f"Array shape: {dx_field.array.shape}")
    print(f"Scale factors: {dx_field.scale_factors}")

    # Compute number of shells from field properties
    nb_shells = 6
    shells_to_plot = [0, nb_shells // 2, nb_shells - 1]
    shell_idx_array = jnp.array(shells_to_plot)
    max_radius = dx_field.max_comoving_radius
    density_plane_width = dx_field.density_width(nb_shells=nb_shells)
    print(f"Max comoving radius: {max_radius} Mpc/h")
    print(f"Density plane width: {density_plane_width} Mpc/h")
    print(f"Number of shells: {nb_shells}")

    # Run N-body with geometry='particles'
    print("\n## Run N-body with geometry='particles'")
    print("\nRun N-body simulation saving particle positions at each shell.")
    print("This is flexible (can paint to any projection later) but memory-intensive.")

    ts = jnp.linspace(t0, t1, nb_shells)
    density_snapshots = nbody(
        cosmo,
        dx_field,
        p_field,
        t1=t1,
        dt0=dt0,
        ts=ts,
        geometry='particles',
    )

    print(f"Density snapshots: {density_snapshots}")
    print(f"Array shape: {density_snapshots.array.shape}")
    print(f"Scale factors shape: {density_snapshots.scale_factors.shape}")

    shell_centers_mpc = jc.background.radial_comoving_distance(
        cosmo, density_snapshots.scale_factors
    )
    cosmo._workspace = {}
    print(f"Shell centers (Mpc/h): {shell_centers_mpc}")

    # Post-hoc Painting to Flat Geometry
    print("\n## Post-hoc Painting to Flat Geometry")
    print("\nNote: For post-hoc painting, we would typically save displacement fields (geometry='particles')")
    print("and then use ParticleField.paint_2d() on each snapshot.")
    print(f"Density snapshots contain {density_snapshots.array.shape[0]} shells.")

    flat_from_particles = density_snapshots.paint_2d(center=shell_centers_mpc)
    fig, axes = flat_from_particles[shell_idx_array].plot(
        figsize=(15, 5),
        titles=[f"Shell {i}, a={density_snapshots.scale_factors[i]:.3f}" for i in shells_to_plot],
    )
    flat_particles_path = os.path.join(output_dir, "flat_from_particles.png")
    fig.savefig(flat_particles_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved post-hoc flat maps to: {flat_particles_path}")

    density_snapshots = density_snapshots.replace(nside=nside)
    spherical_from_particles = density_snapshots.paint_spherical(
        center=shell_centers_mpc,
        scheme="ngp",
    )
    fig = spherical_from_particles[shell_idx_array].plot(
        figsize=(15, 15),
        titles=[f"Shell {i}, a={density_snapshots.scale_factors[i]:.3f}" for i in shells_to_plot],
        apply_log=False,
    )
    spherical_particles_path = os.path.join(output_dir, "spherical_from_particles.png")
    fig.savefig(spherical_particles_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved post-hoc spherical maps to: {spherical_particles_path}")

    # Memory-Efficient: On-the-fly Flat Painting
    print("\n## Memory-Efficient: On-the-fly Flat Painting")
    print("\nInstead of saving all 3D fields, compute flat-sky projections during integration.")
    print("This is memory-efficient.")

    flat_lightcone = nbody(
        cosmo,
        dx_field,
        p_field,
        t1=t1,
        dt0=dt0,
        nb_shells=nb_shells,
        ts=ts,
        geometry="flat",
    )

    print(f"Flat lightcone: {flat_lightcone}")
    print(f"Array shape: {flat_lightcone.array.shape}")
    print(f"Scale factors shape: {flat_lightcone.scale_factors.shape}")

    # Visualize Flat Lightcone
    print("\n## Visualize Flat Lightcone")
    print("\nSaving flat-sky lightcone density maps for selected shells.")

    fig, axes = flat_lightcone[shell_idx_array].plot(
        figsize=(15, 5),
        titles=[f"Shell {i}, a={flat_lightcone.scale_factors[i]:.3f}" for i in shells_to_plot]
    )
    flat_output_path = os.path.join(output_dir, "flat_lightcone.png")
    fig.savefig(flat_output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved flat lightcone plot to: {flat_output_path}")

    # Memory-Efficient: On-the-fly Spherical Painting
    print("\n## Memory-Efficient: On-the-fly Spherical Painting")
    print("\nCompute HEALPix projections during integration for maximum memory efficiency.")

    spherical_lightcone = nbody(
        cosmo,
        dx_field,
        p_field,
        t1=t1,
        dt0=dt0,
        nb_shells=nb_shells,
        geometry="spherical",
    )

    print(f"Spherical lightcone: {spherical_lightcone}")
    print(f"Array shape: {spherical_lightcone.array.shape}")
    print(f"Scale factors shape: {spherical_lightcone.scale_factors.shape}")

    # Visualize Spherical Lightcone
    print("\n## Visualize Spherical Lightcone")
    print("\nSaving HEALPix spherical lightcone density maps for selected shells.")

    fig = spherical_lightcone[shell_idx_array].plot(
        figsize=(15, 15),
        titles=[f"Shell {i}, a={spherical_lightcone.scale_factors[i]:.3f}" for i in shells_to_plot]
    )
    spherical_output_path = os.path.join(output_dir, "spherical_lightcone.png")
    fig.savefig(spherical_output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spherical lightcone plot to: {spherical_output_path}")

    # Summary
    print("\n## Memory Trade-off Summary")
    print("\nThe geometry parameter offers a memory-flexibility trade-off:")
    print("\n- geometry='particles': Saves particle positions at each shell")
    print("  - Pros: Flexible - can paint to any projection later")
    print("  - Cons: Memory-intensive (stores particle positions)")
    print("  - Use case: When you need multiple projections or post-processing")
    print("\n- geometry='flat': Computes flat-sky projections on-the-fly")
    print("  - Pros: Memory-efficient (only stores 2D maps)")
    print("  - Cons: Fixed projection, cannot change later")
    print("  - Use case: When you only need flat-sky maps")
    print("\n- geometry='spherical': Computes HEALPix projections on-the-fly")
    print("  - Pros: Memory-efficient, full-sky coverage")
    print("  - Cons: Fixed projection, cannot change later")
    print("  - Use case: When you only need all-sky maps")

    print("\n" + "=" * 80)
    print("Script completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
