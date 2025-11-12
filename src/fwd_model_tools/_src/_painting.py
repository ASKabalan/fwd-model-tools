"""
Pure painting functions for ParticleField operations.

These functions are designed to work with jax.lax.map for efficient batched painting.
"""
from __future__ import annotations

from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from jaxpm.painting import cic_paint, cic_paint_2d, cic_paint_dx
from jaxpm.spherical import paint_particles_spherical
from jaxpm.distributed import uniform_particles

PaintMode = Literal["relative", "absolute"]
SphericalScheme = Literal["ngp", "bilinear", "rbf_neighbor"]
Float = float


def _single_paint(
    array: jax.Array,
    mesh_size: Tuple[int, int, int],
    box_size: Tuple[float, float, float],
    observer_position: Tuple[float, float, float],
    sharding: any,
    halo_size: Tuple[int, int],
    mode: PaintMode,
    mesh: jax.Array | None,
    weights: jax.Array | float,
    chunk_size: int,
) -> jax.Array:
    """
    Paint a single shell of particles onto a 3D density mesh.

    Parameters
    ----------
    array : jax.Array
        Particle positions or displacements, shape (X, Y, Z, 3).
    mesh_size : Tuple[int, int, int]
        3D mesh dimensions.
    box_size : Tuple[float, float, float]
        Physical box size in Mpc.
    observer_position : Tuple[float, float, float]
        Observer position as fractions [0, 1].
    sharding : any
        JAX sharding specification.
    halo_size : Tuple[int, int]
        Halo padding size (left, right).
    mode : PaintMode
        "relative" for displacements, "absolute" for positions.
    mesh : jax.Array | None
        Pre-allocated mesh for absolute mode (optional).
    weights : jax.Array | float
        Particle weights for painting.
    chunk_size : int
        Chunk size for painting operations.

    Returns
    -------
    jax.Array
        Painted 3D density mesh, shape mesh_size.
    """
    mode = mode.lower()
    if mode == "relative":
        density = cic_paint_dx(
            array,
            halo_size=halo_size,
            sharding=sharding,
            weight=weights,
            chunk_size=chunk_size,
        )
    elif mode == "absolute":
        grid_mesh = (
            mesh if mesh is not None else jnp.zeros(mesh_size, dtype=array.dtype)
        )
        density = cic_paint(
            grid_mesh,
            array,
            weight=weights,
            halo_size=halo_size,
            sharding=sharding,
        )
    else:
        raise ValueError("mode must be either 'relative' or 'absolute'")

    return density


def _single_paint_2d(
    array_center : Tuple[jax.Array, float],
    mesh_size: Tuple[int, int, int],
    box_size: Tuple[float, float, float],
    observer_position: Tuple[float, float, float],
    sharding: any,
    flatsky_npix: Tuple[int, int],
    halo_size: Tuple[int, int],
    density_plane_width: Float,
    weights: jax.Array | float | None,
    mode: PaintMode,
    max_comoving_radius: float,
    nb_shells: int,
) -> jax.Array:
    """
    Paint a single shell of particles onto a 2D flat-sky grid.

    Parameters
    ----------
    array : jax.Array
        Particle positions or displacements, shape (X, Y, Z, 3).
    mesh_size : Tuple[int, int, int]
        3D mesh dimensions.
    box_size : Tuple[float, float, float]
        Physical box size in Mpc.
    observer_position : Tuple[float, float, float]
        Observer position as fractions [0, 1].
    sharding : any
        JAX sharding specification.
    flatsky_npix : Tuple[int, int]
        2D flat-sky grid dimensions.
    halo_size : Tuple[int, int]
        Halo padding size (left, right).
    center : float
        Center of the density plane in Mpc.
    density_plane_width : Float
        Physical width of the density plane in Mpc.
    weights : jax.Array | float | None
        Particle weights for painting.
    mode : PaintMode
        "relative" for displacements, "absolute" for positions.
    max_comoving_radius : float
        Maximum comoving radius from observer.
    nb_shells : int
        Number of shells in the lightcone.

    Returns
    -------
    jax.Array
        Painted 2D flat-sky map, shape flatsky_npix.
    """
    nx, ny, nz = mesh_size
    mpc_x, mpc_y, mpc_z = box_size
    array, center = array_center

    # Compute density width from max_comoving_radius and nb_shells
    density_width = max_comoving_radius / nb_shells
    width = density_width if density_plane_width is None else density_plane_width
    width = (width * nz) / mpc_z  # thickness in grid units
    center_grid = (center * nz) / mpc_z  # center in grid units

    if mode == "relative":
        positions = uniform_particles(mesh_size, sharding=sharding) + array
    else:
        positions = array

    xy = positions[..., :2]
    dz = positions[..., 2]

    # Scale xy coordinates from mesh grid to flatsky_npix grid
    scale_x = flatsky_npix[0] / nx
    scale_y = flatsky_npix[1] / ny
    xy = xy * jnp.array([scale_x, scale_y])

    # Apply 2D periodic conditions
    xy = jnp.mod(xy, jnp.array(flatsky_npix))

    # Compute weights based on z-position and user-defined weights
    weights_dz = jnp.where(
        (dz > (center_grid - width / 2)) & (dz <= (center_grid + width / 2)), 1.0, 0.0
    )
    painting_weights = weights_dz if weights is None else weights * weights_dz

    # Prepare the output flat-sky grid
    grid = jnp.zeros(flatsky_npix, dtype=array.dtype)

    if sharding is not None:
        xy = jax.lax.with_sharding_constraint(xy, sharding)

    # Apply CIC painting
    painted = cic_paint_2d(grid, xy, painting_weights)

    shell_thickness_physical = width  # Already in physical units
    pixel_area = (mpc_x / flatsky_npix[0]) * (mpc_y / flatsky_npix[1])
    pixel_volume = pixel_area * shell_thickness_physical

    density_plane = painted / pixel_volume

    return density_plane


def _single_paint_spherical(
    array_center: Tuple[jax.Array, float],
    mesh_size: Tuple[int, int, int],
    box_size: Tuple[float, float, float],
    observer_position: Tuple[float, float, float],
    sharding: any,
    nside: int,
    halo_size: Tuple[int, int],
    mode: PaintMode,
    scheme: SphericalScheme,
    weights: jax.Array | None,
    density_plane_width: Float | None,
    kernel_width_arcmin: float | None,
    smoothing_interpretation: str,
    paint_nside: int | None,
    ud_grade_power: float,
    ud_grade_order_in: str,
    ud_grade_order_out: str,
    ud_grade_pess: bool,
    max_comoving_radius: float,
    nb_shells: int,
) -> jax.Array:
    """
    Paint a single shell of particles onto a HEALPix grid.

    Parameters
    ----------
    array : jax.Array
        Particle positions or displacements, shape (X, Y, Z, 3).
    mesh_size : Tuple[int, int, int]
        3D mesh dimensions.
    box_size : Tuple[float, float, float]
        Physical box size in Mpc.
    observer_position : Tuple[float, float, float]
        Observer position as fractions [0, 1].
    sharding : any
        JAX sharding specification.
    nside : int
        HEALPix nside parameter.
    halo_size : Tuple[int, int]
        Halo padding size (left, right).
    center : float
        Center of the density shell in Mpc.
    mode : PaintMode
        "relative" for displacements, "absolute" for positions.
    scheme : SphericalScheme
        Painting method: "ngp", "bilinear", or "rbf_neighbor".
    weights : jax.Array | None
        Particle weights for painting.
    density_plane_width : Float | None
        Physical width of the density shell in Mpc.
    kernel_width_arcmin : float | None
        Kernel width in arcminutes (for RBF method).
    smoothing_interpretation : str
        How to interpret kernel_width_arcmin ("fwhm" or "sigma").
    paint_nside : int | None
        Higher nside for painting before downgrading.
    ud_grade_power : float
        Power for ud_grade operation.
    ud_grade_order_in : str
        HEALPix ordering for input ("RING" or "NESTED").
    ud_grade_order_out : str
        HEALPix ordering for output ("RING" or "NESTED").
    ud_grade_pess : bool
        Use pessimistic ud_grade.
    max_comoving_radius : float
        Maximum comoving radius from observer.
    nb_shells : int
        Number of shells in the lightcone.

    Returns
    -------
    jax.Array
        Painted HEALPix map, shape (npix,).
    """
    array , center = array_center
    if mode == "relative":
        positions = uniform_particles(mesh_size, sharding=sharding) + array
    else:
        positions = array

    # Compute density width from max_comoving_radius and nb_shells
    density_width = max_comoving_radius / nb_shells
    width = density_width if density_plane_width is None else density_plane_width
    rmin, rmax = center - (width / 2), center + (width / 2)

    # Observer position in Mpc
    observer_position_mpc = tuple(
        frac * length for frac, length in zip(observer_position, box_size)
    )

    def error_message(rmin, rmax, density_plane_width, max_comoving_radius):
        if (
            rmin < density_plane_width / 2
            or rmax > max_comoving_radius - density_plane_width / 2
        ):
            return (
                f"Requested spherical shell (rmin={rmin}, rmax={rmax}) "
                f"lies outside the box limits [0, {max_comoving_radius}]. "
                "Adjust center to be between {density_plane_width / 2} and "
                f"{max_comoving_radius - density_plane_width / 2}."
            )

    jax.debug.callback(
        error_message,
        rmin,
        rmax,
        density_plane_width if density_plane_width is not None else width,
        max_comoving_radius,
    )

    if sharding is not None:
        positions = jax.lax.with_sharding_constraint(positions, sharding)

    painted = paint_particles_spherical(
        positions=positions,
        nside=nside,
        observer_position=observer_position_mpc,
        R_min=rmin,
        R_max=rmax,
        box_size=box_size,
        mesh_shape=mesh_size,
        weights=weights,
        method=scheme,
        kernel_width_arcmin=kernel_width_arcmin,
        smoothing_interpretation=smoothing_interpretation,
        paint_nside=paint_nside,
        ud_grade_power=ud_grade_power,
        ud_grade_order_in=ud_grade_order_in,
        ud_grade_order_out=ud_grade_order_out,
        ud_grade_pess=ud_grade_pess,
    )
    return painted
