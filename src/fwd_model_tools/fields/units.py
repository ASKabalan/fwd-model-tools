from typing import Any, Optional

import jax.numpy as jnp
from jaxpm.distributed import uniform_particles
from jaxtyping import Array

from fwd_model_tools._src.base._enums import ConvergenceUnit, DensityUnit, PhysicalUnit, PositionUnit


def convert_units(
    array: Array,
    origin: PhysicalUnit,
    destination: PhysicalUnit,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    h: Optional[float] = None,  # Hubble parameter, needed for MPC conversions
    omega_m: Optional[float] = None,  # Matter density parameter, needed for MSUN conversions
    mean_density: Optional[float] = None,  # Mean density for overdensity conversions
    volume_element: Optional[float] = None,  # Volume per voxel/pixel for density conversions
    sharding: Optional[Any] = None,
) -> Array:
    """
    Convert array between units of the same physical quantity.

    Parameters
    ----------
    array : Array
        The data to convert
    origin : PhysicalUnit
        Current unit of the array
    destination : PhysicalUnit
        Target unit
    mesh_size : tuple of int
        Grid dimensions (nx, ny, nz)
    box_size : tuple of float
        Box size in Mpc/h
    h : float, optional
        Hubble parameter (h = H0 / 100 km/s/Mpc), required for MPC conversions

    Returns
    -------
    Array
        Array in the new units
    """
    # Same unit, no conversion needed
    if origin == destination:
        return array

    # Validate same unit family
    if type(origin) is not type(destination):
        raise TypeError(
            f"Cannot convert between different unit families: {type(origin).__name__} -> {type(destination).__name__}")

    # Dispatch to specific converter
    if isinstance(origin, PositionUnit):
        return _convert_position(array, origin, destination, mesh_size, box_size, sharding)
    elif isinstance(origin, DensityUnit):
        return _convert_density(array, origin, destination, volume_element, omega_m, h, mean_density)
    elif isinstance(origin, ConvergenceUnit):
        return _convert_convergence(array, origin, destination)
    else:
        raise NotImplementedError(f"Conversion not implemented for {type(origin).__name__}")


def _convert_position(
    array: Array,
    origin: PositionUnit,
    destination: PositionUnit,
    mesh_size: tuple[int, int, int],
    box_size: tuple[float, float, float],
    sharding: Optional[Any] = None,
) -> Array:
    """
    Convert position units via GRID_ABSOLUTE as the canonical hub.

    Parameters
    ----------
    array : Array
        Position data to convert
    origin : PositionUnit
        Current unit
    destination : PositionUnit
        Target unit
    mesh_size : tuple of int
        Grid dimensions (nx, ny, nz)
    box_size : tuple of float
        Box size in Mpc/h
    sharding : optional
        JAX sharding for distributed computation

    Returns
    -------
    Array
        Positions in new units
    """
    mesh_size_arr = jnp.array(mesh_size)
    box_size_arr = jnp.array(box_size)

    # Step 1: Convert to GRID_ABSOLUTE (hub)
    if origin == PositionUnit.GRID_ABSOLUTE:
        grid_coords = array
    elif origin == PositionUnit.GRID_RELATIVE:
        grid_coords = array + uniform_particles(mesh_size, sharding=sharding)
    elif origin == PositionUnit.MPC_H:
        grid_coords = (array / box_size_arr) * mesh_size_arr
    else:
        raise ValueError(f"Unknown origin unit: {origin}")

    # Step 2: Convert from GRID_ABSOLUTE to destination
    if destination == PositionUnit.GRID_ABSOLUTE:
        return grid_coords
    elif destination == PositionUnit.GRID_RELATIVE:
        return grid_coords - uniform_particles(mesh_size, sharding=sharding)
    elif destination == PositionUnit.MPC_H:
        return (grid_coords / mesh_size_arr) * box_size_arr
    else:
        raise ValueError(f"Unknown destination unit: {destination}")


def _convert_density(
    array: Array,
    origin: DensityUnit,
    destination: DensityUnit,
    volume_element: Optional[float] = None,
    omega_m: Optional[float] = None,
    h: Optional[float] = None,
    mean_density: Optional[float] = None,
) -> Array:
    """
    Convert between density units.

    Uses DENSITY as the canonical hub.

    Parameters
    ----------
    array : Array
        Input density field
    origin : DensityUnit
        Current unit of the array
    destination : DensityUnit
        Target unit
    volume_element : float
        Volume per voxel/pixel in (Mpc/h)³.

        3D voxel:
            volume_element = (Lx * Ly * Lz) / (Nx * Ny * Nz)

        Flat sky pixel:
            volume_element = (θ_pix × r)² × Δr
            where θ_pix is pixel angular size in radians,
            r is comoving distance in Mpc/h

        HEALPix pixel (thin shell):
            volume_element = (4π / 12 / nside²) × r² × Δr

        HEALPix pixel (thick shell):
            volume_element = (4π / 12 / nside²) × (R_max³ - R_min³) / 3

    omega_m : float, optional
        Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
    h : float, optional
        Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
    mean_density : float, optional
        Mean density ρ̄ in particles per (Mpc/h)³. Required when converting
        FROM OVERDENSITY to other units.

    Returns
    -------
    Array
        Converted density field

    Notes
    -----
    Conversion formulas:

        COUNTS → DENSITY:       ρ = N / V
        DENSITY → OVERDENSITY:  δ = ρ / ρ̄ - 1
        DENSITY → MSUN:         ρ_phys = ρ × m_particle / V × h

    where:
        m_particle = Ωm × ρ_crit × V_physical
        ρ_crit = 2.775 × 10¹¹ h² M☉/Mpc³
        V_physical = V / h³
    """
    if volume_element is None:
        raise ValueError("volume_element is required for density unit conversions")
    # Compute mass per particle if needed for MSUN conversions
    mass_per_particle = None
    if origin == DensityUnit.MSUN_H_PER_MPC3 or destination == DensityUnit.MSUN_H_PER_MPC3:
        if omega_m is None or h is None:
            raise ValueError("omega_m and h required for MSUN_H_PER_MPC3 conversion")
        rho_crit = 2.775e11 * h**2  # M☉/Mpc³
        rho_matter = omega_m * rho_crit  # M☉/Mpc³
        volume_physical = volume_element / h**3  # Mpc³
        mass_per_particle = rho_matter * volume_physical  # M☉ (for mean 1 particle/voxel)

    # ===== Step 1: Convert to DENSITY (hub) =====

    if origin == DensityUnit.DENSITY:
        density = array

    elif origin == DensityUnit.COUNTS:
        density = array / volume_element

    elif origin == DensityUnit.OVERDENSITY:
        if mean_density is None:
            raise ValueError("mean_density required to convert from OVERDENSITY")
        density = (array + 1.0) * mean_density

    elif origin == DensityUnit.MSUN_H_PER_MPC3:
        mass_msun_h = array * volume_element  # M☉/h per voxel
        mass_msun = mass_msun_h / h  # M☉ per voxel
        counts = mass_msun / mass_per_particle  # particles per voxel
        density = counts / volume_element  # particles per (Mpc/h)³

    else:
        raise ValueError(f"Unknown origin unit: {origin}")

    # ===== Step 2: Convert from DENSITY to destination =====

    if destination == DensityUnit.DENSITY:
        return density

    elif destination == DensityUnit.COUNTS:
        return density * volume_element

    elif destination == DensityUnit.OVERDENSITY:
        rho_mean = jnp.mean(density)
        eps = jnp.finfo(density.dtype).eps
        safe_mean = jnp.where(rho_mean == 0, eps, rho_mean)
        return density / safe_mean - 1.0

    elif destination == DensityUnit.MSUN_H_PER_MPC3:
        counts = density * volume_element  # particles per voxel
        mass_msun = counts * mass_per_particle  # M☉ per voxel
        mass_msun_h = mass_msun * h  # M☉/h per voxel
        return mass_msun_h / volume_element  # M☉/h / (Mpc/h)³

    else:
        raise ValueError(f"Unknown destination unit: {destination}")


def _convert_convergence(
    array: Array,
    origin: ConvergenceUnit,
    destination: ConvergenceUnit,
) -> Array:
    """
    Convert convergence units.

    Both DIMENSIONLESS and EFFECTIVE_DENSITY are κ = Σ/Σ_crit,
    just different semantic interpretations.
    """
    # Both are numerically identical
    return array
