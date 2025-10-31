from typing import Any, NamedTuple

import jax.numpy as jnp
from diffrax import RecursiveCheckpointAdjoint
from jaxtyping import Array


class Configurations(NamedTuple):
    """
    Configuration parameters for forward modeling and lensing simulations.

    This class encapsulates all configuration parameters needed for running
    forward models with JAXPM, including box geometry, time evolution, cosmology,
    observation parameters, and computational options.

    Parameters
    ----------
    field_size : float
        Angular size of the field in degrees.
    field_npix : int
        Number of pixels along one side of the field.
    box_shape : tuple
        Shape of the simulation box (nx, ny, nz).
    box_size : list
        Physical size of the box in each dimension (Mpc/h).
    number_of_shells : int
        Number of shells for lensing ray tracing. density_plane_width is computed as box_size[2] / number_of_shells.
    density_plane_npix : int
        Number of pixels per density plane side.
    nside : int
        HEALPix nside parameter for spherical geometry.
    density_plane_smoothing : float
        Smoothing scale for density planes (Mpc/h).
    nz_shear : list
        List of redshift distributions for shear observations.
    fiducial_cosmology : jax_cosmo.Cosmology
        Fiducial cosmology object.
    sigma_e : float
        Intrinsic ellipticity dispersion per component.
    priors : dict
        Dictionary of prior distributions for cosmological parameters.
    t0 : float
        Initial scale factor for time evolution.
    dt0 : float
        Initial time step for ODE integration.
    t1 : float
        Final scale factor for time evolution.
    sharding : Any or None, default=None
        JAX sharding specification for distributed computation.
    halo_size : int, default=0
        Halo exchange size for distributed painting operations.
    adjoint : RecursiveCheckpointAdjoint, default=RecursiveCheckpointAdjoint(5)
        Adjoint method for gradient computation during ODE integration.
    min_redshift : float, default=0.01
        Minimum redshift for source integration.
    max_redshift : float, default=3.0
        Maximum redshift for source integration.
    geometry : str, default="spherical"
        Coordinate system geometry: "spherical" (HEALPix) or "flat" (Cartesian).
    observer_position : tuple or list, default=(0.5, 0.5, 0.5)
        Observer position as fraction of box size (x, y, z) between 0 and 1.
    log_lightcone : bool, default=False
        Whether the probabilistic model should record the lightcone as a deterministic site.
    log_ic : bool, default=False
        Whether the probabilistic model should record the linear field as a deterministic site.

    Notes
    -----
    - For spherical geometry, output convergence maps are 1D HEALPix arrays
    - For flat geometry, output convergence maps are 2D Cartesian arrays
    - The sharding parameter must be consistent across all JAXPM function calls
    """

    field_size: float
    field_npix: int
    box_shape: tuple
    box_size: list
    number_of_shells: int
    density_plane_npix: int
    nside: int
    density_plane_smoothing: float
    nz_shear: list
    fiducial_cosmology: Any
    sigma_e: float
    priors: dict
    t0: float
    dt0: float
    t1: float
    sharding: Any | None = None
    halo_size: int = 0
    adjoint: RecursiveCheckpointAdjoint = RecursiveCheckpointAdjoint(5)
    min_redshift: float = 0.01
    max_redshift: float = 3.0
    geometry: str = "spherical"
    observer_position: tuple | list = (0.5, 0.5, 0.5)
    log_lightcone: bool = False
    log_ic: bool = False
    ells: Array = jnp.arange(2, 2048)
