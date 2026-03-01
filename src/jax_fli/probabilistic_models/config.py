from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array


@dataclass
class Configurations:
    """Configuration shared by full-field and power-spectrum models.

    This dataclass intentionally stores only probabilistic / cosmological
    configuration. Geometry and mesh/box metadata belong to DensityField
    instances or separate function arguments, not here.
    """

    # Mandatory parameters
    mesh_size: tuple[int, int, int]
    box_size: tuple[float, float, float]
    # Cosmological parameters
    fiducial_cosmology: Any
    nz_shear: list[Any]
    # Inference parameters
    priors: dict[str, Any]
    sigma_e: float
    log_lightcone: bool = False
    # Simulation settings
    halo_size: tuple[float, float] = (0, 0)
    observer_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sharding: Any = None
    nside: int = None
    flatsky_npix: tuple[int, int] = None
    field_size: tuple[float, float] = None
    # Lensing parameters
    density_plane_smoothing: float = 0.0
    min_redshift: float = 0.01
    max_redshift: float = 3.0
    lensing: str = "born"
    # Simulation parameters
    lpt_order: int = 2
    t0: float = 0.01
    dt0: float = 0.01
    t1: float = 1.0
    adjoint: str = "checkpointed"
    checkpoints: int = 10
    number_of_shells: int = 8
    equal_vol: bool = False
    min_width: float = 50.0  # Mpc/h comoving, minimum shell width for equal_vol mode
    geometry: str = "spherical"
    drift_on_lightcone: bool = False
    # Power spectrum settings (for power-spectrum model, not used in full-field model)
    ells: Array = field(default_factory=lambda: jnp.arange(2, 2048))
    f_sky: float = 1.0  # sky fraction for Gaussian covariance mode-count
