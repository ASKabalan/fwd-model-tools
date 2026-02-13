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

    density_plane_smoothing: float
    nz_shear: list[Any]
    fiducial_cosmology: Any
    sigma_e: float
    priors: dict[str, Any]
    t0: float
    dt0: float
    t1: float
    adjoint: str = 'checkpointed'
    min_redshift: float = 0.01
    max_redshift: float = 3.0
    geometry: str = "spherical"
    log_lightcone: bool = False
    log_ic: bool = False
    ells: Array = field(default_factory=lambda: jnp.arange(2, 2048))
    number_of_shells: int = 8
    lensing: str = "born"
    lpt_order: int = 2
