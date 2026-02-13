from __future__ import annotations

from typing import Literal, Optional

import equinox as eqx
from jaxtyping import Array

__all__ = ["PaintingOptions"]

SphericalScheme = Literal["ngp", "bilinear", "rbf_neighbor"]
PaintingTarget = Literal["spherical", "flat", "density", "particles"]
DEFAULT_CHUNK_SIZE = 2**24


class PaintingOptions(eqx.Module):
    """
    Configuration for painting in nbody/lpt pipelines.

    This frozen configuration class specifies what type of output to produce
    and the parameters for painting particles onto density grids.

    Parameters
    ----------
    target : {"spherical", "flat", "density", "particles"}
        Output type:
        - "spherical": HEALPix maps via paint_spherical()
        - "flat": Flat-sky 2D grids via paint_2d()
        - "density": 3D density fields via paint()
        - "particles": Return particle displacements directly (no painting)

    scheme : {"ngp", "bilinear", "rbf_neighbor"}, default="bilinear"
        Interpolation scheme for spherical painting.

    kernel_width_arcmin : float or None, default=None
        Smoothing kernel width in arcminutes for spherical painting.

    smoothing_interpretation : str, default="fwhm"
        How to interpret kernel_width_arcmin ("fwhm" or "sigma").

    paint_nside : int or None, default=None
        Resolution for spherical painting (if different from field nside).

    ud_grade_power : float, default=0.0
        Power for ud_grade operation in spherical painting.

    ud_grade_order_in : str, default="RING"
        Input ordering for ud_grade.

    ud_grade_order_out : str, default="RING"
        Output ordering for ud_grade.

    ud_grade_pess : bool, default=False
        Pessimistic ud_grade mode.

    chunk_size : int, default=2**24
        Chunk size for 3D painting.

    weights : array, float, or None, default=None
        Weights for painting.

    batch_size : int or None, default=None
        Batch size for painting operations.

    Examples
    --------
    >>> # Spherical painting with bilinear interpolation
    >>> painting = PaintingOptions(target="spherical", scheme="bilinear")
    >>> lightcone = lpt(cosmo, field, a, painting=painting)

    >>> # Flat-sky painting
    >>> painting = PaintingOptions(target="flat")
    >>> lightcone = nbody(cosmo, dx, p, painting=painting)

    >>> # Return particles directly (default behavior when painting=None)
    >>> dx, p = lpt(cosmo, field, a)
    """

    # Output target
    target: PaintingTarget = eqx.field(static=True)

    # Spherical-specific
    scheme: SphericalScheme = eqx.field(static=True, default="bilinear")
    kernel_width_arcmin: Optional[float] = eqx.field(static=True, default=None)
    smoothing_interpretation: str = eqx.field(static=True, default="fwhm")
    paint_nside: Optional[int] = eqx.field(static=True, default=None)
    ud_grade_power: float = eqx.field(static=True, default=0.0)
    ud_grade_order_in: str = eqx.field(static=True, default="RING")
    ud_grade_order_out: str = eqx.field(static=True, default="RING")
    ud_grade_pess: bool = eqx.field(static=True, default=False)

    # 3D paint specific
    chunk_size: int = eqx.field(static=True, default=DEFAULT_CHUNK_SIZE)

    # Shared
    weights: Optional[Array | float] = 1.0
    batch_size: Optional[int] = eqx.field(static=True, default=None)

    # Physics
    drift_on_lightcone: bool = eqx.field(static=True, default=False)
