from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxpm.pm import lpt as jaxpm_lpt

from ..fields import DensityField, FieldStatus, ParticleField, particle_from_density
from ..utils import compute_snapshot_scale_factors

__all__ = ["lpt"]

@partial(jax.jit, static_argnames=['order'])
def lpt(
    cosmo,
    initial_field: DensityField,
    a,
    order: int = 1,
) -> Tuple[ParticleField, ParticleField]:
    """
    Compute LPT displacements/momenta for a DensityField.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology describing the background expansion.
    initial_field : DensityField
        Linear density field packaged with mesh metadata.
    a : float or array-like
        Scale factor(s) at which to evaluate the growth.
        - If scalar: returns (X, Y, Z, 3) shaped ParticleFields
        - If array: returns (N, X, Y, Z, 3) shaped ParticleFields (batched lightcone)
    order : int, default=1
        LPT order (1 or 2 supported via underlying JAXPM implementation).

    Returns
    -------
    tuple[ParticleField, ParticleField]
        Displacement and momentum particle fields.
        Shape depends on input `a`:
        - Scalar a: (X, Y, Z, 3)
        - Array a of shape (N,): (N, X, Y, Z, 3)

    Examples
    --------
    Single scale factor:
    >>> dx, p = lpt(cosmo, field, a=0.5, order=1)
    >>> dx.array.shape  # (256, 256, 256, 3)

    Batched lightcone:
    >>> scale_factors = compute_snapshot_scale_factors(cosmo, field)
    >>> dx, p = lpt(cosmo, field, a=scale_factors, order=1)
    >>> dx.array.shape  # (10, 256, 256, 256, 3) if nb_shells=10
    """
    if not isinstance(initial_field, DensityField):
        raise TypeError("initial_field must be a DensityField instance.")
    if order not in (1, 2):
        raise ValueError("order must be either 1 or 2.")
    if initial_field.status != FieldStatus.INITIAL_FIELD:
        raise ValueError(
            "initial_field must have status FieldStatus.INITIAL_FIELD.")
    a = jnp.atleast_1d(a)

    if a.size > 1:
        a = a.reshape((1, 1, -1, 1))
        if a.shape[2] != initial_field.shape[2]:
            raise ValueError(
                "When passing multiple scale factors, the number of scale factors "
                "must match the number of shells in the DensityField.")

    dx, p, _ = jaxpm_lpt(
        cosmo,
        initial_field.array,
        particles=None,
        a=a,
        halo_size=initial_field.halo_size,
        sharding=initial_field.sharding,
        order=order,
    )

    status = FieldStatus.LPT1 if order == 1 else FieldStatus.LPT2
    dx_field = particle_from_density(dx,
                                      initial_field,
                                      scale_factors=a,
                                      status=status)
    p_field = particle_from_density(p,
                                     initial_field,
                                     scale_factors=a,
                                     status=status)
    return dx_field, p_field
