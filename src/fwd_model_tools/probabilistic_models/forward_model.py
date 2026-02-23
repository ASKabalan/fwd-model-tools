"""Deterministic forward models used by the probabilistic layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import DensityField
from ..fields.painting import PaintingOptions
from ..lensing import born, raytrace
from ..pm import DriftInterp, NoCorrection, NoInterp, ReversibleDoubleKickDrift, lpt, nbody
from .config import Configurations

__all__ = ["make_full_field_model"]


def make_full_field_model(config: Configurations, ):
    """Build the deterministic forward model returning kappa maps and lightcone."""

    geometry = config.geometry
    if geometry not in {"flat", "spherical"}:
        raise ValueError("geometry must be either 'flat' or 'spherical'")

    def forward_model(cosmo, initial_conditions):
        # warmstart NZ
        if config.nz_shear is not None:
            if isinstance(config.nz_shear, (list, tuple)):
                for nz in config.nz_shear:
                    if callable(nz) and hasattr(nz, '_norm'):
                        nz._norm = None
            else:
                nz = config.nz_shear
                if callable(nz) and hasattr(nz, '_norm'):
                    nz._norm = None

        dx_field, p_field = lpt(
            cosmo,
            initial_conditions,
            ts=config.t0,
            order=config.lpt_order,
        )

        # Create solver with appropriate interp_kernel based on geometry
        if config.drift_on_lightcone:
            interp_kernel = DriftInterp(painting=PaintingOptions(target=geometry))
        else:
            interp_kernel = NoInterp(painting=PaintingOptions(target=geometry))

        solver = ReversibleDoubleKickDrift(
            pgd_kernel=NoCorrection(),
            interp_kernel=interp_kernel,
        )

        lightcone = nbody(
            cosmo,
            dx_field,
            p_field,
            t1=config.t1,
            dt0=config.dt0,
            nb_shells=config.number_of_shells,
            solver=solver,
            adjoint=config.adjoint,
            checkpoints=config.checkpoints,
        )

        lensing_fn = raytrace if config.lensing == "raytrace" else born
        kappa_fields = lensing_fn(
            cosmo,
            lightcone,
            nz_shear=config.nz_shear,
            min_z=config.min_redshift,
            max_z=config.max_redshift,
        )

        return kappa_fields, lightcone

    return jax.jit(forward_model)
