"""
Symplectic ODE solvers for N-body simulations with ParticleField support.

This module provides symplectic integrators adapted from JaxPM that work with
ParticleField objects instead of raw arrays, automatically preserving metadata
like sharding, observer position, and field status throughout integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxpm.growth import E, Gf, dGfa
from jaxpm.growth import growth_factor as Gp
from jaxpm.pm import pm_forces

from ...fields import ParticleField, PositionUnit

__all__ = ["single_ode", "symplectic_fpm"]


def single_ode(cosmo, reference_field: ParticleField):
    mesh_shape = reference_field.mesh_size
    paint_absolute_pos = reference_field.unit == PositionUnit.GRID_ABSOLUTE
    halo_size = reference_field.halo_size
    sharding = reference_field.sharding

    def nbody_ode(a, state, args):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = (pm_forces(
            pos.array,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
        ) * 1.5 * cosmo.Omega_m)

        # Computes the update of position (drift)
        dpos = 1.0 / (a**3 * E(cosmo, a)) * vel

        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces

        stacked = jnp.stack([dpos, dvel])

        return ParticleField.FromDensityMetadata(
            array=stacked,
            field=pos,
            scale_factors=a,
        )

    return nbody_ode

#TODO Growth is not efficient remove the optional to use it and make it mandatory
def symplectic_fpm(
    cosmo,
    reference_field: ParticleField,
    dt0: float,
    use_growth: bool = False,
):
    """
    Create drift, pgd, kick, and first_kick functions for FastPM-style symplectic integration.

    This function returns operators for FastPM integration with optional growth factor
    corrections and PGD correction. All operators work with ParticleField objects.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for background expansion.
    reference_field : ParticleField
        Reference field containing all metadata (mesh_size, box_size, sharding,
        halo_size, etc.) needed for the integration.
    dt0 : float
        Base time step size in scale factor units.
    pgd_kernel : AbstractCorrection
        Correction kernel. If NoCorrection is used, returns zeros.
    use_growth : bool, optional
        Whether to use growth factor corrections (True) or simple scale factor
        evolution (False). Default is False.

    Returns
    -------
    drift : callable
        Drift operator: (a, vel, args) -> dpos
    pgd : callable
        PGD correction operator: (a, pos, args) -> dpos_correction
    kick : callable
        Kick operator: (a, pos, args) -> dvel
    first_kick : callable
        First kick operator for initialization: (a, pos, args) -> dvel

    Notes
    -----
    The FastPM integrator uses modified kick and drift factors that account for
    geometric means of scale factors between time steps. This improves accuracy
    compared to the standard leapfrog scheme.

    When use_growth=True, the integrator uses linear growth factors Gf and Gp
    for more accurate evolution in the mildly non-linear regime.

    The PGD correction is applied after drift and before kick, so forces are
    computed at PGD-corrected positions.

    Examples
    --------
    >>> pgd_kernel = PGDKernel(alpha=1.0, kl=0.3, ks=1.0)
    >>> drift, pgd, kick, first_kick = symplectic_ode_fpm(
    ...     cosmo,
    ...     dx_field,
    ...     dt0=0.01,
    ...     pgd_kernel=pgd_kernel,
    ...     use_growth=True
    ... )
    >>> # Initialize with first kick
    >>> dvel = first_kick(a=0.1, pos=initial_positions, args=cosmo)
    """
    # Extract metadata from reference field
    mesh_shape = reference_field.mesh_size
    paint_absolute_pos = reference_field.unit == PositionUnit.GRID_ABSOLUTE
    halo_size = reference_field.halo_size
    sharding = reference_field.sharding

    def drift(a, vel, args):
        """
        Drift operator with FastPM geometric mean correction.

        Parameters
        ----------
        a : float
            Current scale factor.
        vel : ParticleField
            Velocity field.
        args : tuple
            First element is cosmology object.

        Returns
        -------
        ParticleField
            Position update (displacement).
        """
        # Get the time steps
        t0 = a
        t1 = a + dt0
        # Set the scale factors
        ai = t0
        ac = (t0 * t1)**0.5  # Geometric mean of t0 and t1
        af = t1

        if use_growth:
            # Use growth factor to compute the drift factor
            drift_contr = (Gp(cosmo, af) - Gp(cosmo, ai)) / dGfa(cosmo, ac)
        else:
            drift_contr = (af - ai) / ac

        # Computes the update of position (drift)
        dpos = 1 / (ac**3 * E(cosmo, ac)) * vel.array

        return ParticleField.FromDensityMetadata(
            array=dpos * (drift_contr / dt0),
            field=vel,
            scale_factors=a,
        )

    def kick(a, pos, args):
        """
        Kick operator with FastPM two-point correction.

        Parameters
        ----------
        a : float
            Current scale factor.
        pos : ParticleField
            Position field.
        args : cosmology object or tuple
            Cosmology describing the background expansion.

        Returns
        -------
        ParticleField
            Velocity update (acceleration).
        """
        # Computes the update of velocity (kick)
        # Get the time steps
        t0 = a
        t1 = t0 + dt0
        t2 = t1 + dt0
        t0t1 = (t0 * t1)**0.5  # Geometric mean of t0 and t1
        t1t2 = (t1 * t2)**0.5  # Geometric mean of t1 and t2
        # Set the scale factors
        ac = t1

        # Compute forces using JaxPM (pass raw array)
        forces_array = (pm_forces(
            pos.array,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
        ) * 1.5 * cosmo.Omega_m)

        # Computes the update of velocity (kick)
        dvel = 1.0 / (ac**2 * E(cosmo, ac)) * forces_array

        # First kick control factor
        if use_growth:
            # Use growth factor to compute the kick factors
            kick_factor_1 = (Gf(cosmo, t1) - Gf(cosmo, t0t1)) / dGfa(cosmo, t1)
            # Second kick control factor
            kick_factor_2 = (Gf(cosmo, t1t2) - Gf(cosmo, t1)) / dGfa(cosmo, t1)
        else:
            # Use scale factor to compute the kick factors
            kick_factor_1 = (t1 - t0t1) / t1
            # Second kick control factor
            kick_factor_2 = (t2 - t1t2) / t2

        # Wrap back into ParticleField
        return ParticleField.FromDensityMetadata(
            array=dvel * ((kick_factor_1 + kick_factor_2) / dt0),
            field=pos,
            scale_factors=a,
        )

    def first_kick(a, pos, args):
        """
        First kick operator for initialization.

        Parameters
        ----------
        a : float
            Initial scale factor.
        pos : ParticleField
            Initial position field.
        args : cosmology object or tuple
            Cosmology describing the background expansion.

        Returns
        -------
        ParticleField
            Initial velocity update.
        """
        # Computes the update of velocity (kick)
        # Get the time steps
        t0 = a
        t1 = t0 + dt0
        t0t1 = (t0 * t1)**0.5  # Geometric mean of t0 and t1

        # Compute forces using JaxPM (pass raw array)
        forces_array = (pm_forces(
            pos.array,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
        ) * 1.5 * cosmo.Omega_m)

        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces_array

        # First kick control factor
        if use_growth:
            kick_factor = (Gf(cosmo, t0t1) - Gf(cosmo, t0)) / dGfa(cosmo, t0)
        else:
            kick_factor = (t0t1 - t0) / t0

        # Wrap back into ParticleField
        return ParticleField.FromDensityMetadata(
            array=dvel * (kick_factor / dt0),
            field=pos,
            scale_factors=a,
        )

    return drift, kick, first_kick
