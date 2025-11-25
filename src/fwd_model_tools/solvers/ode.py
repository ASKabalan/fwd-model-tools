"""
Symplectic ODE solvers for N-body simulations with ParticleField support.

This module provides symplectic integrators adapted from JaxPM that work with
ParticleField objects instead of raw arrays, automatically preserving metadata
like sharding, observer position, and field status throughout integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxpm.growth import E, Gf, dGfa
from jaxpm.growth import growth_factor as Gp
from jaxpm.pm import pm_forces

from ..fields import FieldStatus, ParticleField

__all__ = ["symplectic_ode", "symplectic_fpm_ode"]


def symplectic_ode(reference_field: ParticleField,
                   paint_mode: str = "relative"):
    """
    Create drift and kick functions for standard symplectic N-body integration.

    This function returns drift and kick operators that work with ParticleField objects.
    The symplectic integrator maintains the Hamiltonian structure of the N-body system.

    Parameters
    ----------
    reference_field : ParticleField
        Reference field containing all metadata (mesh_size, box_size, sharding,
        halo_size, etc.) needed for the integration.
    paint_mode : str, optional
        Painting mode for force computation: "relative" for displacements or
        "absolute" for absolute positions. Default is "relative".

    Returns
    -------
    drift : callable
        Drift operator: (a, vel, args) -> dpos
        Updates particle positions based on velocities.
    kick : callable
        Kick operator: (a, pos, args) -> dvel
        Updates particle velocities based on gravitational forces.

    Notes
    -----
    Both drift and kick functions accept and return ParticleField objects,
    automatically preserving all metadata (sharding, observer_position, etc.).

    The integrator uses the standard leapfrog scheme without growth factor corrections.

    Examples
    --------
    >>> drift, kick = symplectic_ode(dx_field, paint_mode="relative")
    >>> # Use with a symplectic integrator
    >>> dpos = drift(a=0.5, vel=velocity_field, args=cosmo)
    >>> dvel = kick(a=0.5, pos=position_field, args=cosmo)
    """
    # Extract metadata from reference field
    mesh_shape = reference_field.mesh_size
    paint_absolute_pos = (paint_mode == "absolute")
    halo_size = reference_field.halo_size
    sharding = reference_field.sharding

    def drift(a, vel, args):
        """
        Drift operator: updates positions based on velocities.

        Parameters
        ----------
        a : float
            Scale factor.
        vel : ParticleField
            Velocity field.
        args : cosmology object
            Cosmology describing the background expansion.

        Returns
        -------
        ParticleField
            Position update (displacement).
        """
        cosmo = args
        # Computes the update of position (drift)
        dpos = 1 / (a**3 * E(cosmo, a)) * vel

        return dpos

    def kick(a, pos, args):
        """
        Kick operator: updates velocities based on gravitational forces.

        Parameters
        ----------
        a : float
            Scale factor.
        pos : ParticleField
            Position field.
        args : cosmology object
            Cosmology describing the background expansion.

        Returns
        -------
        ParticleField
            Velocity update (acceleration).
        """
        cosmo = args

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

        # Wrap back into ParticleField
        return ParticleField.FromDensityMetadata(
            dvel,
            pos,
            scale_factors=a,
        )

    return drift, kick


def symplectic_fpm_ode(reference_field: ParticleField,
                       dt0: float,
                       paint_mode: str = "relative",
                       use_growth: bool = False):
    """
    Create drift, kick, and first_kick functions for FastPM-style symplectic integration.

    This function returns operators for FastPM integration with optional growth factor
    corrections. All operators work with ParticleField objects.

    Parameters
    ----------
    reference_field : ParticleField
        Reference field containing all metadata (mesh_size, box_size, sharding,
        halo_size, etc.) needed for the integration.
    dt0 : float
        Base time step size in scale factor units.
    paint_mode : str, optional
        Painting mode for force computation: "relative" for displacements or
        "absolute" for absolute positions. Default is "relative".
    use_growth : bool, optional
        Whether to use growth factor corrections (True) or simple scale factor
        evolution (False). Default is False.

    Returns
    -------
    drift : callable
        Drift operator: (a, vel, args) -> dpos
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

    Examples
    --------
    >>> drift, kick, first_kick = symplectic_fpm_ode(
    ...     dx_field,
    ...     dt0=0.01,
    ...     paint_mode="relative",
    ...     use_growth=True
    ... )
    >>> # Initialize with first kick
    >>> dvel = first_kick(a=0.1, pos=initial_positions, args=cosmo)
    """
    # Extract metadata from reference field
    mesh_shape = reference_field.mesh_size
    paint_absolute_pos = (paint_mode == "absolute")
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
        cosmo = args[0]
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
            cosmo._workspace = {}
        else:
            drift_contr = (af - ai) / ac

        # Computes the update of position (drift)
        dpos = 1 / (ac**3 * E(cosmo, ac)) * vel.array

        return ParticleField.FromDensityMetadata(
            dpos * (drift_contr / dt0),
            vel,
            status=FieldStatus.PARTICLES,
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
        cosmo = args
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
            cosmo._workspace = {}
        else:
            # Use scale factor to compute the kick factors
            kick_factor_1 = (t1 - t0t1) / t1
            # Second kick control factor
            kick_factor_2 = (t2 - t1t2) / t2

        # Wrap back into ParticleField
        return ParticleField.FromDensityMetadata(
            dvel * ((kick_factor_1 + kick_factor_2) / dt0),
            pos,
            status=FieldStatus.PARTICLES,
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
        cosmo = args
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
        kick_factor = (Gf(cosmo, t0t1) - Gf(cosmo, t0)) / dGfa(cosmo, t0)
        cosmo._workspace = {}

        # Wrap back into ParticleField
        return ParticleField.FromDensityMetadata(
            dvel * (kick_factor / dt0),
            pos,
            status=FieldStatus.PARTICLES,
            scale_factors=a,
        )

    return drift, kick, first_kick
