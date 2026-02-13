from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxpm.growth import E, Gf, dGfa
from jaxpm.growth import growth_factor as Gp
from jaxpm.pm import pm_forces

from ..fields import ParticleField, PositionUnit
from .correction import AbstractCorrection, NoCorrection
from .interp import AbstractInterp, InterpTilerState

__all__ = ["NBodyState", "AbstractNBodySolver", "EfficientDriftDoubleKick", "ReversibleDoubleKickDrift"]


def _forces(pos: ParticleField, cosmo) -> ParticleField:
    """Compute PM forces, returned as a ParticleField."""
    forces_array = (pm_forces(
        pos.array,
        mesh_shape=pos.mesh_size,
        paint_absolute_pos=(pos.unit == PositionUnit.GRID_ABSOLUTE),
        halo_size=pos.halo_size,
        sharding=pos.sharding,
    ) * 1.5 * cosmo.Omega_m)
    return pos.replace(array=forces_array)


class NBodyState(eqx.Module):
    """
    Combined state for N-body integration (auxiliary states).

    Attributes
    ----------
    interp_state : InterpTilerState
        State for the interpolation kernel.
    """

    interp_state: InterpTilerState


class AbstractNBodySolver(eqx.Module):
    """
    Abstract base class for N-body solvers.

    Note: cosmo and dt are passed as method parameters, not stored on the class.
    This allows the solver to be used with different cosmologies and time steps.
    """

    interp_kernel: AbstractInterp
    pgd_kernel: AbstractCorrection = NoCorrection()

    @abstractmethod
    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        ts: jnp.ndarray,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Initialize solver state and perform first kick."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Perform one integration step."""
        raise NotImplementedError

    @abstractmethod
    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        """Extract output at the current step."""
        raise NotImplementedError

    @abstractmethod
    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """Reverse step."""
        raise NotImplementedError


class EfficientDriftDoubleKick(AbstractNBodySolver):
    """
    Efficient symplectic KDK solver (FastPM scheme).
    Not reversible.
    """

    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        ts: jnp.ndarray,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        # Initialize sub-states
        interp_state = self.interp_kernel.init()

        # Physics: First Kick
        t0t1 = (t0 * t1)**0.5
        ai, ac, af = t0, t0, t0t1

        kick_factor = (Gf(cosmo, af) - Gf(cosmo, ai)) / dGfa(cosmo, ac)

        forces = _forces(displacement, cosmo)

        prefactor = 1.0 / (ac**2 * E(cosmo, ac))
        dvel = forces * (prefactor * kick_factor)

        vel_new = (velocities + dvel).replace(scale_factors=jnp.atleast_1d(af))

        return displacement, vel_new, NBodyState(interp_state=interp_state), ts

    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        # t0 -> t1 step.

        # 1. Drift
        # ai = t0, ac = t0t1, af = t1
        t0t1 = (t0 * t1)**0.5
        ai, ac, af = t0, t0t1, t1
        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / dGfa(cosmo, ac)
        prefactor_drift = 1.0 / (ac**3 * E(cosmo, ac))

        dpos = velocities * (prefactor_drift * drift_factor)

        x_drifted = (displacement + dpos).replace(scale_factors=jnp.atleast_1d(af))

        # 2. PGD correction (position-based, full_drift_factor=1.0 since we correct position directly)
        x_corrected, v_boosted = self.pgd_kernel.apply(t1, x_drifted, velocities, full_drift_factor=drift_factor)

        # 3. Double Kick
        t2 = t1 + dt
        t1t2 = (t1 * t2)**0.5
        ac = t1
        ai_k1 = t0t1
        af_k1 = t1
        ai_k2 = t1
        af_k2 = t1t2

        k1 = (Gf(cosmo, af_k1) - Gf(cosmo, ai_k1)) / dGfa(cosmo, ac)
        k2 = (Gf(cosmo, af_k2) - Gf(cosmo, ai_k2)) / dGfa(cosmo, ac)
        kick_factor = k1 + k2

        forces = _forces(x_corrected, cosmo)

        prefactor_kick = 1.0 / (ac**2 * E(cosmo, ac))
        dvel = forces * (prefactor_kick * kick_factor)

        v_new = (v_boosted + dvel).replace(scale_factors=jnp.atleast_1d(af))

        # 4. Advance Interp
        new_interp_state = self.interp_kernel.advance(state.interp_state, t1)

        new_state = NBodyState(interp_state=new_interp_state)

        return x_corrected, v_new, new_state

    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        return self.interp_kernel.paint(state.interp_state, t1, (displacement, velocities), cosmo)

    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        raise NotImplementedError("EfficientDriftDoubleKick is not reversible.")


class ReversibleDoubleKickDrift(AbstractNBodySolver):
    """
    Reversible symplectic KKD solver with PGD correction storage.

    Structure:
    - Init: Kick (v0 -> v0.5) -> Boost (at x0) -> Drift (x0 -> x1)
    - Step: Kick (v0.5 -> v1.5) -> Boost (at x1) -> Drift (x1 -> x2)

    Reversibility:
    - Reverse uses exact Un-Drift -> Un-Boost -> Un-Kick.
    - PGD is applied as a velocity boost before drift, calculated at the *current* position.
    """

    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        ts: jnp.ndarray,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Initialization: First Kick-Drift (KD)
        Moves system from t0 -> t1, with velocity at t0t1.
        """
        # Initialize sub-states
        interp_state = self.interp_kernel.init()

        t0t1 = (t0 * t1)**0.5

        # 1. First Kick: v0 -> v0.5 (t0 -> t0t1)
        ai, ac, af = t0, t0, t0t1

        forces = _forces(displacement, cosmo)

        kick_factor = (Gf(cosmo, af) - Gf(cosmo, ai)) / dGfa(cosmo, ac)
        prefactor_kick = 1.0 / (ac**2 * E(cosmo, ac))

        dvel = forces * (prefactor_kick * kick_factor)

        # v_half is v0.5
        v_half = (velocities + dvel).replace(scale_factors=jnp.atleast_1d(af))

        # 2. PGD Boost (at x0) + Drift (x0 -> x1)
        ai, ac, af = t0, t0t1, t1

        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / dGfa(cosmo, ac)
        prefactor_drift = 1.0 / (ac**3 * E(cosmo, ac))
        full_drift_factor = prefactor_drift * drift_factor

        # Apply PGD Boost.
        # Note: We use t0 for PGD time as it depends on x0
        x_curr, v_boosted = self.pgd_kernel.apply(t0, displacement, v_half, full_drift_factor=full_drift_factor)

        dpos = v_boosted * full_drift_factor

        x_new = (displacement + dpos).replace(scale_factors=jnp.atleast_1d(af))

        # We return v_boosted so that 'velocities' in state carries the boost
        return x_new, v_boosted, NBodyState(interp_state=interp_state), ts - dt

    def step(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Step: Kick-Kick-Drift with PGD Boost
        Input: x1, v0.5_boosted (from prev step)
        """
        t0t1 = (t0 * t1)**0.5
        t2 = t1 + dt
        t1t2 = (t1 * t2)**0.5

        # 1. Forces at x1
        forces = _forces(displacement, cosmo)

        # 2. Double Kick: v(t0t1) -> v(t1t2)
        # Note: input `velocities` is v0.5_boosted.
        # We treat PGD as a kick, so we just add the next kick to it.
        ac = t1
        ai_1, af_1 = t0t1, t1
        k1 = (Gf(cosmo, af_1) - Gf(cosmo, ai_1)) / dGfa(cosmo, ac)

        ai_2, af_2 = t1, t1t2
        k2 = (Gf(cosmo, af_2) - Gf(cosmo, ai_2)) / dGfa(cosmo, ac)

        kick_factor = k1 + k2
        prefactor_kick = 1.0 / (ac**2 * E(cosmo, ac))
        dvel = forces * (prefactor_kick * kick_factor)

        v_new = (velocities + dvel).replace(scale_factors=jnp.atleast_1d(t1t2))

        # 3. PGD Boost (at x1) + Drift (x1 -> x2)
        ai, ac, af = t1, t1t2, t2
        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / dGfa(cosmo, ac)
        prefactor_drift = 1.0 / (ac**3 * E(cosmo, ac))
        full_drift_factor = prefactor_drift * drift_factor

        # Apply PGD Boost at t1 (current time/pos)
        x_curr, v_boosted = self.pgd_kernel.apply(t1, displacement, v_new, full_drift_factor=full_drift_factor)

        dpos = v_boosted * full_drift_factor

        x_new = (displacement + dpos).replace(scale_factors=jnp.atleast_1d(af))

        # 4. Advance Interp
        # Clip t2 to max ts to prevent drifting past end of simulation
        max_ts = self.interp_kernel.ts[-1]
        t2_clipped = jnp.minimum(t2, max_ts)
        new_interp_state = self.interp_kernel.advance(state.interp_state, t2_clipped)
        new_state = NBodyState(interp_state=new_interp_state)

        return x_new, v_boosted, new_state

    def save_at(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> Any:
        return self.interp_kernel.paint(state.interp_state, t1 + dt, (displacement, velocities), cosmo)

    def reverse(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        state: NBodyState,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Reverse Step: Un-Drift -> Un-Boost -> Un-Kick
        Input: x2, v1.5_boosted
        """
        t0t1 = (t0 * t1)**0.5
        t2 = t1 + dt
        t1t2 = (t1 * t2)**0.5

        # 1. Un-Drift (x2 -> x1) using v1.5_boosted
        ai, ac, af = t1, t1t2, t2
        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / dGfa(cosmo, ac)
        prefactor_drift = 1.0 / (ac**3 * E(cosmo, ac))
        full_drift_factor = prefactor_drift * drift_factor

        dpos = velocities * full_drift_factor

        # displacement here is x2
        x_old = (displacement - dpos).replace(scale_factors=jnp.atleast_1d(t1))

        # 2. Un-Boost (at x1)
        # Recompute boost at recovered x1
        vel_zero = velocities * 0.0
        # Note: pgd.apply returns (x, v_boosted). We need just the velocity boost part.
        x_uncorrected, v_uncorrected = self.pgd_kernel.rewind(t1, x_old, vel_zero, full_drift_factor=drift_factor)

        # 3. Un-Kick (v1.5 -> v0.5)
        # Recompute forces at recovered position x1
        forces = _forces(x_uncorrected, cosmo)

        ac = t1
        ai_1, af_1 = t0t1, t1
        k1 = (Gf(cosmo, af_1) - Gf(cosmo, ai_1)) / dGfa(cosmo, ac)
        ai_2, af_2 = t1, t1t2
        k2 = (Gf(cosmo, af_2) - Gf(cosmo, ai_2)) / dGfa(cosmo, ac)

        kick_factor = k1 + k2
        prefactor_kick = 1.0 / (ac**2 * E(cosmo, ac))
        dvel = forces * (prefactor_kick * kick_factor)

        v_old = (v_uncorrected - dvel).replace(scale_factors=jnp.atleast_1d(t0t1))

        # 4. Rewind Interp State
        max_ts = self.interp_kernel.ts[-1]
        t2_clipped = jnp.minimum(t2, max_ts)
        prev_interp_state = self.interp_kernel.rewind(state.interp_state, t2_clipped)
        prev_state = NBodyState(interp_state=prev_interp_state)

        return x_old, v_old, prev_state
