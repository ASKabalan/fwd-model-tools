from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxpm.growth import E, Gf, dGfa, gp
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
    t_initial : float
        Initial time of the simulation (used for clipping kicks).
    """

    interp_state: InterpTilerState
    t_initial: float


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

        return displacement, vel_new, NBodyState(interp_state=interp_state, t_initial=t0)

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
        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / gp(cosmo, ac)
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

        new_state = NBodyState(interp_state=new_interp_state, t_initial=state.t_initial)

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
    - Step: Kick (prev_mid -> current) + Kick (current -> next_mid) -> PGD Boost -> Drift
    - Init: No-op physically, effectively sets up state such that the first 'prev' kick is zero.

    This ensures reversibility and correct time synchronization with the integrator loop.
    """

    def init(
        self,
        displacement: ParticleField,
        velocities: ParticleField,
        t0: float,
        t1: float,
        dt: float,
        cosmo: Any,
    ) -> tuple[ParticleField, ParticleField, NBodyState]:
        """
        Initialization: No physical update.
        Sets t_initial in state to handle boundary conditions for kicks.
        """
        interp_state = self.interp_kernel.init()
        # Warm start the growth factor to ensure any JIT compilation happens here.
        #_ = Gf(cosmo, t0)
        # No physical evolution here.
        return displacement, velocities, NBodyState(interp_state=interp_state, t_initial=t0)

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
        Step: Forces -> Double Kick -> PGD Boost -> Drift
        """
        # Times
        # t0 is current time (x_i), t1 is next time (x_{i+1})
        # We need to construct the previous interval midpoint for the kick.
        # We assume dt was constant or use t0-dt.
        # t_prev = t0 - dt. Midpoint is geometric mean (t_prev * t0)^0.5
        t_prev = t0 - dt
        t_prev_clamped = jnp.maximum(t_prev, state.t_initial)

        # Midpoints
        ac = t0  # Force evaluation time

        # Previous half-step (K1)
        ai_1 = (t_prev_clamped * t0)**0.5
        af_1 = t0

        # Next half-step (K2)
        ai_2 = t0
        af_2 = (t0 * t1)**0.5

        # 1. Forces at current position (x_i at t0)
        forces = _forces(displacement, cosmo)

        # 2. Compute Kick Factor
        # Note: If t0 == t_initial, then t_prev_clamped = t0, ai_1 = t0.
        # So K1 term Gf(t0)-Gf(t0) becomes 0.
        k1 = (Gf(cosmo, af_1) - Gf(cosmo, ai_1)) / dGfa(cosmo, ac)
        k2 = (Gf(cosmo, af_2) - Gf(cosmo, ai_2)) / dGfa(cosmo, ac)

        kick_factor = k1 + k2
        prefactor_kick = 1.0 / (ac**2 * E(cosmo, ac))

        dvel = forces * (prefactor_kick * kick_factor)

        # v_mid: velocity updated to the middle of the drift interval (af_2)
        v_mid = (velocities + dvel).replace(scale_factors=jnp.atleast_1d(af_2))

        # 3. PGD Boost (at x_i) + Drift (x_i -> x_{i+1})
        ai, ac_drift, af = t0, af_2, t1
        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / gp(cosmo, ac_drift)
        prefactor_drift = 1.0 / (ac_drift**3 * E(cosmo, ac_drift))
        full_drift_factor = prefactor_drift * drift_factor

        # Apply PGD Boost at t0
        x_curr, v_boosted = self.pgd_kernel.apply(t0, displacement, v_mid, full_drift_factor=full_drift_factor)

        dpos = v_boosted * full_drift_factor

        x_new = (displacement + dpos).replace(scale_factors=jnp.atleast_1d(af))

        # 4. Advance Interp
        new_interp_state = self.interp_kernel.advance(state.interp_state, t1)

        new_state = NBodyState(interp_state=new_interp_state, t_initial=state.t_initial)

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
        # In this scheme, t1 is the current time of the particles
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
        """
        Reverse Step: Un-Drift -> Un-Boost -> Un-Kick
        Input: x_{i+1} (t1), v_{mid_boosted}
        Target: x_i (t0), v_{old}
        """
        # Reconstruct drift parameters
        # t0 is prev time (target), t1 is current time (source)
        # Note: in reverse, t0 and t1 are swapped relative to forward sense in integration loop?
        # integrate.py reverse loop passes: t_prev, tc.
        # t_prev becomes t0 (target), tc becomes t1 (source).

        # Drift calc
        t0t1 = (t0 * t1)**0.5
        ai, ac_drift, af = t0, t0t1, t1

        drift_factor = (Gp(cosmo, af) - Gp(cosmo, ai)) / dGfa(cosmo, ac_drift)
        prefactor_drift = 1.0 / (ac_drift**3 * E(cosmo, ac_drift))
        full_drift_factor = prefactor_drift * drift_factor

        # 1. Un-Drift (x_{i+1} -> x_i) using boosted velocity
        dpos = velocities * full_drift_factor
        x_old = (displacement - dpos).replace(scale_factors=jnp.atleast_1d(t0))

        # 2. Un-Boost (at x_i)
        # Recover unboosted velocity. Note: rewinding at t0 (target time)
        x_uncorrected, v_uncorrected = self.pgd_kernel.rewind(t0,
                                                              x_old,
                                                              velocities,
                                                              full_drift_factor=full_drift_factor)

        # 3. Un-Kick
        # Need to reconstruct kick factors identical to forward pass
        t_prev = t0 - dt
        t_prev_clamped = jnp.maximum(t_prev, state.t_initial)

        # Previous half-step (K1)
        ai_1 = (t_prev_clamped * t0)**0.5
        af_1 = t0

        # Next half-step (K2)
        ai_2 = t0
        af_2 = t0t1  # (t0 * t1)**0.5

        ac_kick = t0

        k1 = (Gf(cosmo, af_1) - Gf(cosmo, ai_1)) / dGfa(cosmo, ac_kick)
        k2 = (Gf(cosmo, af_2) - Gf(cosmo, ai_2)) / dGfa(cosmo, ac_kick)

        kick_factor = k1 + k2
        prefactor_kick = 1.0 / (ac_kick**2 * E(cosmo, ac_kick))

        forces = _forces(x_uncorrected, cosmo)
        dvel = forces * (prefactor_kick * kick_factor)

        v_old = (v_uncorrected - dvel)  # Time unit isn't critical for v return in reverse

        # 4. Rewind Interp State
        max_ts = self.interp_kernel.ts[-1]
        t1_clipped = jnp.minimum(t1, max_ts)
        prev_interp_state = self.interp_kernel.rewind(state.interp_state, t1_clipped)
        prev_state = NBodyState(interp_state=prev_interp_state, t_initial=state.t_initial)

        return x_old, v_old, prev_state
