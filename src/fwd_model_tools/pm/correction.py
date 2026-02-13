"""
Correction kernels for PM N-body simulations.

Includes:
- PGDKernel: Position-based correction (non-reversible)
- SharpeningKernel: Velocity-based correction (reversible)

Based on https://arxiv.org/abs/1804.00671
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.kernels import fftk, gradient_kernel, invlaplace_kernel

from ..fields import ParticleField, PositionUnit

__all__ = ["PGDKernel", "SharpeningKernel", "NoCorrection", "AbstractCorrection"]


class AbstractCorrection(eqx.Module):
    """
    Abstract base class for correction kernels.
    """

    @property
    @abstractmethod
    def reversible(self) -> bool:
        """Whether this kernel is compatible with reverse adjoint."""
        pass

    @abstractmethod
    def apply(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Apply correction to particle state.

        Parameters
        ----------
        t : float
            Current time.
        pos : ParticleField
            Particle positions/displacements.
        vel : ParticleField
            Particle velocities.
        full_drift_factor : float
            The complete drift factor (prefactor * drift_factor) for this step.
            Used by SharpeningKernel to convert displacement to velocity.
            For PGDKernel this parameter is ignored.

        Returns
        -------
        Tuple[ParticleField, ParticleField]
            (Corrected positions, Corrected velocities).
        """
        pass

    @abstractmethod
    def rewind(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Rewind correction from particle state.

        Parameters
        ----------
        t : float
            Current time.
        pos : ParticleField
            Particle positions/displacements.
        vel : ParticleField
            Particle velocities (already corrected/boosted).
        full_drift_factor : float
            The complete drift factor (prefactor * drift_factor) for this step.

        Returns
        -------
        Tuple[ParticleField, ParticleField]
            (Uncorrected positions, Uncorrected velocities).
        """
        pass


class NoCorrection(AbstractCorrection):
    """
    No-op correction kernel.
    """

    @property
    def reversible(self) -> bool:
        return True

    def apply(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Return unchanged positions and velocities.
        """
        return pos, vel

    def rewind(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Return unchanged positions and velocities.
        """
        return pos, vel


class PGDKernel(AbstractCorrection):
    """
    PGD correction kernel with trainable parameters.

    This kernel applies correction directly to positions AFTER drift.
    It is NOT reversible because reversing requires knowing x_drifted
    to compute S(x_drifted), but x_drifted is the unknown we're solving for.

    Parameters
    ----------
    alpha : float
        Scaling factor.
    kl : float
        Long-range scale parameter.
    ks : float
        Short-range scale parameter.
    """

    alpha: float = 0.2
    kl: float = 0.3
    ks: float = 1.0

    @property
    def reversible(self) -> bool:
        return False  # Cannot be used with adjoint='reverse'

    def apply(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Apply PGD correction to positions.

        Note: full_drift_factor is ignored for PGDKernel since we
        correct positions directly rather than via velocity boost.
        """
        # Paint particles to density
        delta = pos.paint()

        # FFT to Fourier space
        delta_k = delta.apply_fn(fft3d)

        # Get wave vectors
        kvec = fftk(delta_k.array)

        # Inline PGD Kernel Computation
        kk = sum(ki**2 for ki in kvec)
        kl2 = self.kl**2
        ks4 = self.ks**4
        # Avoid division by zero at k=0
        kk_safe = jnp.where(kk == 0, 1, kk)
        pgd_range = jnp.exp(-kl2 / kk_safe) * jnp.exp(-(kk**2) / ks4)
        # Zero out the k=0 mode
        pgd_range = jnp.where(kk == 0, 0, pgd_range)

        pot_k_pgd = delta_k.array * invlaplace_kernel(kvec) * pgd_range

        # Compute PGD forces via gradient of potential
        forces_pgd = jnp.stack(
            [pos.read_out(delta.replace(array=ifft3d(-gradient_kernel(kvec, i) * pot_k_pgd))).array for i in range(3)],
            axis=-1,
        )

        # Calculate displacement correction
        dpos_pgd_array = forces_pgd * self.alpha
        dpos_pgd = pos.replace(array=dpos_pgd_array, unit=PositionUnit.GRID_RELATIVE)

        # Return corrected position, velocity unchanged
        return pos + dpos_pgd, vel

    def rewind(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        raise NotImplementedError("PGDKernel is not reversible.")


class SharpeningKernel(AbstractCorrection):
    """
    Reversible symplectic sharpening kernel.

    Applies PGD-style displacement as a velocity boost BEFORE drift.
    Based on Chin (1997) Force-Gradient method.

    The displacement S(x) computed at the known position x is converted
    to a velocity: v_pgd = S(x) / full_drift_factor, then added to velocity.
    During drift, this effectively applies:
        x_new = x + (v + v_pgd) * full_drift_factor
              = x + v*drift + S(x)

    Reversibility: In reverse(), the position is recovered by un-drifting,
    then S(x) is recomputed at the recovered position and subtracted from velocity.
    No state storage needed.

    Parameters
    ----------
    alpha : float
        Scaling factor.
    kl : float
        Long-range scale parameter.
    ks : float
        Short-range scale parameter.
    """

    alpha: float = 0.2
    kl: float = 0.3
    ks: float = 1.0

    @property
    def reversible(self) -> bool:
        return True

    def apply(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Apply sharpening as velocity boost.

        Forward:
            1. Compute S(x) displacement at current position
            2. Convert to velocity: v_pgd = S / full_drift_factor
            3. Return (pos_unchanged, vel + v_pgd)
        """
        v_pgd = self._compute_boost(pos, vel, full_drift_factor)

        # Boost velocity (position unchanged)
        vel_boosted = vel + v_pgd

        return pos, vel_boosted

    def rewind(
        self,
        t: float,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> tuple[ParticleField, ParticleField]:
        """
        Rewind sharpening boost.

        Reverse:
            1. Compute S(x) displacement at current position
            2. Convert to velocity: v_pgd = S / full_drift_factor
            3. Return (pos_unchanged, vel - v_pgd)
        """
        v_pgd = self._compute_boost(pos, vel, full_drift_factor)

        # Un-Boost velocity
        vel_pure = vel - v_pgd

        return pos, vel_pure

    def _compute_boost(
        self,
        pos: ParticleField,
        vel: ParticleField,
        full_drift_factor: float,
    ) -> ParticleField:
        """Internal helper to compute velocity boost."""
        # Paint particles to density
        delta = pos.paint()

        # FFT to Fourier space
        delta_k = delta.apply_fn(fft3d)

        # Get wave vectors
        kvec = fftk(delta_k.array)

        # PGD kernel in Fourier space
        kk = sum(ki**2 for ki in kvec)
        kl2 = self.kl**2
        ks4 = self.ks**4
        # Avoid division by zero at k=0
        kk_safe = jnp.where(kk == 0, 1, kk)
        pgd_range = jnp.exp(-kl2 / kk_safe) * jnp.exp(-(kk**2) / ks4)
        # Zero out the k=0 mode
        pgd_range = jnp.where(kk == 0, 0, pgd_range)

        pot_k_pgd = delta_k.array * invlaplace_kernel(kvec) * pgd_range

        # Compute displacement correction S(x) in grid units
        forces_pgd = jnp.stack(
            [pos.read_out(delta.replace(array=ifft3d(-gradient_kernel(kvec, i) * pot_k_pgd))).array for i in range(3)],
            axis=-1,
        )

        S_array = forces_pgd * self.alpha

        # Convert displacement to velocity boost
        # v_pgd = S / full_drift_factor
        v_pgd_array = S_array / full_drift_factor
        v_pgd = vel.replace(array=v_pgd_array)
        return v_pgd
