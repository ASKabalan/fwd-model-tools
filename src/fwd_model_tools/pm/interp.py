from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_healpy as jhp
from jaxpm.growth import E, dGfa
from jaxpm.growth import growth_factor as Gp

from .._src.base._warn import warning_if
from ..fields import (
    AbstractField,
    DensityField,
    DensityUnit,
    FieldStatus,
    FlatDensity,
    ParticleField,
    PositionUnit,
    SphericalDensity,
)
from ..fields.painting import PaintingOptions

__all__ = ["InterpTilerState", "OnionTiler", "TelephotoInterp", "NoInterp", "get_all_47_symmetries", "AbstractInterp"]


def get_all_47_symmetries() -> jnp.ndarray:
    """Generate all 48 cubic symmetry matrices."""
    permutations = list(itertools.permutations([0, 1, 2]))
    signs = list(itertools.product([1, -1], repeat=3))
    matrices = []
    for s in signs:
        for p in permutations:
            M = jnp.zeros((3, 3), dtype=jnp.int16)
            M = M.at[0, p[0]].set(s[0])
            M = M.at[1, p[1]].set(s[1])
            M = M.at[2, p[2]].set(s[2])
            matrices.append(M)
    return jnp.array(matrices)[1:]  # Exclude identity


class InterpTilerState(eqx.Module):
    """
    Dynamic state for interpolation kernels.
    """
    rotation_idx: int = -1
    shell_idx: int = -1
    # Rotations are generated at init and stored here
    rotations: jnp.ndarray | None = None


class AbstractInterp(eqx.Module):
    """
    Abstract base class for interpolation kernels (Executor).
    Holds configuration (Painting, Geometry).
    """
    painting: PaintingOptions
    ts: Optional[jnp.ndarray] = None
    r_centers: Optional[jnp.ndarray] = None
    density_widths: Optional[jnp.ndarray] = None
    max_comoving_distance: Optional[float] = None

    def update_geometry(
        self,
        ts: jnp.ndarray,
        r_centers: jnp.ndarray,
        density_widths: jnp.ndarray,
        max_comoving_distance: float,
    ) -> AbstractInterp:
        """Update geometry configuration."""
        return eqx.tree_at(lambda s: (s.ts, s.r_centers, s.density_widths, s.max_comoving_distance), self,
                           (ts, r_centers, density_widths, max_comoving_distance))

    @abstractmethod
    def init(self) -> InterpTilerState:
        """Initialize dynamic state."""
        raise NotImplementedError

    @abstractmethod
    def advance(self, state: InterpTilerState) -> InterpTilerState:
        """Advance the interpolation state to the next shell."""
        raise NotImplementedError

    @abstractmethod
    def rewind(self, state: InterpTilerState) -> InterpTilerState:
        """Rewind the interpolation state to the previous shell (inverse of advance)."""
        raise NotImplementedError

    @abstractmethod
    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
    ) -> Any:
        """Produce a map."""
        raise NotImplementedError

    def _drift_to_time(
        self,
        positions: jnp.ndarray,
        velocities: jnp.ndarray,
        a_current: float,
        a_target: float,
        cosmo,
    ) -> jnp.ndarray:
        """Drift particles using growth factor form."""
        ac = (a_current * a_target)**0.5
        drift_factor = (Gp(cosmo, a_target) - Gp(cosmo, a_current)) / dGfa(cosmo, ac)
        return positions + drift_factor * velocities


class NoInterp(AbstractInterp):
    """Simple painting without tiling."""

    def init(self) -> InterpTilerState:
        # Check geometry
        if self.r_centers is not None:
            furthest_shell = self.r_centers[0] + self.density_widths[0] / 2
            # Warning logic here if needed, but warnings in JIT are tricky.
            # Assuming update_geometry handles checks or we assume valid.

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=None,
        )

    def advance(self, state: InterpTilerState) -> InterpTilerState:
        return eqx.tree_at(lambda s: s.shell_idx, state, state.shell_idx + 1)

    def rewind(self, state: InterpTilerState) -> InterpTilerState:
        return eqx.tree_at(lambda s: s.shell_idx, state, state.shell_idx - 1)

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
    ) -> Any:
        dx, p = y

        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]

        if self.painting is None or self.painting.target == "particles":
            result = dx
        elif self.painting.target == "spherical":
            result = dx.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        elif self.painting.target == "flat":
            result = dx.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )
        elif self.painting.target == "density":
            result = dx.paint(
                weights=self.painting.weights,
                chunk_size=self.painting.chunk_size,
                batch_size=self.painting.batch_size,
            )
        else:
            result = dx

        result = result.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return result


class OnionTiler(AbstractInterp):
    """27-tile spherical painting with rotation decorrelation."""

    def init(self) -> InterpTilerState:
        if self.painting.target != "spherical":
            raise ValueError("OnionTiler only supports spherical painting.")

        # Generate rotations
        all_symmetries = get_all_47_symmetries()
        rng = jax.random.PRNGKey(42)
        rot_indices = jax.random.randint(rng, shape=(27, ), minval=0, maxval=47)
        selected_rotations = all_symmetries[rot_indices]

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=selected_rotations,
        )

    def advance(self, state: InterpTilerState) -> InterpTilerState:
        return eqx.tree_at(lambda s: s.shell_idx, state, state.shell_idx + 1)

    def rewind(self, state: InterpTilerState) -> InterpTilerState:
        return eqx.tree_at(lambda s: s.shell_idx, state, state.shell_idx - 1)

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
    ) -> Any:
        dx, p = y
        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]

        R_min = r_center - width / 2
        R_max = r_center + width / 2

        inside_box = (r_center + width / 2) <= self.max_comoving_distance

        def inside_branch(current_state):
            return self._paint_single(t, p, dx, cosmo, r_center, width, current_state)

        def outside_branch(current_state):
            return self._paint_tiled_27(t, p, dx, cosmo, r_center, R_min, R_max, current_state)

        sph_map = jax.lax.cond(inside_box, inside_branch, outside_branch, state)
        sph_map = sph_map.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return sph_map

    def _paint_single(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        R_center: float,
        width: float,
        state: InterpTilerState,
    ) -> SphericalDensity:
        sph_map = dx.paint_spherical(
            center=R_center,
            density_plane_width=width,
            scheme=self.painting.scheme,
            weights=self.painting.weights,
            kernel_width_arcmin=self.painting.kernel_width_arcmin,
            smoothing_interpretation=self.painting.smoothing_interpretation,
            paint_nside=self.painting.paint_nside,
            ud_grade_power=self.painting.ud_grade_power,
            ud_grade_order_in=self.painting.ud_grade_order_in,
            ud_grade_order_out=self.painting.ud_grade_order_out,
            ud_grade_pess=self.painting.ud_grade_pess,
            batch_size=self.painting.batch_size,
        )
        return sph_map.replace(status=FieldStatus.LIGHTCONE,
                               unit=DensityUnit.DENSITY,
                               scale_factors=t,
                               comoving_centers=R_center,
                               density_width=width)

    def _paint_tiled_27(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        r_center: float,
        R_min: float,
        R_max: float,
        state: InterpTilerState,
    ) -> SphericalDensity:
        nside = dx.nside
        observer_mpc = jnp.array(dx.observer_position_mpc)
        observer_relative = jnp.array(dx.observer_position)
        box_size = jnp.array(dx.box_size)
        width = R_max - R_min

        positions = dx.to(PositionUnit.MPC_H).array
        positions_centered = positions - observer_mpc
        velocities = p.array

        shifts = jnp.array(list(itertools.product([0, 1, -1], repeat=3)), dtype=jnp.int16)
        shifts = jnp.where(observer_relative == 0, shifts + 1, shifts)
        shifts = jnp.where(observer_relative == 1, shifts - 1, shifts)
        shifts *= box_size

        rot_matrices = state.rotations

        npix = jhp.nside2npix(nside)
        init_map = SphericalDensity.FromDensityMetadata(
            array=jnp.zeros(npix),
            field=dx,
            unit=DensityUnit.DENSITY,
        )

        def scan_body(accum_map, tile_data):
            shift_idx, rot_mat = tile_data
            shift_vec = shift_idx.astype(jnp.float32)

            p_rotated = jnp.einsum('ij,...j->...i', rot_mat, positions_centered)
            p_tiled = p_rotated + shift_vec

            p_final = p_tiled
            p_final = p_final + observer_mpc
            p_final = ParticleField.FromDensityMetadata(
                array=p_final,
                field=dx,
                unit=PositionUnit.MPC_H,
            )

            tile_map = p_final.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
            return accum_map + tile_map, None

        final_map, _ = jax.lax.scan(scan_body, init_map, (shifts, rot_matrices))
        final_map = final_map.replace(status=FieldStatus.LIGHTCONE,
                                      unit=DensityUnit.DENSITY,
                                      scale_factors=t,
                                      comoving_centers=r_center,
                                      density_width=width)
        return final_map


class TelephotoInterp(AbstractInterp):
    """Single rotation+shift painting for narrow FOV."""

    def init(self) -> InterpTilerState:
        if self.painting.target != "spherical" and self.painting.target != "flat":
            raise ValueError("TelephotoInterp only supports projections (flat or spherical).")

        if self.painting.target == "spherical":
            warnings.warn("""
            TelephotoInterp is designed for narrow FOV flat-sky projections.
            Using it for spherical painting will cause significant artifacts.
            use for illustriative purposes only.
            """)

        # Check logic moved to update_geometry/advance or assumed checked

        return InterpTilerState(
            rotation_idx=-1,
            shell_idx=-1,
            rotations=get_all_47_symmetries(),
        )

    def advance(self, state: InterpTilerState) -> InterpTilerState:
        # 1. Update shell index to next
        next_shell_idx = state.shell_idx + 1

        # 2. Check geometry of this NEW shell
        r_center = self.r_centers[next_shell_idx]
        width = self.density_widths[next_shell_idx]
        inside_box = (r_center + width / 2) <= self.max_comoving_distance

        # 3. Update rotation index
        next_rot_idx = jnp.where(
            inside_box,
            state.rotation_idx,  # Keep current
            state.rotation_idx + 1  # Increment
        )

        return eqx.tree_at(lambda s: (s.shell_idx, s.rotation_idx), state, (next_shell_idx, next_rot_idx))

    def rewind(self, state: InterpTilerState) -> InterpTilerState:
        # Check if CURRENT shell is outside box (meaning we incremented rot_idx when entering)
        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]
        current_outside = (r_center + width / 2) > self.max_comoving_distance

        prev_rot_idx = jnp.where(current_outside, state.rotation_idx - 1, state.rotation_idx)
        prev_shell_idx = state.shell_idx - 1

        return eqx.tree_at(lambda s: (s.shell_idx, s.rotation_idx), state, (prev_shell_idx, prev_rot_idx))

    def paint(
        self,
        state: InterpTilerState,
        t: float,
        y: tuple[ParticleField, ParticleField],
        cosmo,
    ) -> Any:
        dx, p = y

        r_center = self.r_centers[state.shell_idx]
        width = self.density_widths[state.shell_idx]
        inside_box = (r_center + width / 2) <= self.max_comoving_distance

        def inside_branch(current_state):
            return self._paint_single(t, p, dx, cosmo, r_center, width, current_state)

        def outside_branch(current_state):
            return self._paint_telephoto(t, p, dx, cosmo, r_center, width, current_state)

        sph_map = jax.lax.cond(inside_box, inside_branch, outside_branch, state)

        sph_map = sph_map.replace(scale_factors=t, comoving_centers=r_center, density_width=width)
        return sph_map

    def _paint_single(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        R_center: float,
        width: float,
        state: InterpTilerState,
    ) -> SphericalDensity:
        if self.painting.target == "spherical":
            return dx.paint_spherical(
                center=R_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        else:  # flat
            return dx.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )

    def _paint_telephoto(
        self,
        t: float,
        p: ParticleField,
        dx: ParticleField,
        cosmo,
        r_center: float,
        width: float,
        state: InterpTilerState,
    ) -> SphericalDensity:
        observer_mpc = jnp.array(dx.observer_position_mpc)

        rot_idx = state.rotation_idx % 47
        M = state.rotations[rot_idx]

        shift_distance = jnp.floor(r_center / self.max_comoving_distance) * self.max_comoving_distance

        positions = dx.to(PositionUnit.MPC_H).array
        velocities = p.array
        box_size = jnp.array(dx.box_size)

        p_centered = positions - observer_mpc
        p_rotated = jnp.einsum('ij,...j->...i', M, p_centered)
        z_axis = jnp.array([0., 0., 1.])
        p_shifted = p_rotated + z_axis * shift_distance
        v_rotated = jnp.einsum('ij,...j->...i', M, velocities)

        a_current = t
        dist_mpc = jnp.linalg.norm(p_shifted - observer_mpc, axis=-1)
        a_target_particle = jc.background.a_of_chi(cosmo, dist_mpc)[..., None]
        p_drifted = self._drift_to_time(p_shifted, v_rotated, a_current, a_target_particle, cosmo)

        dx_transformed = ParticleField.FromDensityMetadata(
            array=p_drifted + observer_mpc,
            field=dx,
            unit=PositionUnit.MPC_H,
        )

        if self.painting.target == "spherical":
            return dx_transformed.paint_spherical(
                center=r_center,
                density_plane_width=width,
                scheme=self.painting.scheme,
                weights=self.painting.weights,
                kernel_width_arcmin=self.painting.kernel_width_arcmin,
                smoothing_interpretation=self.painting.smoothing_interpretation,
                paint_nside=self.painting.paint_nside,
                ud_grade_power=self.painting.ud_grade_power,
                ud_grade_order_in=self.painting.ud_grade_order_in,
                ud_grade_order_out=self.painting.ud_grade_order_out,
                ud_grade_pess=self.painting.ud_grade_pess,
                batch_size=self.painting.batch_size,
            )
        else:  # flat
            return dx_transformed.paint_2d(
                center=r_center,
                density_plane_width=width,
                weights=self.painting.weights,
                batch_size=self.painting.batch_size,
            )
