from __future__ import annotations

from functools import partial
from typing import Any, Optional, Sequence, Tuple, Literal

import jax
import jax.numpy as jnp
from jaxpm.painting import cic_paint, cic_paint_dx, cic_read, cic_read_dx
from jaxpm.spherical import paint_particles_spherical
from jaxtyping import Array , Float

from fwd_model_tools._src._painting import _single_paint, _single_paint_2d, _single_paint_spherical

from .density import  DensityField, DensityStatus, FieldStatus

DEFAULT_CHUNK_SIZE = 2**24
PaintMode = Literal["relative", "absolute"]
SphericalScheme = Literal["ngp", "bilinear", "rbf_neighbor"]


@jax.tree_util.register_pytree_node_class
class ParticleField(DensityField):
    """
    Field subclass representing particle positions or displacements.

    Painting helpers live here to ensure only particle-based data can be
    rasterized into density grids.
    """

    def __init__(
        self,
        *,
        array: Array,
        mesh_size: Tuple[int, int, int],
        box_size: Tuple[float, float, float],
        observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sharding: Optional[Any] = None,
        nside: Optional[int] = None,
        flatsky_npix: Optional[Tuple[int, int]] = None,
        field_size: Optional[float] = None,
        halo_size: int | Tuple[int, int] = 0,
        z_source: Optional[Any] = None,
        status: FieldStatus = FieldStatus.RAW,
        scale_factors: float = 1.0,
    ):
        if isinstance(array, Array):
            if not ((array.ndim == 4 and array.shape[-1] == 3) or (array.ndim == 5 and array.shape[-1] == 3)):
                raise ValueError("ParticleField array must have shape (X, Y, Z, 3) or (N, X, Y, Z, 3)")

        super().__init__(
            array=array,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            sharding=sharding,
            nside=nside,
            flatsky_npix=flatsky_npix,
            field_size=field_size,
            halo_size=halo_size,
            z_source=z_source,
            status=status,
            scale_factors=scale_factors,
        )

    def __getitem__(self, key) -> "ParticleField":
        """
        Index into batched ParticleField.
        """
        if self.array.ndim != 5:
            raise ValueError("Indexing only supported for batched ParticleField (5D array)")
        return super().__getitem__(key)

    @partial(jax.jit, static_argnames=['mode', 'weights', 'chunk_size', 'batch_size'])
    def paint(
        self,
        mode: PaintMode = "relative",
        *,
        mesh: Optional[Array] = None,
        weights: Array | float = 1.0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_size: Optional[int] = None,
    ) -> DensityField:
        """
        Paint particles onto a 3D density mesh using CIC interpolation.
        """
        data = jnp.asarray(self.array)

        paint_fn = jax.tree_util.Partial(
            _single_paint,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            halo_size=self.halo_size,
            mode=mode,
            mesh=mesh,
            weights=weights,
            chunk_size=chunk_size,
        )

        if data.ndim == 4:
            data = data[None, ...]
        elif data.ndim != 5:
            raise ValueError(f"paint() expects 4D or 5D array, got shape {data.shape}")

        painted = jax.lax.map(paint_fn, data, batch_size=batch_size)
        painted = painted.squeeze()

        return DensityField(
            array=painted,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            halo_size=self.halo_size,
            status=FieldStatus.DENSITY_FIELD,
            scale_factors=self.scale_factors,
        )

    @partial(jax.jit, static_argnames=['mode'])
    def read_out(
        self,
        density_mesh: DensityField | Array,
        *,
        mode: PaintMode = "relative",
    ) -> "ParticleField":
        """
        Interpolate displacements/positions back from a 3D density mesh.
        """
        mode = mode.lower()
        mesh = density_mesh.array if isinstance(density_mesh, DensityField) else density_mesh
        if mode == "relative":
            read_data = cic_read_dx(
                mesh,
                self.array,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )
        elif mode == "absolute":
            read_data = cic_read(
                mesh,
                self.array,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )
        else:
            raise ValueError("mode must be either 'relative' or 'absolute'")

        return DensityField(
            array=read_data,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            halo_size=self.halo_size,
            status=self.status,
            scale_factors=self.scale_factors,
        )

    @partial(jax.jit, static_argnames=['mode', 'weights', 'density_plane_width', 'batch_size'])
    def paint_2d(
        self,
        center: Float | Array,
        *,
        density_plane_width: Optional[Float] = None,
        weights: Optional[Array | float] = None,
        mode: PaintMode = "relative",
        batch_size: Optional[int] = None,
    ) -> "DensityField":
        """
        Project particles onto a flat-sky grid using CIC painting.
        """
        if self.flatsky_npix is None:
            raise ValueError("ParticleField requires `flatsky_npix` for paint_2d.")

        data = jnp.asarray(self.array)
        center_arr = jnp.atleast_1d(center)

        if data.ndim == 5:
            nb_shells = data.shape[0]
            if center_arr.size != nb_shells:
                raise ValueError(f"Batched input: center must have {nb_shells} elements, got {center_arr.size}")
            density_plane_width = self.density_width(nb_shells) if density_plane_width is None else density_plane_width
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            if density_plane_width is None:
                raise ValueError("density_plane_width must be specified for single shell")
        else:
            raise ValueError(f"paint_2d() expects 4D or 5D array, got shape {data.shape}")

        paint_fn = jax.tree_util.Partial(
            _single_paint_2d,
            mesh_size=self.mesh_size,
            flatsky_npix=self.flatsky_npix,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            halo_size=self.halo_size,
            mode=mode,
            weights=weights,
            density_plane_width=density_plane_width,
            max_comoving_radius=self.max_comoving_radius,
        )

        painted = jax.lax.map(paint_fn, (data, center_arr), batch_size=batch_size)
        painted = painted.squeeze()

        from .lightcone import FlatDensity

        return FlatDensity.FromDensityMetadata(
            array=painted,
            density_field=self,
            status=DensityStatus.LIGHTCONE,
        )

    @partial(jax.jit,
             static_argnames=[
                 'mode', 'scheme', 'weights', 'density_plane_width', 'kernel_width_arcmin', 'smoothing_interpretation',
                 'paint_nside', 'ud_grade_power', 'ud_grade_order_in', 'ud_grade_order_out', 'ud_grade_pess',
                 'batch_size'
             ])
    def paint_spherical(
        self,
        center: Float | Array,
        *,
        mode: PaintMode = "relative",
        scheme: SphericalScheme = "rbf_neighbor",
        weights: Optional[Array] = None,
        density_plane_width: Optional[Float] = None,
        kernel_width_arcmin: Optional[float] = None,
        smoothing_interpretation: str = "fwhm",
        paint_nside: Optional[int] = None,
        ud_grade_power: float = 0.0,
        ud_grade_order_in: str = "RING",
        ud_grade_order_out: str = "RING",
        ud_grade_pess: bool = False,
        batch_size: Optional[int] = None,
    ) -> "DensityField":
        """
        Paint particles onto a HEALPix grid using spherical painting.
        """
        if self.nside is None:
            raise ValueError("Spherical painting requires `nside`.")

        data = jnp.asarray(self.array)
        center_arr = jnp.atleast_1d(center)

        if data.ndim == 5:
            nb_shells = data.shape[0]
            if center_arr.size != nb_shells:
                raise ValueError(f"Batched input: center must have {nb_shells} elements, got {center_arr.size}")
            density_plane_width = self.density_width(nb_shells) if density_plane_width is None else density_plane_width
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            if density_plane_width is None:
                raise ValueError("density_plane_width must be specified for single shell")
        else:
            raise ValueError(f"paint_spherical() expects 4D or 5D array, got shape {data.shape}")

        paint_fn = jax.tree_util.Partial(
            _single_paint_spherical,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            halo_size=self.halo_size,
            mode=mode,
            scheme=scheme,
            weights=weights,
            density_plane_width=density_plane_width,
            kernel_width_arcmin=kernel_width_arcmin,
            smoothing_interpretation=smoothing_interpretation,
            paint_nside=paint_nside,
            ud_grade_power=ud_grade_power,
            ud_grade_order_in=ud_grade_order_in,
            ud_grade_order_out=ud_grade_order_out,
            ud_grade_pess=ud_grade_pess,
            max_comoving_radius=self.max_comoving_radius,
        )

        painted = jax.lax.map(paint_fn, (data, center_arr), batch_size=batch_size)
        painted = painted.squeeze()

        from .lightcone import SphericalDensity

        return SphericalDensity.FromDensityMetadata(
            array=painted,
            density_field=self,
            status=DensityStatus.LIGHTCONE,
        )


@partial(jax.jit, static_argnames=['status'])
def particle_from_density(
    array: Array,
    reference: DensityField,
    scale_factors: float | None = None,
    status: FieldStatus | None = None,
) -> ParticleField:
    """
    Create a ParticleField from an array, inheriting metadata from a reference DensityField.
    """
    return ParticleField.FromDensityMetadata(
        array,
        reference,
        status=status,
        scale_factors=scale_factors,
    )
