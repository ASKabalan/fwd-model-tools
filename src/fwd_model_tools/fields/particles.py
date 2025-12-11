from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, Literal, Optional

import jax
import jax.core
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.painting import cic_read, cic_read_dx
from jaxtyping import Array, Float

from .._src.base._core import AbstractField
from .._src.fields._painting import _single_paint, _single_paint_2d, _single_paint_spherical
from .._src.fields._plotting import (
    generate_titles,
    plot_3d_particles,
    prepare_axes,
    resolve_particle_weights,
)
from .density import DensityField, FieldStatus
from .lightcone import FlatDensity, SphericalDensity
from .units import DensityUnit, PositionUnit, convert_units

DEFAULT_CHUNK_SIZE = 2**24
SphericalScheme = Literal["ngp", "bilinear", "rbf_neighbor"]


@jax.tree_util.register_pytree_node_class
class ParticleField(AbstractField):
    """
    Field subclass representing particle positions or displacements.

    The `unit` encodes how the array is interpreted:

    - PositionUnit.GRID_RELATIVE : displacements from the uniform grid
      (used with CIC *_dx painting).
    - PositionUnit.GRID_ABSOLUTE : absolute grid coordinates in [0, N_mesh).
    - PositionUnit.MPC_H         : comoving Mpc/h positions (not paintable
      without an explicit conversion to a grid unit).

    Painting helpers live here so only particle-based data can be
    rasterized into density grids.
    """

    def __init__(
        self,
        *,
        array: Array,
        mesh_size: tuple[int, int, int],
        box_size: tuple[float, float, float],
        observer_position: tuple[float, float, float] = (0.5, 0.5, 0.5),
        sharding: Optional[Any] = None,
        halo_size: int | tuple[int, int] = 0,
        #  Lightcone geometry and metadata
        nside: Optional[int] = None,
        flatsky_npix: Optional[tuple[int, int]] = None,
        field_size: Optional[float] = None,
        # Dynamic metadata related to redshift of the field
        z_sources: Optional[Any] = None,
        scale_factors: Optional[Any] = None,
        comoving_centers: Optional[Any] = None,
        density_width: Optional[Any] = None,
        # Unit & status metadata
        status: FieldStatus = FieldStatus.PARTICLES,
        unit: PositionUnit = PositionUnit.GRID_RELATIVE,
    ):
        # DIFFRAX traces array with None value when checking terms; during
        # shape inference diffrax/equinox may pass ShapeDtypeStruct instead of a
        # concrete array, so avoid forcing a materialization in that case.
        if array is not None:
            array_shape = getattr(array, "shape", ())
            if not (
                (len(array_shape) == 4 and array_shape[-1] == 3)
                or (len(array_shape) == 5 and array_shape[-1] == 3)
                or array_shape == ()  # diffrax term compatibility traces shape ()
            ):
                raise ValueError(
                    f"ParticleField array must have shape (X, Y, Z, 3) or (N, X, Y, Z, 3); got shape {array_shape}"
                )

            if not isinstance(unit, PositionUnit):
                raise TypeError(f"ParticleField.unit must be a PositionUnit, got {unit!r}")

        super().__init__(
            array=array,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            sharding=sharding,
            halo_size=halo_size,
            # Lightcone geometry and metadata
            nside=nside,
            flatsky_npix=flatsky_npix,
            field_size=field_size,
            # Dynamic metadata related to redshift of the field
            z_sources=z_sources,
            scale_factors=scale_factors,
            comoving_centers=comoving_centers,
            density_width=density_width,
            status=status,
            unit=unit,
        )

    # ------------------------------------------------------------------ PyTree indexing

    def __getitem__(self, key) -> ParticleField:
        """
        Index into batched ParticleField.

        For a 5D array, slices the leading batch dimension.
        """
        if self.array.ndim != 5:
            raise ValueError(
                "Indexing only supported for batched ParticleField (5D array); "
                f"got array with {self.array.ndim} dimensions"
            )
        # Use the DensityField __getitem__ implementation (tree.map).
        return super().__getitem__(key)

        # ------------------------------------------------------------------ unit conversion

    def to(self, unit: PositionUnit) -> ParticleField:
        """
        Convert particle coordinates to a different PositionUnit.

        Examples
        --------
        - GRID_RELATIVE -> GRID_ABSOLUTE
        - GRID_ABSOLUTE -> MPC_H
        - MPC_H -> GRID_ABSOLUTE, etc.
        """
        if self.unit == unit:
            return self

        if not isinstance(self.unit, PositionUnit):
            raise TypeError(f"ParticleField.to expects self.unit to be a PositionUnit, got {self.unit!r}")

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            sharding=self.sharding,
        )

        return self.replace(array=new_array, unit=unit)

    # ------------------------------------------------------------------ painting: 3D

    @partial(jax.jit, static_argnames=("chunk_size", "batch_size"))
    def paint(
        self,
        *,
        mesh: Optional[Array] = None,
        weights: Array | float = 1.0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_size: Optional[int] = None,
    ) -> DensityField:
        """
        Paint particles onto a 3D density mesh using CIC interpolation.

        The interpretation of `self.array` is controlled by `self.unit`:

        - GRID_RELATIVE -> displacements, uses CIC-DX.
        - GRID_ABSOLUTE -> absolute grid coordinates, uses standard CIC.
        - MPC_H -> forbidden; convert first to a grid unit.
        """
        data = jnp.asarray(self.array)
        if self.unit == PositionUnit.MPC_H:
            raise ValueError(
                "Cannot paint ParticleField with unit MPC_H; convert to GRID_RELATIVE or GRID_ABSOLUTE first."
            )
        mode = "relative" if self.unit == PositionUnit.GRID_RELATIVE else "absolute"

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
            # (X,Y,Z,3) -> add batch axis
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
            unit=DensityUnit.DENSITY,
        )

    # ------------------------------------------------------------------ read-out from mesh

    @partial(jax.jit)
    def read_out(
        self,
        density_mesh: DensityField,
    ) -> DensityField:
        """
        Interpolate values from a 3D density mesh back to particle positions.

        Uses the same unit->mode mapping as `paint`:

        - GRID_RELATIVE -> cic_read_dx
        - GRID_ABSOLUTE -> cic_read
        - MPC_H -> error (convert first)
        """
        if self.unit == PositionUnit.MPC_H:
            raise ValueError(
                "Cannot read_out ParticleField with unit MPC_H; convert to GRID_RELATIVE or GRID_ABSOLUTE first."
            )
        mode = "relative" if self.unit == PositionUnit.GRID_RELATIVE else "absolute"

        if mode == "relative":
            read_data = cic_read_dx(
                density_mesh.array,
                self.array,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )
        else:  # "absolute"
            read_data = cic_read(
                density_mesh.array,
                self.array,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )

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
            unit=DensityUnit.DENSITY,
        )

    # ------------------------------------------------------------------ 2D flat-sky painting

    @partial(jax.jit, static_argnames=("batch_size",))
    def paint_2d(
        self,
        center: Float | Array,
        density_plane_width: Float | Array,
        *,
        weights: Optional[Array | float] = None,
        batch_size: Optional[int] = None,
    ) -> FlatDensity:
        """
        Project particles onto a flat-sky grid using CIC painting.

        The interpretation of `self.array` is again controlled by `self.unit`
        (GRID_RELATIVE vs GRID_ABSOLUTE). MPC_H is not allowed.
        """
        if self.flatsky_npix is None:
            raise ValueError("ParticleField requires `flatsky_npix` for paint_2d.")

        self = self.to(PositionUnit.GRID_ABSOLUTE)  # ensure absolute for 2D painting

        data = jnp.asarray(self.array)
        center_arr = jnp.atleast_1d(center)
        width_arr = jnp.atleast_1d(density_plane_width)

        if data.ndim == 5:
            nb_shells = data.shape[0]
            if center_arr.size != nb_shells:
                raise ValueError(f"Batched input: center must have {nb_shells} elements, got {center_arr.size}")
            if width_arr.size != nb_shells:
                raise ValueError(
                    f"Batched input: density_plane_width must have {nb_shells} elements, got {width_arr.size}"
                )
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            width_arr = width_arr[None, ...]
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
            weights=weights,
            max_comoving_radius=self.max_comoving_radius,
        )

        painted = jax.lax.map(paint_fn, (data, center_arr, width_arr), batch_size=batch_size)
        painted = painted.squeeze()
        center_arr.squeeze()
        widths = width_arr.squeeze()

        from .lightcone import FlatDensity

        flat = FlatDensity.FromDensityMetadata(
            array=painted,
            field=self,
            status=FieldStatus.LIGHTCONE,
            comoving_centers=center,
            density_width=widths,
        )
        # Painting produced an actual density field in standard density units.
        return flat.replace(unit=DensityUnit.DENSITY)

    # ------------------------------------------------------------------ spherical painting

    @partial(
        jax.jit,
        static_argnames=(
            "scheme",
            "kernel_width_arcmin",
            "smoothing_interpretation",
            "paint_nside",
            "ud_grade_power",
            "ud_grade_order_in",
            "ud_grade_order_out",
            "ud_grade_pess",
            "batch_size",
        ),
    )
    def paint_spherical(
        self,
        center: Float | Array,
        density_plane_width: Float | Array,
        *,
        scheme: SphericalScheme = "rbf_neighbor",
        weights: Optional[Array] = None,
        kernel_width_arcmin: Optional[float] = None,
        smoothing_interpretation: str = "fwhm",
        paint_nside: Optional[int] = None,
        ud_grade_power: float = 0.0,
        ud_grade_order_in: str = "RING",
        ud_grade_order_out: str = "RING",
        ud_grade_pess: bool = False,
        batch_size: Optional[int] = None,
    ) -> SphericalDensity:
        """
        Paint particles onto a HEALPix grid using spherical painting.
        """
        if self.nside is None:
            raise ValueError("Spherical painting requires `nside`.")
        self = self.to(PositionUnit.GRID_ABSOLUTE)  # ensure absolute for spherical painting

        data = jnp.asarray(self.array)
        center_arr = jnp.atleast_1d(center)
        width_arr = jnp.atleast_1d(density_plane_width)

        if data.ndim == 5:
            nb_shells = data.shape[0]
            if center_arr.size != nb_shells:
                raise ValueError(f"Batched input: center must have {nb_shells} elements, got {center_arr.size}")
            if width_arr.size != nb_shells:
                raise ValueError(
                    f"Batched input: density_plane_width must have {nb_shells} elements, got {width_arr.size}"
                )
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            width_arr = width_arr[None, ...]
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
            scheme=scheme,
            weights=weights,
            kernel_width_arcmin=kernel_width_arcmin,
            smoothing_interpretation=smoothing_interpretation,
            paint_nside=paint_nside,
            ud_grade_power=ud_grade_power,
            ud_grade_order_in=ud_grade_order_in,
            ud_grade_order_out=ud_grade_order_out,
            ud_grade_pess=ud_grade_pess,
            max_comoving_radius=self.max_comoving_radius,
        )

        painted = jax.lax.map(paint_fn, (data, center_arr, width_arr), batch_size=batch_size)
        painted = painted.squeeze()
        center_arr.squeeze()
        widths = width_arr.squeeze()

        from .lightcone import SphericalDensity

        sph = SphericalDensity.FromDensityMetadata(
            array=painted,
            field=self,
            status=FieldStatus.LIGHTCONE,
            comoving_centers=center,
            density_width=widths,
        )
        return sph.replace(unit=DensityUnit.DENSITY)

    # ------------------------------------------------------------------ plotting

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        weights: Optional[Array | str | float] = None,
        thinning: int = 4,
        cmap: str = "viridis",
        point_size: float = 5,
        alpha: float = 0.6,
        figsize: Optional[tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        elev: float = 40,
        azim: float = -30,
        zoom: float = 0.8,
        weights_title: Optional[str] = None,
        labels: tuple[str, str, str] = ("X", "Y", "Z"),
        ticks: Optional[tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        colorbar: bool = True,
    ):
        """
        Plot 3D particle field as scatter visualization.

        Parameters
        ----------
        ax : Axes3D or sequence of Axes3D, optional
            Pre-existing axes. If None, creates new figure.
        weights : array-like, str, float, or None
            Color specification for particles:
            - None: Uniform color (blue)
            - float: Scalar color for all particles
            - str: Keyword ('redshift', 'z', 'scale', 'a', 'comoving', 'r')
            - array: Direct weight values broadcastable to (Nx, Ny, Nz)
        thinning : int
            Take every n-th particle along each axis for visualization.
        cmap : str
            Colormap name.
        point_size : float
            Size of scatter points.
        alpha : float
            Transparency of points.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        ncols : int
            Number of columns for batch grid layout.
        titles : sequence of str, optional
            Titles for each subplot.
        vmin, vmax : float, optional
            Color range limits.
        elev, azim : float
            Elevation and azimuth angles for 3D view.
        zoom : float
            Zoom factor for the 3D view.
        weights_title : str, optional
            Colorbar label. Auto-set for string weight keywords.
        labels : tuple of str
            Axis labels (X, Y, Z).
        ticks : tuple of sequences, optional
            Tick values for each axis.
        colorbar : bool
            Whether to show colorbar.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        axes : ndarray
            Array of axes.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot traced arrays. Use outside of jit context.")

        data = np.asarray(self.array)

        # Handle batch dimension
        if data.ndim == 4:
            data = data[None, ...]
        elif data.ndim != 5:
            raise ValueError(f"Expected 4D or 5D array, got {data.ndim}D")

        n_plots = data.shape[0]
        fig, axes = prepare_axes(ax, n_plots, ncols, projection="3d", figsize=figsize)

        if titles is None:
            titles = generate_titles("Particles", self.scale_factors, n_plots)

        # Resolve weights
        resolved_weights, resolved_title = resolve_particle_weights(
            weights=weights,
            particles_shape=data.shape[1:],  # (Nx, Ny, Nz, 3)
            z_sources=np.asarray(self.z_sources) if self.z_sources is not None else None,
            scale_factors=np.asarray(self.scale_factors) if self.scale_factors is not None else None,
            comoving_centers=np.asarray(self.comoving_centers) if self.comoving_centers is not None else None,
            weights_title=weights_title,
        )

        if ticks is None:
            ticks = ([], [], [])

        for idx, ax_i in enumerate(axes):
            plot_3d_particles(
                ax_i,
                data[idx],
                weights=resolved_weights,
                thinning=thinning,
                cmap=cmap,
                point_size=point_size,
                alpha=alpha,
                elev=elev,
                azim=azim,
                zoom=zoom,
                weights_title=resolved_title,
                labels=labels,
                ticks=ticks,
                colorbar=colorbar,
                vmin=vmin,
                vmax=vmax,
            )
            if titles and idx < len(titles):
                ax_i.set_title(titles[idx])

        fig.tight_layout()
        return fig, axes

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        weights: Optional[Array | str | float] = None,
        thinning: int = 4,
        cmap: str = "viridis",
        point_size: float = 5,
        alpha: float = 0.6,
        figsize: Optional[tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        elev: float = 40,
        azim: float = -30,
        zoom: float = 0.8,
        weights_title: Optional[str] = None,
        labels: tuple[str, str, str] = ("X", "Y", "Z"),
        ticks: Optional[tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        colorbar: bool = True,
    ) -> None:
        """Display 3D particle field using :meth:`plot`."""
        self.plot(
            ax=ax,
            weights=weights,
            thinning=thinning,
            cmap=cmap,
            point_size=point_size,
            alpha=alpha,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            vmin=vmin,
            vmax=vmax,
            elev=elev,
            azim=azim,
            zoom=zoom,
            weights_title=weights_title,
            labels=labels,
            ticks=ticks,
            colorbar=colorbar,
        )
        plt.show()
