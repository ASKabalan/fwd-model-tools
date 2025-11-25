from __future__ import annotations

from enum import Enum
from functools import partial
from math import ceil
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple

import healpy as hp
import jax
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import matplotlib.pyplot as plt
import numpy as np
from jax.image import resize
from jaxpm.distributed import uniform_particles
from jaxpm.painting import (cic_paint, cic_paint_2d, cic_paint_dx, cic_read,
                            cic_read_dx)
from jaxpm.spherical import paint_particles_spherical
from jaxtyping import Array
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fwd_model_tools._src._checks import (_ensure_tuple2, _ensure_tuple3,
                                          _normalize_halo_size,
                                          _optional_positive_int,
                                          _optional_tuple2_positive)
from fwd_model_tools._src._painting import (_single_paint, _single_paint_2d,
                                            _single_paint_spherical)
from fwd_model_tools._src.core import AbstractField
from fwd_model_tools.power import (PowerSpectrum, angular_cl_flat,
                                   angular_cl_spherical, coherence)
from fwd_model_tools.power import power as power_fn
from fwd_model_tools.power import transfer

__all__ = [
    "FieldStatus",
    "DensityStatus",
    "AbstractField",
    "DensityField",
    "ParticleField",
    "FlatDensity",
    "SphericalDensity",
    "stack",
]

PaintMode = Literal["relative", "absolute"]
SphericalScheme = Literal["ngp", "bilinear", "rbf_neighbor"]
DEFAULT_CHUNK_SIZE = 2**24
Float = float


class FieldStatus(str, Enum):
    """Lifecycle state for 3D volumetric fields."""

    RAW = "raw"
    INITIAL_FIELD = "initial_field"
    LPT1 = "lpt1"
    LPT2 = "lpt2"
    DENSITY_FIELD = "density_field"
    PARTICLES = "particles"


class DensityStatus(str, Enum):
    """Lifecycle state for 2D density/shear maps."""

    PROJECTED_DENSITY = "projected_density"
    LIGHTCONE = "lightcone"
    KAPPA = "kappa"
    GAMMA = "gamma"


@jax.tree_util.register_pytree_node_class
class DensityField(AbstractField):
    """
    PyTree container for volumetric simulation arrays and their static metadata.
    """

    __slots__ = (
        "array",
        "mesh_size",
        "box_size",
        "observer_position",
        "sharding",
        "nside",
        "flatsky_npix",
        "field_size",
        "halo_size",
        "z_source",
        "status",
        "scale_factors",
    )

    STATUS_ENUM = FieldStatus

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
        super().__init__(array=array)
        mesh_size = _ensure_tuple3("mesh_size", mesh_size, cast=int)
        box_size = _ensure_tuple3("box_size", box_size, cast=float)
        observer_position = _ensure_tuple3("observer_position",
                                           observer_position,
                                           cast=float)
        if any(not 0.0 <= frac <= 1.0 for frac in observer_position):
            raise ValueError(
                "observer_position entries must lie within [0, 1]")
        halo_size = _normalize_halo_size(halo_size)
        nside = _optional_positive_int("nside", nside)
        flatsky_npix = _optional_tuple2_positive("flatsky_npix", flatsky_npix)

        self.mesh_size = mesh_size
        self.box_size = box_size
        self.observer_position = observer_position
        self.sharding = sharding
        self.nside = nside
        self.flatsky_npix = flatsky_npix
        self.field_size = field_size
        self.halo_size = halo_size
        self.z_source = z_source
        self.status = self._coerce_status(status)
        self.scale_factors = scale_factors

    # ------------------------------------------------------------------ PyTree
    def tree_flatten(self):
        children = (self.array, self.scale_factors)
        aux_data = (
            self.mesh_size,
            self.box_size,
            self.observer_position,
            self.sharding,
            self.nside,
            self.flatsky_npix,
            self.field_size,
            self.halo_size,
            self.z_source,
            self.status,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            mesh_size,
            box_size,
            observer_position,
            sharding,
            nside,
            flatsky_npix,
            field_size,
            halo_size,
            z_source,
            status,
        ) = aux_data
        array, scale_factors = children
        return cls(
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

    # --------------------------------------------------------------- properties
    @property
    def observer_position_mpc(self) -> tuple[float, float, float]:
        """Observer coordinates in physical units (box_size scaled)."""
        return tuple(
            frac * length
            for frac, length in zip(self.observer_position, self.box_size))

    @property
    def max_comoving_radius(self) -> float:
        """Maximum comoving radius accommodated by the box given observer position."""
        box = np.asarray(self.box_size)
        observer_position = np.asarray(self.observer_position)
        factors = np.clip(observer_position, 0.0, 1.0)
        factors = 1.0 + 2.0 * np.minimum(factors, 1.0 - factors)
        return np.min(box / factors)

    def density_width(self, nb_shells) -> float:
        """Physical thickness of each density shell in the lightcone."""
        return self.max_comoving_radius / nb_shells

    @classmethod
    def _coerce_status(cls, status):
        enum_cls = cls.STATUS_ENUM
        return enum_cls(status)

    def with_array(self, array: Array) -> "DensityField":
        return self.replace(array=array)

    def with_scale_factors(self, scale_factors: float) -> "DensityField":
        return self.replace(scale_factors=scale_factors)

    def replace(self, **updates: Any) -> "DensityField":
        params = {
            "array": self.array,
            "mesh_size": self.mesh_size,
            "box_size": self.box_size,
            "observer_position": self.observer_position,
            "sharding": self.sharding,
            "nside": self.nside,
            "flatsky_npix": self.flatsky_npix,
            "field_size": self.field_size,
            "halo_size": self.halo_size,
            "z_source": self.z_source,
            "status": self.status,
            "scale_factors": self.scale_factors,
        }
        allowed_keys = set(params.keys())
        unknown = set(updates) - allowed_keys
        if unknown:
            raise TypeError(f"Unknown DensityField attribute(s): {unknown}")
        params.update(updates)
        instance = object.__new__(type(self))
        for key, value in params.items():
            setattr(instance, key, value)
        return instance

    def set_scale_factors(self, scale_factors: float) -> None:
        self.scale_factors = scale_factors

    # -------------------------------------------------------- power-spectrum API
    def power(
            self,
            mesh2: Optional[DensityField] = None,
            *,
            kedges: Optional[Array | jnp.ndarray] = None,
            multipoles: Optional[Iterable[int]] = 0,
            los: Array | Iterable[float] = (0.0, 0.0, 1.0),
    ) -> "PowerSpectrum":
        """Compute the 3D matter power spectrum P(k).

        Parameters mirror :func:`fwd_model_tools.power.power`. Any keyword
        arguments are forwarded verbatim to that helper.
        """
        box_shape = tuple(self.box_size)
        multipoles_static = tuple(multipoles) if isinstance(
            multipoles, (list, tuple)) else multipoles
        los_tuple = None if multipoles_static == 0 else tuple(
            np.asarray(los, dtype=float))

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 3:
            data1 = jnp.expand_dims(data1, axis=0)
            data2 = jnp.expand_dims(data2,
                                    axis=0) if data2 is not None else None
        elif data1.ndim != 4:
            raise ValueError(
                "DensityField.power expects array shape (X,Y,Z) or (B,X,Y,Z)")

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match mesh shape for power")

        def _power_fn(pair):
            arr1, arr2 = pair
            return power_fn(
                arr1,
                arr2,
                box_shape=box_shape,
                kedges=kedges,
                multipoles=multipoles_static,
                los=los_tuple,
            )

        k, pk = jax.lax.map(_power_fn, (data1, data2))
        k, pk = k[0], pk.squeeze()
        return PowerSpectrum(wavenumber=k, spectra=pk, name="pk")

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        nz_slices: int = 10,
        cmap: str = "magma",
        figsize: Optional[Tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        apply_log: bool = True,
        **kwargs,
    ):
        """Project along z and plot via :class:`FlatDensity` utilities (batch supported)."""
        if not jax.core.is_concrete(self.array):
            raise ValueError(
                "Cannot plot traced arrays. Use outside of jit context.")

        flat = self.project(nz_slices=nz_slices)
        fig, axes = flat.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
            apply_log=apply_log,
            **kwargs,
        )
        return fig, axes

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        nz_slices: int = 10,
        cmap: str = "magma",
        figsize: Optional[Tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        apply_log: bool = True,
        **kwargs,
    ) -> None:
        """Display projected density using :meth:`plot`."""
        self.plot(
            ax=ax,
            nz_slices=nz_slices,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
            apply_log=apply_log,
            **kwargs,
        )
        plt.show()

    def transfer(
        self,
        other: "DensityField",
        *,
        kedges: Optional[Array | jnp.ndarray] = None,
    ) -> "PowerSpectrum":
        """Monopole transfer function sqrt(P_other / P_self)."""

        def _compute(pair):
            arr1, arr2 = pair
            return transfer(
                arr1,
                arr2,
                box_shape=self.box_size,
                kedges=kedges,
            )

        data1 = self.array
        data2 = other.array

        if data1.ndim == 3:
            data1 = data1[None, ...]
            data2 = data2[None, ...]
        elif data1.ndim != 4:
            raise ValueError(
                "DensityField.transfer expects array shape (X,Y,Z) or (B,X,Y,Z)"
            )

        if data2.shape != data1.shape:
            raise ValueError("other array must match shape for transfer")

        k_stack, spectra_stack = jax.lax.map(_compute, (data1, data2))
        wavenumber = k_stack[0]
        spectra = spectra_stack if self.array.ndim == 4 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber,
                             spectra=spectra,
                             name="transfer")

    def coherence(
        self,
        other: "DensityField",
        *,
        kedges: Optional[Array | jnp.ndarray] = None,
    ) -> "PowerSpectrum":
        """Monopole coherence pk01 / sqrt(pk0 pk1)."""

        def _compute(pair):
            arr1, arr2 = pair
            return coherence(
                arr1,
                arr2,
                box_shape=self.box_size,
                kedges=kedges,
            )

        data1 = self.array
        data2 = other.array

        if data1.ndim == 3:
            data1 = data1[None, ...]
            data2 = data2[None, ...]
        elif data1.ndim != 4:
            raise ValueError(
                "DensityField.coherence expects array shape (X,Y,Z) or (B,X,Y,Z)"
            )

        if data2.shape != data1.shape:
            raise ValueError("other array must match shape for coherence")

        k_stack, spectra_stack = jax.lax.map(_compute, (data1, data2))
        wavenumber = k_stack[0]
        spectra = spectra_stack if self.array.ndim == 4 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber,
                             spectra=spectra,
                             name="coherence")

    def block_until_ready(self) -> DensityField:
        """
        Block until the underlying array is ready.

        Returns
        -------
        DensityField
            Self, after blocking.
        """
        self.array.block_until_ready()
        return self

    def __repr__(self) -> str:
        classname = str(self.__class__.__name__)
        return (
            f"{classname}("
            f"array=Array{tuple(self.array.shape)}, "
            f"mesh_size={self.mesh_size}, "
            f"box_size={self.box_size}, "
            f"status={self.status.value}, "
            f"scale_factors_shape={jnp.atleast_1d(self.scale_factors).shape})")

    @partial(jax.jit, static_argnames=['nz_slices'])
    def project(self, nz_slices: int = 10) -> "FlatDensity":
        """
        Create a 2D projection by summing slices along the z-axis.

        Parameters
        ----------
        nz_slices : int, default=10
            Number of z-slices to sum from the end of the array.

        Returns
        -------
        FlatDensity
            2D flat-sky density map.
        """
        data = jnp.asarray(self.array)

        if data.ndim not in (3, 4):
            raise ValueError(
                f"project() expects 3D array or batch of 3D arrays, got shape {data.shape}"
            )

        # Vectorized: sum over last nz_slices along z-axis (last dimension)
        projection = jnp.sum(data[..., -nz_slices:], axis=-1)

        # Update flatsky_npix to match projected shape
        # For 3D: shape is (X, Y), for 4D: shape is (N, X, Y) - take last 2 dims
        flatsky_npix = projection.shape[
            -2:] if data.ndim == 4 else projection.shape
        projected_field = self.replace(flatsky_npix=flatsky_npix)

        return FlatDensity.FromDensityMetadata(
            array=projection,
            density_field=projected_field,
            status=DensityStatus.PROJECTED_DENSITY,
        )

    def __getitem__(self, key) -> "DensityField":
        """
        Index into batched DensityField.

        Enables extracting individual shells or slices from lightcone batches.

        Parameters
        ----------
        key : int or slice
            Index or slice to extract from batch dimension.

        Returns
        -------
        DensityField
            New DensityField with indexed array and scale_factors.

        Raises
        ------
        ValueError
            If array is not batched (ndim < 4).

        Examples
        --------
        >>> lightcone.array.shape  # (10, 256, 256, 256)
        >>> shell = lightcone[0]
        >>> shell.array.shape  # (256, 256, 256)
        >>> reversed_lightcone = lightcone[::-1]
        >>> reversed_lightcone.array.shape  # (10, 256, 256, 256)
        """
        if self.array.ndim < 4:
            raise ValueError(
                f"Indexing only supported for batched DensityField (4D array), "
                f"got array with {self.array.ndim} dimensions")

        indexed_array = self.array[key]
        indexed_scale_factors = self.scale_factors[key]

        return type(self)(
            array=indexed_array,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            halo_size=self.halo_size,
            status=self.status,
            scale_factors=indexed_scale_factors,
        )

    @classmethod
    def stack(cls, fields: Sequence["DensityField"]) -> "DensityField":
        """
        Stack multiple FlatDensity instances along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("FlatDensity.stack requires at least one field.")
        return jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *field_list,
        )


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
            if not ((array.ndim == 4 and array.shape[-1] == 3) or
                    (array.ndim == 5 and array.shape[-1] == 3)):
                raise ValueError(
                    "ParticleField array must have shape (X, Y, Z, 3) or (N, X, Y, Z, 3)"
                )

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

    @classmethod
    def FromDensityMetadata(
        cls,
        array: Array,
        reference: DensityField,
        *,
        status: FieldStatus | None = None,
        scale_factors: float | None = None,
    ) -> "ParticleField":
        """
        Construct a ParticleField from a reference DensityField and a new array.

        This mirrors the older particle_from_density helper but is exposed as a
        classmethod so it can be used by subclasses.
        """
        return cls(
            array=array,
            mesh_size=reference.mesh_size,
            box_size=reference.box_size,
            observer_position=reference.observer_position,
            sharding=reference.sharding,
            nside=reference.nside,
            flatsky_npix=reference.flatsky_npix,
            field_size=reference.field_size,
            halo_size=reference.halo_size,
            z_source=reference.z_source,
            status=status if status is not None else reference.status,
            scale_factors=(scale_factors if scale_factors is not None else
                           reference.scale_factors),
        )

    def __getitem__(self, key) -> "ParticleField":
        """
        Index into batched ParticleField.

        Enables extracting individual shells from lightcone batches.

        Parameters
        ----------
        key : int or slice
            Index or slice to extract from batch dimension.

        Returns
        -------
        ParticleField
            New ParticleField with indexed array.

        Examples
        --------
        >>> dx_lightcone.array.shape  # (10, 256, 256, 256, 3)
        >>> dx_single = dx_lightcone[0]
        >>> dx_single.array.shape  # (256, 256, 256, 3)
        """
        if self.array.ndim != 5:
            raise ValueError(
                "Indexing only supported for batched ParticleField (5D array)")

        indexed_array = self.array[key]
        indexed_scale_factors = self.scale_factors[key]

        return type(self).FromDensityMetadata(
            indexed_array,
            self,
            status=self.status,
            scale_factors=indexed_scale_factors,
        )

    @partial(jax.jit,
             static_argnames=['mode', 'weights', 'chunk_size', 'batch_size'])
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

        Supports both batched (5D) and single (4D) particle arrays.

        Parameters
        ----------
        mode : PaintMode, default="relative"
            "relative" for displacement-based painting, "absolute" for position-based.
        mesh : Array, optional
            Pre-allocated mesh for absolute mode.
        weights : Array | float, default=1.0
            Particle weights for painting.
        chunk_size : int
            Chunk size for painting operations.
        batch_size : int, optional
            Number of iterations to process in parallel per batch. Controls memory/performance
            trade-off when mapping over batched inputs. If None, processes sequentially.

        Returns
        -------
        DensityField
            Painted 3D density field.
        """
        data = jnp.asarray(self.array)

        # Create partial function with all fixed parameters
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
            # Single shell: add leading axis, map, then squeeze
            data = data[None, ...]
        elif data.ndim != 5:
            raise ValueError(
                f"paint() expects 4D or 5D array, got shape {data.shape}")

        # Batched input: map over batch dimension
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
        mesh = density_mesh.array if isinstance(density_mesh,
                                                DensityField) else density_mesh
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

    @partial(jax.jit,
             static_argnames=[
                 'mode', 'weights', 'density_plane_width', 'batch_size'
             ])
    def paint_2d(
        self,
        center: Float | Array,
        *,
        density_plane_width: Optional[Float] = None,
        weights: Optional[Array | float] = None,
        mode: PaintMode = "relative",
        batch_size: Optional[int] = None,
    ) -> "FlatDensity":
        """
        Project particles onto a flat-sky grid using CIC painting.

        Supports batched input: if self.array has shape (N, X, Y, Z, 3) and center
        is an array of shape (N,), returns FlatDensity with shape (N, flatsky_npix[0], flatsky_npix[1]).

        Parameters
        ----------
        center : Float | Array
            Center of density plane(s) in Mpc. Scalar for single shell, array for batched.
        density_plane_width : Float, optional
            Physical width of density plane in Mpc. Defaults to max_comoving_radius / nb_shells.
        weights : Array | float, optional
            Particle weights for painting.
        mode : PaintMode, default="relative"
            "relative" for displacement-based painting, "absolute" for position-based.
        batch_size : int, optional
            Number of iterations to process in parallel per batch. Controls memory/performance
            trade-off when mapping over batched inputs. If None, processes sequentially.

        Returns
        -------
        FlatDensity
            2D flat-sky density map(s).
        """
        if self.flatsky_npix is None:
            raise ValueError(
                "ParticleField requires `flatsky_npix` for paint_2d.")

        data = jnp.asarray(self.array)
        center_arr = jnp.atleast_1d(center)

        if data.ndim == 5:
            # Batched input
            nb_shells = data.shape[0]
            if center_arr.size != nb_shells:
                raise ValueError(
                    f"Batched input: center must have {nb_shells} elements, got {center_arr.size}"
                )
            density_plane_width = self.density_width(
                nb_shells
            ) if density_plane_width is None else density_plane_width
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            if density_plane_width is None:
                raise ValueError(
                    "density_plane_width must be specified for single shell")
        else:
            raise ValueError(
                f"paint_2d() expects 4D or 5D array, got shape {data.shape}")

        # Create partial function with all fixed parameters
        paint_fn = jax.tree_util.Partial(
            _single_paint_2d,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            flatsky_npix=self.flatsky_npix,
            halo_size=self.halo_size,
            density_plane_width=density_plane_width,
            weights=weights,
            mode=mode,
            max_comoving_radius=self.max_comoving_radius,
        )

        painted = jax.lax.map(paint_fn, (data, center_arr),
                              batch_size=batch_size)
        painted = painted.squeeze()

        return FlatDensity.FromDensityMetadata(
            array=painted,
            density_field=self,
            status=DensityStatus.LIGHTCONE,
        )

    @partial(jax.jit,
             static_argnames=[
                 'mode', 'scheme', 'weights', 'density_plane_width',
                 'kernel_width_arcmin', 'smoothing_interpretation',
                 'paint_nside', 'ud_grade_power', 'ud_grade_order_in',
                 'ud_grade_order_out', 'ud_grade_pess', 'batch_size'
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
    ) -> "SphericalDensity":
        """
        Paint particles onto a HEALPix grid using spherical painting.

        Supports batched input: if self.array has shape (N, X, Y, Z, 3) and center
        is an array of shape (N,), returns SphericalDensity with shape (N, npix).

        Parameters
        ----------
        center : Float | Array
            Center of density shell(s) in Mpc. Scalar for single shell, array for batched.
        mode : PaintMode, default="relative"
            "relative" for displacement-based painting, "absolute" for position-based.
        scheme : SphericalScheme, default="rbf_neighbor"
            Painting method: "ngp", "bilinear", or "rbf_neighbor".
        weights : Array, optional
            Particle weights for painting.
        density_plane_width : Float, optional
            Physical width of density shell in Mpc. Defaults to max_comoving_radius / nb_shells.
        kernel_width_arcmin : float, optional
            Kernel width in arcminutes (for RBF method).
        smoothing_interpretation : str, default="fwhm"
            How to interpret kernel_width_arcmin ("fwhm" or "sigma").
        paint_nside : int, optional
            Higher nside for painting before downgrading.
        ud_grade_power : float, default=0.0
            Power for ud_grade operation.
        ud_grade_order_in : str, default="RING"
            HEALPix ordering for input.
        ud_grade_order_out : str, default="RING"
            HEALPix ordering for output.
        ud_grade_pess : bool, default=False
            Use pessimistic ud_grade.
        batch_size : int, optional
            Number of iterations to process in parallel per batch. Controls memory/performance
            trade-off when mapping over batched inputs. If None, processes sequentially.

        Returns
        -------
        SphericalDensity
            HEALPix density map(s).
        """
        if self.nside is None:
            raise ValueError("Spherical painting requires `nside`.")

        data = jnp.asarray(self.array)
        center_arr = jnp.atleast_1d(center)

        if data.ndim == 5:
            # Batched input
            nb_shells = data.shape[0]
            if center_arr.size != nb_shells:
                raise ValueError(
                    f"Batched input: center must have {nb_shells} elements, got {center_arr.size}"
                )
            density_plane_width = self.density_width(
                nb_shells
            ) if density_plane_width is None else density_plane_width
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            if density_plane_width is None:
                raise ValueError(
                    "density_plane_width must be specified for single shell")
        else:
            raise ValueError(
                f"paint_spherical() expects 4D or 5D array, got shape {data.shape}"
            )

        # Create partial function with all fixed parameters
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

        painted = jax.lax.map(paint_fn, (data, center_arr),
                              batch_size=batch_size)
        painted = painted.squeeze()

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

    This helper function is used to construct ParticleField objects while preserving
    all metadata (mesh_size, box_size, observer_position, sharding, etc.) from a
    reference field.

    Parameters
    ----------
    array : Array
        Particle data array with shape (X, Y, Z, 3) or (N, X, Y, Z, 3).
    reference : DensityField
        Reference field to copy metadata from.
    status : FieldStatus
        Status for the new ParticleField (e.g., FieldStatus.PARTICLES, FieldStatus.LPT1).
    scale_factors : float
        Scale factors for the new ParticleField.

    Returns
    -------
    ParticleField
        New ParticleField with the given array and metadata from reference.

    Examples
    --------
    >>> forces_array = compute_forces(positions.array)
    >>> forces = particle_from_density(forces_array, positions,
    ...                                 status=FieldStatus.PARTICLES,
    ...                                 scale_factors=positions.scale_factors)
    """
    return ParticleField.FromDensityMetadata(
        array,
        reference,
        status=status,
        scale_factors=scale_factors,
    )


@jax.tree_util.register_pytree_node_class
class FlatDensity(DensityField):
    """Flat-sky (2D) density or shear maps derived from volumetric simulations."""

    __slots__ = ()
    STATUS_ENUM = DensityStatus

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
        status: DensityStatus = DensityStatus.LIGHTCONE,
        scale_factors: float = 1.0,
    ):
        if flatsky_npix is None:
            raise ValueError("FlatDensity requires `flatsky_npix`.")

        arr = jnp.asarray(array)
        if arr.ndim == 2:
            spatial_shape = arr.shape
        elif arr.ndim == 3:
            spatial_shape = arr.shape[-2:]
        else:
            raise ValueError(
                "FlatDensity array must have shape (ny, nx) or (n_planes, ny, nx)."
            )

        if spatial_shape != tuple(flatsky_npix):
            raise ValueError(
                f"Array spatial shape {spatial_shape} does not match "
                f"flatsky_npix {flatsky_npix}.")

        super().__init__(
            array=arr,
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

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array: Array,
        density_field: DensityField,
        status: DensityStatus = DensityStatus.LIGHTCONE,
        z_source: Optional[Any] = None,
        scale_factors: float | None = None,
    ) -> "FlatDensity":
        """
        Construct a FlatDensity from a reference DensityField and a 2D/3D array.

        This preserves the behavior of the older FlatDensity __init__ that
        accepted a `density_field` argument.
        """
        if density_field.flatsky_npix is None:
            raise ValueError(
                "FlatDensity requires `flatsky_npix` in the source DensityField."
            )

        arr = jnp.asarray(array)
        if arr.ndim == 2:
            spatial_shape = arr.shape
        elif arr.ndim == 3:
            spatial_shape = arr.shape[-2:]
        else:
            raise ValueError(
                "FlatDensity array must have shape (ny, nx) or (n_planes, ny, nx)."
            )

        if spatial_shape != tuple(density_field.flatsky_npix):
            raise ValueError(
                f"Array spatial shape {spatial_shape} does not match "
                f"flatsky_npix {density_field.flatsky_npix}.")

        return cls(
            array=arr,
            mesh_size=density_field.mesh_size,
            box_size=density_field.box_size,
            observer_position=density_field.observer_position,
            sharding=density_field.sharding,
            nside=density_field.nside,
            flatsky_npix=density_field.flatsky_npix,
            field_size=density_field.field_size,
            halo_size=density_field.halo_size,
            z_source=(z_source
                      if z_source is not None else density_field.z_source),
            status=status,
            scale_factors=(scale_factors if scale_factors is not None else
                           density_field.scale_factors),
        )

    def angular_cl(
        self,
        mesh2: Optional["FlatDensity"] = None,
        *,
        field_size: Optional[float] = None,
        pixel_size: Optional[float] = None,
        ell_edges: Iterable[float] | None = None,
    ) -> "PowerSpectrum":
        """Compute a flat-sky angular power spectrum C_ell (auto or cross)."""

        effective_field_size = field_size or self.field_size
        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 2:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 3:
            raise ValueError(
                "FlatDensity.angular_cl expects array shape (ny,nx) or (B,ny,nx)"
            )

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_flat(
                m1,
                m2,
                pixel_size=pixel_size,
                field_size=effective_field_size,
                ell_edges=ell_edges,
            )

        if data2 is None:
            ell_stack, spectra_stack = jax.lax.map(
                lambda m: _compute((m, None)),
                data1,
            )
        else:
            ell_stack, spectra_stack = jax.lax.map(
                lambda pair: _compute(pair),
                (data1, data2),
            )

        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 3 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber, spectra=spectra, name="cl")

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Optional[Tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        apply_log: bool = True,
    ):
        """
        Visualize one or more flat-sky maps using matplotlib.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).

        Parameters
        ----------
        cmap : str, default="magma"
            Matplotlib colormap.
        figsize : tuple, optional
            Figure size in inches. Defaults to (5 * ncols, 5 * nrows).
        ncols : int, default=3
            Number of subplot columns.
        titles : Sequence[str], optional
            Optional per-panel titles.
        show_colorbar : bool, default=True
            Whether to draw a colorbar next to each subplot.
        show_ticks : bool, default=True
            Whether to keep axis ticks/labels. Set to False for cleaner grids.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError(
                "Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 2:
            data = data[None, ...]
        elif data.ndim != 3:
            raise ValueError(
                "FlatDensity.plot expects array shape (ny, nx) or (n, ny, nx)."
            )

        n_maps = data.shape[0]

        def _flatten_axes(axes_obj):
            if axes_obj is None:
                return None
            if isinstance(axes_obj, np.ndarray):
                return axes_obj.ravel()
            if isinstance(axes_obj, Sequence):
                return np.array(axes_obj, dtype=object).ravel()
            return np.array([axes_obj], dtype=object)

        axes_flat = _flatten_axes(ax)

        if axes_flat is None:
            ncols_eff = max(1, min(ncols, n_maps))
            nrows = ceil(n_maps / ncols_eff)
            if figsize is None:
                figsize = (5 * ncols_eff, 5 * nrows)
            fig, axes_created = plt.subplots(nrows,
                                             ncols_eff,
                                             figsize=figsize,
                                             squeeze=False)
            axes_flat = axes_created.ravel()
        else:
            # Derive grid from provided axes shape if possible
            if axes_flat.size < n_maps:
                raise ValueError(
                    "Provided axes array is too small for number of maps")
            fig = axes_flat[0].get_figure()
            # infer columns: if original was ndarray with 2 dims, respect that
            if isinstance(ax, np.ndarray) and ax.ndim == 2:
                nrows, ncols_eff = ax.shape[:2]
            else:
                ncols_eff = min(ncols or axes_flat.size, axes_flat.size)
                nrows = ceil(n_maps / ncols_eff)
            axes_flat = axes_flat[:nrows * ncols_eff]
            axes_created = axes_flat.reshape(nrows, ncols_eff)

        for idx, ax_i in enumerate(axes_flat):
            if idx < n_maps:
                to_plot = jnp.log10(data[idx] + 1) if apply_log else data[idx]
                im = ax_i.imshow(to_plot, origin="lower", cmap=cmap)
                if show_colorbar:
                    divider = make_axes_locatable(ax_i)
                    cax = divider.append_axes("right", size="3%", pad=0.03)
                    fig.colorbar(im, cax=cax)
                if not show_ticks:
                    ax_i.set_xticks([])
                    ax_i.set_yticks([])
                if titles and idx < len(titles):
                    ax_i.set_title(titles[idx])
            else:
                ax_i.axis("off")

        fig.tight_layout()

        if ax is not None:
            axes_out = axes_flat.reshape(axes_created.shape)
        else:
            axes_out = axes_created

        return fig, axes_out

    def __getitem__(self, key) -> "FlatDensity":
        """
        Index into batched FlatDensity.

        Enables extracting individual shells or slices from lightcone batches.

        Parameters
        ----------
        key : int or slice
            Index or slice to extract from batch dimension.

        Returns
        -------
        FlatDensity
            New FlatDensity with indexed array and scale_factors.

        Raises
        ------
        ValueError
            If array is not batched (ndim < 3).

        Examples
        --------
        >>> flat_lightcone.array.shape  # (10, 256, 256)
        >>> shell = flat_lightcone[0]
        >>> shell.array.shape  # (256, 256)
        >>> reversed_lightcone = flat_lightcone[::-1]
        >>> reversed_lightcone.array.shape  # (10, 256, 256)
        """
        if self.array.ndim < 3:
            raise ValueError(
                f"Indexing only supported for batched FlatDensity (3D array), "
                f"got array with {self.array.ndim} dimensions")

        indexed_array = self.array[key]
        indexed_scale_factors = self.scale_factors[key]

        # Reconstruct via metadata-aware factory while updating scale_factors.
        temp_field = self.replace(scale_factors=indexed_scale_factors)
        return type(self).FromDensityMetadata(
            array=indexed_array,
            density_field=temp_field,
            status=self.status,
            scale_factors=indexed_scale_factors,
        )

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Optional[Tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        apply_log: bool = True,
    ) -> None:
        """
        Plot and display flat-sky maps using matplotlib.

        Parameters mirror :meth:`FlatDensity.plot`.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError(
                "Cannot plot/show traced arrays. Use outside of jit context.")

        self.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
            apply_log=apply_log,
        )
        plt.show()

    def ud_sample(self, new_npix):
        """
        Resample to new resolution using jax.image.resize.

        Parameters
        ----------
        new_npix : int
            New pixel resolution (assumes square grid)

        Returns
        -------
        FlatDensity
            Resampled flat-sky map (functional, immutable)

        Examples
        --------
        >>> density_hires = density.ud_sample(new_npix=512)
        """
        # Handle batch dimension: (n_sources, ny, nx) or (ny, nx)
        if self.array.ndim == 3:
            new_shape = (self.array.shape[0], new_npix, new_npix)
        else:
            new_shape = (new_npix, new_npix)

        resampled = resize(self.array, new_shape, method="bilinear")

        return self.replace(array=resampled, flatsky_npix=(new_npix, new_npix))

    @classmethod
    def stack(cls, fields: Sequence["FlatDensity"]) -> "FlatDensity":
        """
        Stack multiple FlatDensity instances along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("FlatDensity.stack requires at least one field.")
        return jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *field_list,
        )


@jax.tree_util.register_pytree_node_class
class SphericalDensity(DensityField):
    """Spherical (HEALPix) density or shear maps produced from simulations."""

    __slots__ = ()
    STATUS_ENUM = DensityStatus

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
        status: DensityStatus = DensityStatus.LIGHTCONE,
        scale_factors: float = 1.0,
    ):
        if nside is None:
            raise ValueError("SphericalDensity requires `nside`.")

        arr = jnp.asarray(array)
        npix = jhp.nside2npix(nside)
        if arr.shape[-1] != npix:
            raise ValueError(
                f"Array last dimension {arr.shape[-1]} does not match HEALPix npix "
                f"{npix} for nside {nside}.")

        super().__init__(
            array=arr,
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

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array: Array,
        density_field: DensityField,
        status: DensityStatus = DensityStatus.LIGHTCONE,
        z_source: Optional[Any] = None,
        scale_factors: float | None = None,
    ) -> "SphericalDensity":
        """
        Construct a SphericalDensity from a reference DensityField and a 1D/2D array.

        This preserves the behavior of the older SphericalDensity __init__ that
        accepted a `density_field` argument.
        """
        if density_field.nside is None:
            raise ValueError("SphericalDensity requires `nside`.")

        arr = jnp.asarray(array)
        npix = jhp.nside2npix(density_field.nside)
        if arr.shape[-1] != npix:
            raise ValueError(
                f"Array last dimension {arr.shape[-1]} does not match HEALPix npix "
                f"{npix} for nside {density_field.nside}.")

        return cls(
            array=arr,
            mesh_size=density_field.mesh_size,
            box_size=density_field.box_size,
            observer_position=density_field.observer_position,
            sharding=density_field.sharding,
            nside=density_field.nside,
            flatsky_npix=density_field.flatsky_npix,
            field_size=density_field.field_size,
            halo_size=density_field.halo_size,
            z_source=(z_source
                      if z_source is not None else density_field.z_source),
            status=status,
            scale_factors=(scale_factors if scale_factors is not None else
                           density_field.scale_factors),
        )

    def angular_cl(
        self,
        mesh2: Optional["SphericalDensity"] = None,
        *,
        lmax: Optional[int] = None,
    ) -> "PowerSpectrum":
        """Compute a spherical (HEALPix) angular power spectrum C_ell (auto or cross)."""

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 1:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 2:
            raise ValueError(
                "SphericalDensity.angular_cl expects array shape (npix) or (B,npix)"
            )

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_spherical(
                m1,
                m2,
                lmax=lmax,
            )

        if data2 is None:
            ell_stack, spectra_stack = jax.lax.map(
                lambda m: _compute((m, None)),
                data1,
            )
        else:
            ell_stack, spectra_stack = jax.lax.map(
                lambda pair: _compute(pair),
                (data1, data2),
            )

        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 2 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber, spectra=spectra, name="cl")

    def __getitem__(self, key) -> "SphericalDensity":
        """
        Index into batched SphericalDensity.

        Enables extracting individual shells or slices from lightcone batches.

        Parameters
        ----------
        key : int or slice
            Index or slice to extract from batch dimension.

        Returns
        -------
        SphericalDensity
            New SphericalDensity with indexed array and scale_factors.

        Raises
        ------
        ValueError
            If array is not batched (ndim < 2).

        Examples
        --------
        >>> spherical_lightcone.array.shape  # (10, 49152)
        >>> shell = spherical_lightcone[0]
        >>> shell.array.shape  # (49152,)
        >>> reversed_lightcone = spherical_lightcone[::-1]
        >>> reversed_lightcone.array.shape  # (10, 49152)
        """
        if self.array.ndim < 2:
            raise ValueError(
                f"Indexing only supported for batched SphericalDensity (2D array), "
                f"got array with {self.array.ndim} dimensions")

        indexed_array = self.array[key]
        indexed_scale_factors = self.scale_factors[key]

        # Reconstruct via metadata-aware factory while updating scale_factors.
        temp_field = DensityField(
            array=jnp.zeros(self.mesh_size),
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            halo_size=self.halo_size,
            z_source=self.z_source,
            status=FieldStatus.RAW,
            scale_factors=indexed_scale_factors,
        )

        return type(self).FromDensityMetadata(
            array=indexed_array,
            density_field=temp_field,
            status=self.status,
            scale_factors=indexed_scale_factors,
        )

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        apply_log: bool = True,
        show_colorbar: bool = True,
    ):
        """
        Visualize one or more spherical maps using ``healpy.mollview``.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError(
                "Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 1:
            data = data[None, ...]
        elif data.ndim < 1:
            raise ValueError("SphericalDensity array rank must be ≥1.")
        else:
            data = data.reshape((-1, data.shape[-1]))

        n_maps = data.shape[0]

        def _flatten_axes(axes_obj):
            if axes_obj is None:
                return None
            if isinstance(axes_obj, np.ndarray):
                return axes_obj.ravel()
            if isinstance(axes_obj, Sequence):
                return np.array(axes_obj, dtype=object).ravel()
            return np.array([axes_obj], dtype=object)

        axes_flat = _flatten_axes(ax)

        if axes_flat is None:
            requested_cols = ncols or 3
            ncols_eff = max(1, min(requested_cols, n_maps))
            nrows_eff = ceil(n_maps / ncols_eff)
            if figsize is None:
                figsize = (4 * ncols_eff, 4 * nrows_eff)
            fig, axes_created = plt.subplots(nrows_eff,
                                             ncols_eff,
                                             figsize=figsize,
                                             squeeze=False)
            axes_flat = axes_created.ravel()
        else:
            if axes_flat.size < n_maps:
                raise ValueError(
                    "Provided axes array is too small for number of maps")
            fig = axes_flat[0].get_figure()
            axes_created = None  # using provided axes; keep their existing grid

        def _sub_from_ax(ax_obj):
            sp = ax_obj.get_subplotspec()
            gs = sp.get_gridspec()
            row = sp.rowspan.start
            col = sp.colspan.start
            return gs.nrows, gs.ncols, row * gs.ncols + col + 1

        def _attach_delegate(ax_obj, delegate):
            ax_obj._healpy_delegate = delegate  # type: ignore[attr-defined]
            ax_obj.set_title = delegate.set_title  # type: ignore[assignment]
            ax_obj.get_title = delegate.get_title  # type: ignore[assignment]

        for idx, ax_i in enumerate(axes_flat):
            if idx < n_maps:
                title = titles[idx] if titles and idx < len(titles) else ""
                map_np = np.asarray(data[idx])
                map_np = np.log10(map_np + 1) if apply_log else map_np

                ax_i.axis("off")
                sub = _sub_from_ax(ax_i)
                delegate = hp.mollview(
                    map_np,
                    fig=fig,
                    sub=sub,
                    cmap=cmap,
                    title=title,
                    bgcolor=(0.0, ) * 4,
                    cbar=show_colorbar,
                    min=0,
                    max=np.percentile(map_np[map_np > 0], 95) if np.any(
                        map_np > 0) else np.max(map_np),
                )
                if delegate is None:
                    delegate = next(
                        (ax for ax in fig.axes
                         if isinstance(ax, hp.projaxes.HpxMollweideAxes)),
                        None,
                    )
                if delegate is None:
                    raise RuntimeError(
                        "healpy.mollview did not return a Mollweide axes.")
                _attach_delegate(ax_i, delegate)
            else:
                ax_i.axis("off")

        axes_out = axes_created
        return fig

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        apply_log: bool = True,
        show_colorbar: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot and display spherical maps using healpy.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError(
                "Cannot plot/show traced arrays. Use outside of jit context.")

        self.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            apply_log=apply_log,
            show_colorbar=show_colorbar,
            **kwargs,
        )
        plt.show()

    def ud_sample(self, new_nside):
        """
        Change HEALPix resolution using jax_healpy.ud_grade.

        Supports batched input automatically: (n_sources, npix) → (n_sources, new_npix)

        Parameters
        ----------
        new_nside : int
            New HEALPix nside parameter (must be power of 2)

        Returns
        -------
        SphericalDensity
            Resampled spherical map (functional, immutable)

        Examples
        --------
        >>> density_hires = density.ud_sample(new_nside=256)

        Notes
        -----
        Uses jax_healpy.ud_grade which automatically handles both single
        and batched arrays.
        """
        # jax_healpy.ud_grade handles both single and batched automatically!
        resampled = jhp.ud_grade(self.array, new_nside)

        return self.replace(array=resampled, nside=new_nside)

    @classmethod
    def stack(cls, fields: Sequence["SphericalDensity"]) -> "SphericalDensity":
        """
        Stack multiple spherical maps along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError(
                "SphericalDensity.stack requires at least one field.")
        return jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *field_list,
        )


def stack(fields: Sequence[DensityField]) -> DensityField:
    """
    Stack a collection of FlatDensity or SphericalDensity instances along axis 0.
    """
    field_list = tuple(fields)
    if not field_list:
        raise ValueError("stack requires at least one field.")
    first = field_list[0]
    if isinstance(first, FlatDensity):
        return FlatDensity.stack(field_list)  # type: ignore[arg-type]
    if isinstance(first, SphericalDensity):
        return SphericalDensity.stack(field_list)  # type: ignore[arg-type]
    if isinstance(first, DensityField):
        return DensityField.stack(field_list)  # type: ignore[arg-type]
    raise TypeError(
        "stack currently supports FlatDensity or SphericalDensity inputs.")
