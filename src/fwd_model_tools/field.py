from __future__ import annotations

from enum import Enum
from functools import partial
from math import ceil
from typing import Any, Iterable, Literal, Sequence, Tuple

import healpy as hp
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.distributed import uniform_particles
from jaxpm.painting import (cic_paint, cic_paint_2d, cic_paint_dx, cic_read,
                            cic_read_dx)
from jaxpm.spherical import paint_particles_spherical
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fwd_model_tools._src._painting import (_single_paint, _single_paint_2d,
                                            _single_paint_spherical)
from fwd_model_tools._src.core import AbstractField
from jaxtyping import Array
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


def _ensure_tuple3(name: str, values: Iterable[Any], *,
                   cast: type) -> tuple[Any, Any, Any]:
    seq = tuple(values)
    if len(seq) != 3:
        raise ValueError(f"{name} must be a tuple of length 3, got {seq}")
    return tuple(cast(v) for v in seq)


def _ensure_tuple2(name: str, values: Iterable[Any], *,
                   cast: type) -> tuple[Any, Any]:
    seq = tuple(values)
    if len(seq) != 2:
        raise ValueError(f"{name} must be a tuple of length 2, got {seq}")
    return tuple(cast(v) for v in seq)


def _normalize_halo_size(value: int | Tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        left, right = _ensure_tuple2("halo_size", value, cast=int)
    else:
        left = right = int(value)
    if left < 0 or right < 0:
        raise ValueError("halo_size entries must be non-negative")
    return left, right


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return ivalue


def _optional_tuple2_positive(
        name: str, value: Tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    x, y = _ensure_tuple2(name, value, cast=int)
    if x <= 0 or y <= 0:
        raise ValueError(f"{name} entries must be positive, got {(x, y)}")
    return x, y


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
        "halo_size",
        "status",
        "scale_factors",
    )

    STATUS_ENUM = FieldStatus

    def __init__(
        self,
        *,
        array: jax.Array,
        mesh_size: Tuple[int, int, int],
        box_size: Tuple[float, float, float],
        observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sharding: Any | None = None,
        nside: int | None = None,
        flatsky_npix: Tuple[int, int] | None = None,
        halo_size: int | Tuple[int, int] = 0,
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
        self.halo_size = halo_size
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
            self.halo_size,
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
            halo_size,
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
            halo_size=halo_size,
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

    def with_array(self, array: jax.Array) -> "DensityField":
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
            "halo_size": self.halo_size,
            "status": self.status,
            "scale_factors": self.scale_factors,
        }
        allowed_keys = set(params.keys())
        unknown = set(updates) - allowed_keys
        if unknown:
            raise TypeError(f"Unknown DensityField attribute(s): {unknown}")
        params.update(updates)
        return type(self)(**params)

    def set_scale_factors(self, scale_factors: float) -> None:
        self.scale_factors = scale_factors

    # -------------------------------------------------------- power-spectrum API
    def compute_power_spectrum(self, k: jax.Array | jnp.ndarray):
        """
        Placeholder for 3D power spectrum evaluation.
        """
        raise NotImplementedError(
            "3D power spectrum computation not implemented yet.")

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

        return FlatDensity(
            array=projection,
            density_field=projected_field,
            status=DensityStatus.PROJECTED_DENSITY,
        )

    def show(
        self,
        nz_slices: int = 10,
        *,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
    ) -> None:
        """
        Paint and display a 2D projection of the density field.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).

        Parameters
        ----------
        nz_slices : int, default=10
            Number of z-slices to sum.
        cmap : str, default="magma"
            Colormap for visualization.
        figsize : tuple, optional
            Figure size (width, height).
        title : str, optional
            Plot title.
        show_colorbar : bool, default=True
            Whether to add a colorbar to each subplot.
        show_ticks : bool, default=True
            Whether to display axis ticks.
        """
        import jax.core
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        flat = self.project(nz_slices=nz_slices)
        flat.show(
            cmap=cmap,
            figsize=figsize,
            titles=[title] if title else None,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
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
                f"got array with {self.array.ndim} dimensions"
            )

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
            halo_size=self.halo_size,
            status=self.status,
            scale_factors=indexed_scale_factors,
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
        array: jax.Array,
        mesh_size: Tuple[int, int, int],
        box_size: Tuple[float, float, float],
        observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sharding: Any | None = None,
        nside: int | None = None,
        flatsky_npix: Tuple[int, int] | None = None,
        halo_size: int | Tuple[int, int] = 0,
        status: FieldStatus = FieldStatus.RAW,
        scale_factors: float = 1.0,
    ):
        if isinstance(array , Array):
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
            halo_size=halo_size,
            status=status,
            scale_factors=scale_factors,
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

        return ParticleField(
            array=indexed_array,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            halo_size=self.halo_size,
            status=self.status,
            scale_factors=indexed_scale_factors,
        )

    @partial(jax.jit, static_argnames=['mode', 'weights', 'chunk_size', 'batch_size'])
    def paint(
        self,
        mode: PaintMode = "relative",
        *,
        mesh: jax.Array | None = None,
        weights: jax.Array | float = 1.0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_size: int | None = None,
    ) -> DensityField:
        """
        Paint particles onto a 3D density mesh using CIC interpolation.

        Supports both batched (5D) and single (4D) particle arrays.

        Parameters
        ----------
        mode : PaintMode, default="relative"
            "relative" for displacement-based painting, "absolute" for position-based.
        mesh : jax.Array, optional
            Pre-allocated mesh for absolute mode.
        weights : jax.Array | float, default=1.0
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
        density_mesh: DensityField | jax.Array,
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

    @partial(jax.jit, static_argnames=['mode', 'weights', 'density_plane_width', 'batch_size'])
    def paint_2d(
        self,
        center: Float | jax.Array,
        *,
        density_plane_width: Float | None = None,
        weights: jax.Array | float | None = None,
        mode: PaintMode = "relative",
        batch_size: int | None = None,
    ) -> "FlatDensity":
        """
        Project particles onto a flat-sky grid using CIC painting.

        Supports batched input: if self.array has shape (N, X, Y, Z, 3) and center
        is an array of shape (N,), returns FlatDensity with shape (N, flatsky_npix[0], flatsky_npix[1]).

        Parameters
        ----------
        center : Float | jax.Array
            Center of density plane(s) in Mpc. Scalar for single shell, array for batched.
        density_plane_width : Float, optional
            Physical width of density plane in Mpc. Defaults to max_comoving_radius / nb_shells.
        weights : jax.Array | float, optional
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
            density_plane_width = self.density_width(nb_shells) if density_plane_width is None else density_plane_width
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            if density_plane_width is None:
                raise ValueError("density_plane_width must be specified for single shell")
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

        painted = jax.lax.map(paint_fn, (data, center_arr), batch_size=batch_size)
        painted = painted.squeeze()

        return FlatDensity(
            array=painted,
            density_field=self,
            status=DensityStatus.LIGHTCONE,
        )

    @partial(jax.jit, static_argnames=['mode', 'scheme', 'weights', 'density_plane_width', 'kernel_width_arcmin', 'smoothing_interpretation', 'paint_nside', 'ud_grade_power', 'ud_grade_order_in', 'ud_grade_order_out', 'ud_grade_pess', 'batch_size'])
    def paint_spherical(
        self,
        center: Float | jax.Array,
        *,
        mode: PaintMode = "relative",
        scheme: SphericalScheme = "rbf_neighbor",
        weights: jax.Array | None = None,
        density_plane_width: Float | None = None,
        kernel_width_arcmin: float | None = None,
        smoothing_interpretation: str = "fwhm",
        paint_nside: int | None = None,
        ud_grade_power: float = 0.0,
        ud_grade_order_in: str = "RING",
        ud_grade_order_out: str = "RING",
        ud_grade_pess: bool = False,
        batch_size: int | None = None,
    ) -> "SphericalDensity":
        """
        Paint particles onto a HEALPix grid using spherical painting.

        Supports batched input: if self.array has shape (N, X, Y, Z, 3) and center
        is an array of shape (N,), returns SphericalDensity with shape (N, npix).

        Parameters
        ----------
        center : Float | jax.Array
            Center of density shell(s) in Mpc. Scalar for single shell, array for batched.
        mode : PaintMode, default="relative"
            "relative" for displacement-based painting, "absolute" for position-based.
        scheme : SphericalScheme, default="rbf_neighbor"
            Painting method: "ngp", "bilinear", or "rbf_neighbor".
        weights : jax.Array, optional
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
            density_plane_width = self.density_width(nb_shells) if density_plane_width is None else density_plane_width
        elif data.ndim == 4:
            data = data[None, ...]
            center_arr = center_arr[None, ...]
            if density_plane_width is None:
                raise ValueError("density_plane_width must be specified for single shell")
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

        painted = jax.lax.map(paint_fn, (data, center_arr), batch_size=batch_size)
        painted = painted.squeeze()

        return SphericalDensity(
            array=painted,
            density_field=self,
            status=DensityStatus.LIGHTCONE,
        )


@partial(jax.jit, static_argnames=['status'])
def particle_from_density(array: jax.Array, reference: DensityField, scale_factors: float = None, status: FieldStatus = None
                          ) -> ParticleField:
    """
    Create a ParticleField from an array, inheriting metadata from a reference DensityField.

    This helper function is used to construct ParticleField objects while preserving
    all metadata (mesh_size, box_size, observer_position, sharding, etc.) from a
    reference field.

    Parameters
    ----------
    array : jax.Array
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
    return ParticleField(
        array=array,
        mesh_size=reference.mesh_size,
        box_size=reference.box_size,
        observer_position=reference.observer_position,
        sharding=reference.sharding,
        nside=reference.nside,
        flatsky_npix=reference.flatsky_npix,
        halo_size=reference.halo_size,
        status=status if status is not None else reference.status,
        scale_factors=scale_factors if scale_factors is not None else reference.scale_factors,)


@jax.tree_util.register_pytree_node_class
class FlatDensity(DensityField):
    """Flat-sky (2D) density or shear maps derived from volumetric simulations."""

    __slots__ = ()
    STATUS_ENUM = DensityStatus

    def __init__(
        self,
        *,
        array: jax.Array,
        density_field: DensityField,
        status: DensityStatus = DensityStatus.LIGHTCONE,
    ):
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
        super().__init__(
            array=arr,
            mesh_size=density_field.mesh_size,
            box_size=density_field.box_size,
            observer_position=density_field.observer_position,
            sharding=density_field.sharding,
            nside=density_field.nside,
            flatsky_npix=density_field.flatsky_npix,
            halo_size=density_field.halo_size,
            status=status,
            scale_factors=density_field.scale_factors,
        )

    def replace(self, **updates: Any) -> "Self":
        """Override replace to handle special __init__ signature."""
        params = {
            "array": self.array,
            "mesh_size": self.mesh_size,
            "box_size": self.box_size,
            "observer_position": self.observer_position,
            "sharding": self.sharding,
            "nside": self.nside,
            "flatsky_npix": self.flatsky_npix,
            "halo_size": self.halo_size,
            "status": self.status,
            "scale_factors": self.scale_factors,
        }
        allowed_keys = set(params.keys())
        unknown = set(updates) - allowed_keys
        if unknown:
            raise TypeError(f"Unknown {type(self).__name__} attribute(s): {unknown}")
        params.update(updates)

        # Create using object.__new__ to bypass __init__
        instance = object.__new__(type(self))
        for key, value in params.items():
            setattr(instance, key, value)
        return instance

    def tree_flatten(self):
        # Use parent's tree_flatten but store status as string to avoid enum mismatch
        children, aux_data = super().tree_flatten()
        # aux_data is (mesh_size, box_size, observer_position, sharding, nside, flatsky_npix, halo_size, status)
        aux_list = list(aux_data)
        aux_list[-1] = str(self.status.value)  # Convert status to string
        return children, tuple(aux_list)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # aux_data has status as string, convert back to DensityStatus
        (
            mesh_size,
            box_size,
            observer_position,
            sharding,
            nside,
            flatsky_npix,
            halo_size,
            status_str,
        ) = aux_data
        array, scale_factors = children

        # Create instance bypassing __init__ to avoid validation issues
        instance = object.__new__(cls)
        instance.array = array
        instance.mesh_size = mesh_size
        instance.box_size = box_size
        instance.observer_position = observer_position
        instance.sharding = sharding
        instance.nside = nside
        instance.flatsky_npix = flatsky_npix
        instance.halo_size = halo_size
        instance.status = DensityStatus(status_str)
        instance.scale_factors = scale_factors
        return instance

    def compute_power_spectrum(self, ells: jax.Array | jnp.ndarray):
        raise NotImplementedError(
            "Flat-sky power spectrum computation not implemented yet.")

    def plot(
        self,
        *,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Sequence[str] | None = None,
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
        import jax.core
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 2:
            data = data[None, ...]
        elif data.ndim != 3:
            raise ValueError(
                "FlatDensity.plot expects array shape (ny, nx) or (n, ny, nx)."
            )

        n_maps = data.shape[0]
        requested_cols = ncols or 3
        ncols = max(1, min(requested_cols, n_maps))
        nrows = ceil(n_maps / ncols)
        if figsize is None:
            figsize = (5 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for idx in range(nrows * ncols):
            ax = axes[idx // ncols, idx % ncols]
            if idx < n_maps:
                to_plot = jnp.log10(data[idx] + 1) if apply_log else data[idx]
                im = ax.imshow(to_plot, origin="lower", cmap=cmap)
                if show_colorbar:
                    # Keep colorbars narrow so they do not fill the figure.
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.03)
                    fig.colorbar(im, cax=cax)
                if not show_ticks:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if titles and idx < len(titles):
                    ax.set_title(titles[idx])
            else:
                ax.axis("off")
        fig.tight_layout()
        return fig, axes

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
                f"got array with {self.array.ndim} dimensions"
            )

        indexed_array = self.array[key]
        indexed_scale_factors = self.scale_factors[key]

        # Create a temporary DensityField to pass to FlatDensity.__init__
        temp_field = DensityField(
            array=jnp.zeros(self.mesh_size),  # Dummy array, not used
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            halo_size=self.halo_size,
            status=FieldStatus.RAW,
            scale_factors=indexed_scale_factors,
        )

        return FlatDensity(
            array=indexed_array,
            density_field=temp_field,
            status=self.status,
        )

    def show(
        self,
        *,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Sequence[str] | None = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
    ) -> None:
        """
        Plot and display flat-sky maps using matplotlib.

        Parameters mirror :meth:`FlatDensity.plot`.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).
        """
        import jax.core
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        self.plot(
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
        )
        plt.show()

    @classmethod
    def stack(cls, fields: Sequence["FlatDensity"]) -> "FlatDensity":
        """
        Stack multiple FlatDensity instances along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("FlatDensity.stack requires at least one field.")
        ref = field_list[0]
        attrs = (
            "mesh_size",
            "box_size",
            "observer_position",
            "sharding",
            "flatsky_npix",
            "halo_size",
            "status",
        )
        for fld in field_list[1:]:
            for attr in attrs:
                if getattr(fld, attr) != getattr(ref, attr):
                    raise ValueError(
                        f"Cannot stack FlatDensity objects with differing {attr}."
                    )
        stacked = jnp.stack([fld.array for fld in field_list], axis=0)
        return cls(array=stacked, density_field=ref, status=ref.status)


@jax.tree_util.register_pytree_node_class
class SphericalDensity(DensityField):
    """Spherical (HEALPix) density or shear maps produced from simulations."""

    __slots__ = ()
    STATUS_ENUM = DensityStatus

    def __init__(
        self,
        *,
        array: jax.Array,
        density_field: DensityField,
        status: DensityStatus = DensityStatus.LIGHTCONE,
    ):
        if density_field.nside is None:
            raise ValueError("SphericalDensity requires `nside`.")
        arr = jnp.asarray(array)
        npix = jhp.nside2npix(density_field.nside)
        if arr.shape[-1] != npix:
            raise ValueError(
                f"Array last dimension {arr.shape[-1]} does not match HEALPix npix "
                f"{npix} for nside {density_field.nside}.")

        super().__init__(
            array=arr,
            mesh_size=density_field.mesh_size,
            box_size=density_field.box_size,
            observer_position=density_field.observer_position,
            sharding=density_field.sharding,
            nside=density_field.nside,
            flatsky_npix=density_field.flatsky_npix,
            halo_size=density_field.halo_size,
            status=status,
            scale_factors=density_field.scale_factors,
        )

    def replace(self, **updates: Any) -> "Self":
        """Override replace to handle special __init__ signature."""
        params = {
            "array": self.array,
            "mesh_size": self.mesh_size,
            "box_size": self.box_size,
            "observer_position": self.observer_position,
            "sharding": self.sharding,
            "nside": self.nside,
            "flatsky_npix": self.flatsky_npix,
            "halo_size": self.halo_size,
            "status": self.status,
            "scale_factors": self.scale_factors,
        }
        allowed_keys = set(params.keys())
        unknown = set(updates) - allowed_keys
        if unknown:
            raise TypeError(f"Unknown {type(self).__name__} attribute(s): {unknown}")
        params.update(updates)

        # Create using object.__new__ to bypass __init__
        instance = object.__new__(type(self))
        for key, value in params.items():
            setattr(instance, key, value)
        return instance

    def tree_flatten(self):
        # Use parent's tree_flatten but store status as string to avoid enum mismatch
        children, aux_data = super().tree_flatten()
        # aux_data is (mesh_size, box_size, observer_position, sharding, nside, flatsky_npix, halo_size, status)
        aux_list = list(aux_data)
        aux_list[-1] = str(self.status.value)  # Convert status to string
        return children, tuple(aux_list)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # aux_data has status as string, convert back to DensityStatus
        (
            mesh_size,
            box_size,
            observer_position,
            sharding,
            nside,
            flatsky_npix,
            halo_size,
            status_str,
        ) = aux_data
        array, scale_factors = children

        # Create instance bypassing __init__ to avoid validation issues
        instance = object.__new__(cls)
        instance.array = array
        instance.mesh_size = mesh_size
        instance.box_size = box_size
        instance.observer_position = observer_position
        instance.sharding = sharding
        instance.nside = nside
        instance.flatsky_npix = flatsky_npix
        instance.halo_size = halo_size
        instance.status = DensityStatus(status_str)
        instance.scale_factors = scale_factors
        return instance

    def compute_power_spectrum(self, ells: jax.Array | jnp.ndarray):
        raise NotImplementedError(
            "Spherical power spectrum computation not implemented yet.")

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
                f"got array with {self.array.ndim} dimensions"
            )

        indexed_array = self.array[key]
        indexed_scale_factors = self.scale_factors[key]

        # Create a temporary DensityField to pass to SphericalDensity.__init__
        temp_field = DensityField(
            array=jnp.zeros(self.mesh_size),  # Dummy array, not used
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            sharding=self.sharding,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            halo_size=self.halo_size,
            status=FieldStatus.RAW,
            scale_factors=indexed_scale_factors,
        )

        return SphericalDensity(
            array=indexed_array,
            density_field=temp_field,
            status=self.status,
        )

    def plot(
        self,
        *,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Sequence[str] | None = None,
        apply_log: bool = True,
    ):
        """
        Visualize one or more spherical maps using ``healpy.mollview``.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).
        """
        import jax.core
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 1:
            data = data[None, ...]
        elif data.ndim < 1:
            raise ValueError("SphericalDensity array rank must be ≥1.")
        else:
            data = data.reshape((-1, data.shape[-1]))

        n_maps = data.shape[0]
        requested_cols = ncols or 3
        ncols = max(1, min(requested_cols, n_maps))
        nrows = ceil(n_maps / ncols)
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)

        fig = plt.figure(figsize=figsize)
        for idx in range(n_maps):
            title = titles[idx] if titles and idx < len(titles) else ""
            map_np = np.asarray(data[idx])
            map_np = np.log10(map_np + 1) if apply_log else map_np
            hp.mollview(map_np,
                        fig=fig,
                        sub=(nrows, ncols, idx + 1),
                        cmap=cmap,
                        title=title,
                        bgcolor=(0.0, ) * 4,
                        cbar=True,
                        min=0,
                        max=np.percentile(map_np[map_np > 0], 95) if np.any(
                            map_np > 0) else np.max(map_np))

        return fig

    def show(
        self,
        *,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Sequence[str] | None = None,
    ) -> None:
        """
        Plot and display spherical maps using healpy.

        Raises
        ------
        ValueError
            If called within a jit context (i.e., array is traced).
        """
        import jax.core
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        self.plot(cmap=cmap, figsize=figsize, ncols=ncols, titles=titles)
        plt.show()

    @classmethod
    def stack(cls, fields: Sequence["SphericalDensity"]) -> "SphericalDensity":
        """
        Stack multiple spherical maps along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError(
                "SphericalDensity.stack requires at least one field.")
        ref = field_list[0]
        attrs = (
            "mesh_size",
            "box_size",
            "observer_position",
            "sharding",
            "nside",
            "halo_size",
            "status",
        )
        for fld in field_list[1:]:
            for attr in attrs:
                if getattr(fld, attr) != getattr(ref, attr):
                    raise ValueError(
                        f"Cannot stack SphericalDensity objects with differing {attr}."
                    )
        stacked = jnp.stack([fld.array for fld in field_list], axis=0)
        return cls(array=stacked, density_field=ref, status=ref.status)


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
    raise TypeError(
        "stack currently supports FlatDensity or SphericalDensity inputs.")
