from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array , Float

from fwd_model_tools._src._checks import (
    _ensure_tuple2,
    _ensure_tuple3,
    _normalize_halo_size,
    _optional_positive_int,
    _optional_tuple2_positive,
)
from fwd_model_tools._src.core import AbstractField
from fwd_model_tools.power import PowerSpectrum, angular_cl_flat, angular_cl_spherical, coherence
from fwd_model_tools.power import power as power_fn
from fwd_model_tools.power import transfer

__all__ = [
    "FieldStatus",
    "DensityStatus",
    "AbstractField",
    "DensityField",
]



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
        """
        Initialize a density-like field along with its geometric metadata.

        Parameters are kept explicit so downstream painting/analysis utilities
        (flat, spherical, lightcone) have all required context.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> arr = jnp.zeros((16, 16, 16))
        >>> field = DensityField(
        ...     array=arr,
        ...     mesh_size=(16, 16, 16),
        ...     box_size=(200.0, 200.0, 200.0),
        ...     flatsky_npix=(32, 32),
        ... )
        >>> field.mesh_size
        (16, 16, 16)
        """
        super().__init__(array=array)
        mesh_size = _ensure_tuple3("mesh_size", mesh_size, cast=int)
        box_size = _ensure_tuple3("box_size", box_size, cast=float)
        observer_position = _ensure_tuple3("observer_position", observer_position, cast=float)
        if any(not 0.0 <= frac <= 1.0 for frac in observer_position):
            raise ValueError("observer_position entries must lie within [0, 1]")
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
        return tuple(frac * length for frac, length in zip(self.observer_position, self.box_size))

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
            raise ValueError(f"project() expects 3D array or batch of 3D arrays, got shape {data.shape}")

        # Vectorized: sum over last nz_slices along z-axis (last dimension)
        projection = jnp.sum(data[..., -nz_slices:], axis=-1)

        # Update flatsky_npix to match projected shape
        # For 3D: shape is (X, Y), for 4D: shape is (N, X, Y) - take last 2 dims
        flatsky_npix = projection.shape[-2:] if data.ndim == 4 else projection.shape
        projected_field = self.replace(flatsky_npix=flatsky_npix)
        from .lightcone import FlatDensity
        return FlatDensity.FromDensityMetadata(
            array=projection,
            density_field=projected_field,
            status=DensityStatus.PROJECTED_DENSITY,
        )

    # -------------------------------------------------------- power-spectrum API
    def power(
            self,
            mesh2: Optional[DensityField] = None,
            *,
            kedges: Optional[Array | jnp.ndarray] = None,
            multipoles: Optional[Iterable[int]] = 0,
            los: Array | Iterable[float] = (0.0, 0.0, 1.0),
            batch_size: Optional[int] = None,
    ) -> "PowerSpectrum":
        """Compute the 3D matter power spectrum P(k).

        Parameters mirror :func:`fwd_model_tools.power.power`. Any keyword
        arguments are forwarded verbatim to that helper.
        """
        box_shape = tuple(self.box_size)
        multipoles_static = tuple(multipoles) if isinstance(multipoles, (list, tuple)) else multipoles
        los_tuple = None if multipoles_static == 0 else tuple(np.asarray(los, dtype=float))

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 3:
            data1 = jnp.expand_dims(data1, axis=0)
            data2 = jnp.expand_dims(data2, axis=0) if data2 is not None else None
        elif data1.ndim != 4:
            raise ValueError("DensityField.power expects array shape (X,Y,Z) or (B,X,Y,Z)")

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

        k, pk = jax.lax.map(_power_fn, (data1, data2) , batch_size=batch_size)
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
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ):
        """Project along z and plot via :class:`FlatDensity` utilities (batch supported)."""
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot traced arrays. Use outside of jit context.")

        flat = self.project(nz_slices=nz_slices)
        fig, axes = flat.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
            vmin=vmin,
            vmax=vmax,
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
        vmin: float | None = None,
        vmax: float | None = None,
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
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        plt.show()

    def transfer(
        self,
        other: "DensityField",
        *,
        kedges: Optional[Array | jnp.ndarray] = None,
        batch_size: Optional[int] = None,
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
            raise ValueError("DensityField.transfer expects array shape (X,Y,Z) or (B,X,Y,Z)")

        if data2.shape != data1.shape:
            raise ValueError("other array must match shape for transfer")

        k_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = k_stack[0]
        spectra = spectra_stack if self.array.ndim == 4 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber, spectra=spectra, name="transfer")

    def coherence(
        self,
        other: "DensityField",
        *,
        kedges: Optional[Array | jnp.ndarray] = None,
        batch_size: Optional[int] = None,
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
            raise ValueError("DensityField.coherence expects array shape (X,Y,Z) or (B,X,Y,Z)")

        if data2.shape != data1.shape:
            raise ValueError("other array must match shape for coherence")

        k_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = k_stack[0]
        spectra = spectra_stack if self.array.ndim == 4 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber, spectra=spectra, name="coherence")

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
        return (f"{classname}("
                f"array=Array{tuple(self.array.shape)}, "
                f"mesh_size={self.mesh_size}, "
                f"box_size={self.box_size}, "
                f"status={self.status.value}, "
                f"scale_factors_shape={jnp.atleast_1d(self.scale_factors).shape})")

    # ------------------------------------------------------------------ Factory
    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array: Array,
        density_field: "DensityField",
        status: FieldStatus | None = None,
        z_source: Any | None = None,
        scale_factors: float | None = None,
    ) -> "DensityField":
        """
        Rebuild a field of the same class using metadata from a reference DensityField.

        Parameters
        ----------
        array : Array
            New data array to wrap (shape validation is delegated to ``cls.__init__``).
        density_field : DensityField
            Reference field supplying geometric metadata (mesh_size, box_size, nside, etc.).
        status : FieldStatus, optional
            Override status; defaults to ``density_field.status``.
        z_source : any, optional
            Override source redshift(s); defaults to ``density_field.z_source``.
        scale_factors : float or array, optional
            Override scale factors; defaults to ``density_field.scale_factors``.

        Returns
        -------
        DensityField subclass
            Instance of ``cls`` with copied metadata and the provided array.
        """
        return cls(
            array=array,
            mesh_size=density_field.mesh_size,
            box_size=density_field.box_size,
            observer_position=density_field.observer_position,
            sharding=density_field.sharding,
            nside=density_field.nside,
            flatsky_npix=density_field.flatsky_npix,
            field_size=density_field.field_size,
            halo_size=density_field.halo_size,
            z_source=(z_source if z_source is not None else density_field.z_source),
            status=status if status is not None else density_field.status,
            scale_factors=(scale_factors if scale_factors is not None else density_field.scale_factors),
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
        return jax.tree.map(lambda x: x[key], self)

    @classmethod
    def stack(cls, fields: Sequence["DensityField"]) -> "DensityField":
        """
        Stack multiple DensityField instances along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("stack requires at least one field.")
        return jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *field_list,
        )
