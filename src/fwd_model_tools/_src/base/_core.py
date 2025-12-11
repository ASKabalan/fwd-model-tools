"""
Core base classes for field objects.

This module contains the foundational abstract classes that define the
interface for all field types in fwd_model_tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._checks import (
    _ensure_tuple3,
    _normalize_halo_size,
    _optional_positive_int,
    _optional_tuple2_positive,
)
from ._enums import FieldStatus, PhysicalUnit


class AbstractPytree(ABC):
    """
    Minimal base class capturing the array payload shared by all field objects.
    """

    __slots__ = ("array",)
    __array_priority__ = 1000

    def __init__(self, *, array: jax.Array):
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shorthand for ``array.shape``."""
        return tuple(self.array.shape)

    @property
    def dtype(self) -> jnp.dtype:
        """Shorthand for ``array.dtype``."""
        return self.array.dtype

    @abstractmethod
    def replace(self, **updates: Any) -> AbstractField:
        """Return a copy with the provided attributes replaced."""

    # ----------------------------------------------------------- math helpers
    def _coerce_other(self, other: Any) -> Any:
        return other.array if isinstance(other, AbstractField) else other

    def _binary_op(self, other: Any, op) -> AbstractField:
        return self.replace(array=op(self.array, self._coerce_other(other)))

    def _binary_rop(self, other: Any, op) -> AbstractField:
        return self.replace(array=op(self._coerce_other(other), self.array))

    def __add__(self, other: Any):
        return self._binary_op(other, lambda x, y: x + y)

    def __radd__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x + y)

    def __sub__(self, other: Any):
        return self._binary_op(other, lambda x, y: x - y)

    def __rsub__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x - y)

    def __mul__(self, other: Any):
        return self._binary_op(other, lambda x, y: x * y)

    def __rmul__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x * y)

    def __truediv__(self, other: Any):
        return self._binary_op(other, lambda x, y: x / y)

    def __rtruediv__(self, other: Any):
        return self._binary_rop(other, lambda x, y: x / y)

    def __neg__(self):
        return self.replace(array=-self.array)

    def __pos__(self):
        return self

    def min(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.min(self.array, *args, **kwargs)``."""
        return self.replace(array=jnp.min(self.array, *args, **kwargs))

    def max(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.max(self.array, *args, **kwargs)``."""
        return self.replace(array=jnp.max(self.array, *args, **kwargs))

    def mean(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.mean(self.array, *args, **kwargs)``."""
        return self.replace(array=jnp.mean(self.array, *args, **kwargs))

    def std(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.std(self.array, *args, **kwargs)``."""
        return self.replace(array=jnp.std(self.array, *args, **kwargs))

    def transpose(self, *axes: int) -> Self:
        """Shorthand for ``jnp.transpose(self.array, *axes)``."""
        return self.replace(array=jnp.transpose(self.array, *axes))

    @property
    def T(self) -> Self:
        """Shorthand for ``jnp.transpose(self.array)``."""
        return self.replace(array=jnp.transpose(self.array))

    def apply_fn(self, fn, *args, **kwargs) -> Self:
        """
        Apply a function to the underlying array.

        Parameters
        ----------
        fn : callable
            Function to apply to the array.
        *args
            Additional positional arguments to pass to `fn`.
        **kwargs
            Additional keyword arguments to pass to `fn`.

        Returns
        -------
        Self
            New instance with the transformed array.
        """
        return self.replace(array=fn(self.array, *args, **kwargs))


@jax.tree_util.register_pytree_node_class
class AbstractField(AbstractPytree):
    """
    PyTree container for volumetric simulation arrays and their static metadata.
    """

    __slots__ = (
        # Main array object inherited from AbstractField
        "array",
        # Simulation configuration
        "mesh_size",
        "box_size",
        "observer_position",
        "sharding",
        "halo_size",
        # Lightcone geometry and metadata
        "nside",
        "flatsky_npix",
        "field_size",
        # Dynamic metadata related to redshift of the field
        "z_sources",
        "scale_factors",
        "comoving_centers",
        "density_width",
        # Unit and status metadata
        "unit",
        "status",
    )

    STATUS_ENUM = FieldStatus

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
        # Unit metadata
        status: FieldStatus = FieldStatus.UNKNOWN,
        unit: PhysicalUnit = PhysicalUnit.INVALID_UNIT,
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
        self.halo_size = halo_size
        # Lightcone geometry and metadata
        self.nside = nside
        self.flatsky_npix = flatsky_npix
        self.field_size = field_size
        self.z_sources = z_sources
        self.scale_factors = scale_factors
        self.comoving_centers = comoving_centers
        self.density_width = density_width
        # Unit and status metadata
        self.status = self._coerce_status(status)
        self.unit = unit

    # ------------------------------------------------------------------ PyTree
    def tree_flatten(self):
        children = (
            self.array,
            self.z_sources,
            self.scale_factors,
            self.comoving_centers,
            self.density_width,
        )
        # static metadata
        aux_data = (
            self.mesh_size,
            self.box_size,
            self.observer_position,
            self.sharding,
            self.nside,
            self.flatsky_npix,
            self.field_size,
            self.halo_size,
            self.status,
            self.unit,
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
            status,
            unit,
        ) = aux_data
        (
            array,
            z_sources,
            scale_factors,
            comoving_centers,
            density_width,
        ) = children

        return cls(
            array=array,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            sharding=sharding,
            halo_size=halo_size,
            #
            nside=nside,
            flatsky_npix=flatsky_npix,
            field_size=field_size,
            z_sources=z_sources,
            scale_factors=scale_factors,
            comoving_centers=comoving_centers,
            density_width=density_width,
            #
            status=status,
            unit=unit,
        )

    @classmethod
    def _coerce_status(cls, status):
        enum_cls = cls.STATUS_ENUM
        return enum_cls(status)

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

    @classmethod
    def stack(cls, fields: Sequence[Self]) -> Self:
        """
        Stack multiple Field instances along axis 0.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("stack requires at least one field.")
        return jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *field_list,
        )

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array: Array,
        field: AbstractField,
        # Dynamic metadata related to redshift of the field
        z_sources: Any | None = None,
        scale_factors: float | None = None,
        comoving_centers: float | None = None,
        density_width: float | None = None,
        # Unit and status metadata
        status: FieldStatus | None = None,
        unit: PhysicalUnit | None = None,
    ) -> Self:
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
        z_sources : any, optional
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
            mesh_size=field.mesh_size,
            box_size=field.box_size,
            observer_position=field.observer_position,
            sharding=field.sharding,
            halo_size=field.halo_size,
            # Lightcone geometry and metadata
            nside=field.nside,
            flatsky_npix=field.flatsky_npix,
            field_size=field.field_size,
            # Dynamic metadata related to redshift of the field
            z_sources=(z_sources if z_sources is not None else field.z_sources),
            scale_factors=(scale_factors if scale_factors is not None else field.scale_factors),
            comoving_centers=(comoving_centers if comoving_centers is not None else field.comoving_centers),
            density_width=(density_width if density_width is not None else field.density_width),
            # Unit and status metadata
            status=status if status is not None else field.status,
            unit=unit if unit is not None else field.unit,
        )

    def replace(self, **updates: Any) -> Self:
        params = {
            "array": self.array,
            "mesh_size": self.mesh_size,
            "box_size": self.box_size,
            "observer_position": self.observer_position,
            "sharding": self.sharding,
            "halo_size": self.halo_size,
            #
            "nside": self.nside,
            "flatsky_npix": self.flatsky_npix,
            "field_size": self.field_size,
            #
            "z_sources": self.z_sources,
            "scale_factors": self.scale_factors,
            "comoving_centers": self.comoving_centers,
            "density_width": self.density_width,
            #
            "status": self.status,
            "unit": self.unit,
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

    def block_until_ready(self) -> Self:
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
        classname = type(self).__name__
        return (
            f"{classname}("
            f"array=Array{self.array.shape}, "
            f"mesh_size={self.mesh_size}, "
            f"box_size={self.box_size}, "
            f"observer_position={self.observer_position}, "
            f"sharding={self.sharding}, "
            f"halo_size={self.halo_size}, "
            f"nside={self.nside}, "
            f"flatsky_npix={self.flatsky_npix}, "
            f"field_size={self.field_size}, "
            f"status={self.status.name}, "
            f"unit={self.unit.name})"
        )

    def runtime_inspect(self) -> None:
        """
        Runtime debug helper: prints static metadata, dynamic metadata,
        basic array stats, and sharding info using JAX debug utilities.

        Works both inside and outside of `jax.jit` (using `jax.debug.print`).
        """
        dbg = jax.debug
        classname = type(self).__name__

        # ---- Static metadata (matches aux_data in tree_flatten) ----
        dbg.print(
            "{} static:\n"
            "  mesh_size={}\n"
            "  box_size={}\n"
            "  observer_position={}\n"
            "  sharding={}\n"
            "  nside={}\n"
            "  flatsky_npix={}\n"
            "  field_size={}\n"
            "  halo_size={}\n"
            "  status={}\n"
            "  unit={}",
            classname,
            self.mesh_size,
            self.box_size,
            self.observer_position,
            self.sharding,
            self.nside,
            self.flatsky_npix,
            self.field_size,
            self.halo_size,
            self.status,
            self.unit,
        )

        # ---- Dynamic metadata ----
        dbg.print(
            "{} dynamic:\n  z_sources={}\n  scale_factors={}\n  comoving_centers={}\n  density_width={}",
            classname,
            self.z_sources,
            self.scale_factors,
            self.comoving_centers,
            self.density_width,
        )

        # ---- Array stats ----
        arr = self.array
        mean = jnp.mean(arr)
        std = jnp.std(arr)

        dbg.print(
            "{} array:\n  shape={}\n  dtype={}\n  mean={}\n  std={}",
            classname,
            arr.shape,
            arr.dtype,
            mean,
            std,
        )

        # ---- Sharding info ----
        dbg.print("{} array sharding:", classname)
        dbg.inspect_array_sharding(arr)

    # ------------------------------------------------------------------ Factory
    def __getitem__(self, key) -> Self:
        return jax.tree.map(lambda x: x[key], self)
