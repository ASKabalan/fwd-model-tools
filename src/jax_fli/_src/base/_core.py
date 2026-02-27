"""
Core base classes for field objects.

This module contains the foundational abstract classes that define the
interface for all field types in fwd_model_tools.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from typing_extensions import Self

from ._enums import FieldStatus, PhysicalUnit


class AbstractPytree(eqx.Module):
    """
    Minimal base class capturing the array payload shared by all field objects.
    Inherits from eqx.Module, so it is an immutable PyTree by default.
    """

    array: Array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shorthand for ``array.shape``."""
        return tuple(self.array.shape)

    @property
    def ndim(self) -> int:
        """Shorthand for ``array.ndim``."""
        return self.array.ndim

    @property
    def size(self) -> int:
        """Shorthand for ``array.size``."""
        return self.array.size

    @property
    def dtype(self) -> jnp.dtype:
        """Shorthand for ``array.dtype``."""
        return self.array.dtype

    def __len__(self) -> int:
        """Shorthand for ``len(array)``."""
        return len(self.array)

    def __iter__(self):
        """
        Iterate over the first dimension of the underlying array.
        Yields slices wrapped in the same class via __getitem__.
        """
        for i in range(len(self)):
            yield self[i]

    def replace(self, **kwargs: Any) -> Self:
        """
        Return a copy with the provided attributes replaced.
        Uses equinox.tree_at under the hood.
        """
        if not kwargs:
            return self
        return dataclasses.replace(self, **kwargs)

    # ----------------------------------------------------------- math helpers
    def _coerce_other(self, other: Any) -> Any:
        return other.array if isinstance(other, AbstractPytree) else other

    def _binary_op(self, other: Any, op) -> Self:
        return self.replace(array=op(self.array, self._coerce_other(other)))

    def _binary_rop(self, other: Any, op) -> Self:
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

    def __abs__(self):
        return self.replace(array=jnp.abs(self.array))

    # DO NOT IMPLEMENT THIS
    # IT BREAKS EQUINOX OMEGA
    # TODO: implemented and make exception of other is of type Omega
    # def __pow__(self, other):
    #    return self.replace(array=self.array**other)

    def min(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.min(self.array, *args, **kwargs)``."""
        return jnp.min(self.array, *args, **kwargs)

    def max(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.max(self.array, *args, **kwargs)``."""
        return jnp.max(self.array, *args, **kwargs)

    def mean(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.mean(self.array, *args, **kwargs)``."""
        return jnp.mean(self.array, *args, **kwargs)

    def std(self, *args, **kwargs) -> Self:
        """Shorthand for ``jnp.std(self.array, *args, **kwargs)``."""
        return jnp.std(self.array, *args, **kwargs)

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

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.array)

    def __jax_array__(self, dtype=None) -> Array:
        return jnp.asarray(self.array)


class AbstractField(AbstractPytree):
    """
    PyTree container for volumetric simulation arrays and their static metadata.
    Inherits from eqx.Module via AbstractPytree.
    """

    # Mandatory metadata
    mesh_size: tuple[int, int, int] = eqx.field(static=True)
    box_size: tuple[float, float, float] = eqx.field(static=True)

    # Dynamic metadata (JAX traced) - array inherited from AbstractPytree
    z_sources: Any | None = eqx.field(default=None)
    scale_factors: Any | None = eqx.field(default=None)
    comoving_centers: Any | None = eqx.field(default=None)
    density_width: Any | None = eqx.field(default=None)

    # Static metadata (not traced by JAX)
    observer_position: tuple[float, float, float] = eqx.field(static=True, default=(0.5, 0.5, 0.5))
    sharding: Any | None = eqx.field(static=True, default=None)
    halo_size: tuple[int, int] = eqx.field(static=True, default=(0, 0))
    nside: int | None = eqx.field(static=True, default=None)
    flatsky_npix: tuple[int, int] | None = eqx.field(static=True, default=None)
    field_size: float | None = eqx.field(static=True, default=None)
    status: FieldStatus = eqx.field(static=True, default=FieldStatus.UNKNOWN)
    unit: PhysicalUnit = eqx.field(static=True, default=PhysicalUnit.INVALID_UNIT)

    STATUS_ENUM = FieldStatus

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        # Validate mesh_size
        if not (isinstance(self.mesh_size, tuple) and len(self.mesh_size) == 3):
            raise ValueError(f"mesh_size must be a tuple of 3, got {self.mesh_size}")
        # Validate box_size
        if not (isinstance(self.box_size, tuple) and len(self.box_size) == 3):
            raise ValueError(f"box_size must be a tuple of 3, got {self.box_size}")
        # Validate observer_position
        if not (isinstance(self.observer_position, tuple) and len(self.observer_position) == 3):
            raise ValueError(f"observer_position must be a tuple of 3, got {self.observer_position}")
        if any(not 0.0 <= frac <= 1.0 for frac in self.observer_position):
            raise ValueError("observer_position entries must lie within [0, 1]")
        # Validate halo_size
        if not (isinstance(self.halo_size, tuple) and len(self.halo_size) == 2):
            raise ValueError(f"halo_size must be a tuple of 2, got {self.halo_size}")
        if any(h < 0 for h in self.halo_size):
            raise ValueError("halo_size entries must be >= 0")
        # Validate nside if provided
        if self.nside is not None and self.nside <= 0:
            raise ValueError(f"nside must be positive, got {self.nside}")
        # Validate flatsky_npix if provided
        if self.flatsky_npix is not None:
            if not (len(self.flatsky_npix) == 2 and all(x > 0 for x in self.flatsky_npix)):
                raise ValueError(f"flatsky_npix must be 2 positive ints, got {self.flatsky_npix}")

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

        Stacking (S, ...) inputs produces an (N, S, ...) multi-batched field.
        Cannot stack already-multi-batched (N, S, ...) inputs; use concat() instead.
        """
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("stack requires at least one field.")
        if any(f.is_multi_batched() for f in field_list):
            raise ValueError(
                "Cannot stack multi-batched (N, S, ...) fields — this would create an ambiguous "
                "extra leading dimension. Use <FieldClass>.concat(fields) to join along the N axis instead."
            )
        return jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0),
            *field_list,
        )

    @classmethod
    def concat(cls, fields: Sequence[Self]) -> Self:
        """Concatenate multi-batched (N, S, ...) fields along the N axis (axis 0)."""
        field_list = tuple(fields)
        if not field_list:
            raise ValueError("concat requires at least one field.")
        if any(not f.is_multi_batched() for f in field_list):
            raise ValueError(
                "concat only works on multi-batched (N, S, ...) fields. "
                "Use stack() to create an (N, S, ...) field from (S, ...) inputs."
            )
        return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *field_list)

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

    @classmethod
    @abstractmethod
    def full_like(cls, field: AbstractField, fill_value: float = 0.0) -> Self:
        """
        Create a new field of the same class and metadata as `field`, filled with `fill_value`.

        Parameters
        ----------
        field : AbstractField
            Reference field supplying metadata.
        fill_value : float, optional
            Value to fill the new array with (default is 0.0).

        Returns
        -------
        AbstractField subclass
            New instance with the same metadata as `field` and an array filled with `fill_value`.
        """
        raise NotImplementedError("full_like must be implemented in subclasses.")

    @abstractmethod
    def is_batched(self) -> bool:
        """
        Return True if the field is batched (i.e. has a leading batch dimension).

        This is used to determine whether to apply certain operations in a batched manner.
        The exact definition of "batched" may depend on the context and should be implemented in subclasses.

        Returns
        -------
        bool
            True if the field is considered batched, False otherwise.
        """
        raise NotImplementedError("is_batched must be implemented in subclasses.")

    @abstractmethod
    def is_multi_batched(self) -> bool:
        """Return True when the field has both a simulation-batch (N) and snapshot (S) dimension."""
        raise NotImplementedError("is_multi_batched must be implemented in subclasses.")

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
        array_shape = self.array.shape if self.array is not None else "(None)"
        dtype = self.array.dtype if self.array is not None else None
        return (
            f"{classname}("
            f"array  = Array{array_shape}\n, "
            f"dtype  = {dtype}, "
            f"  mesh_size         ={self.mesh_size}, "
            f"  box_size          ={self.box_size}, "
            f"  observer_position ={self.observer_position}, "
            f"  sharding          ={self.sharding}, "
            f"  halo_size         ={self.halo_size}, "
            f"  nside             ={self.nside}, "
            f"  flatsky_npix      ={self.flatsky_npix}, "
            f"  field_size        ={self.field_size}, "
            f"  status            ={self.status.name}, "
            f"  unit              ={self.unit.name})"
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
        if arr is None:
            dbg.print("{} array is None", classname)
            return
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
        dbg.inspect_array_sharding(arr, callback=print)

    def to_metadata(self) -> FieldMetadata:
        """Return a AbstractField with array=None, preserving all metadata."""
        return FieldMetadata(
            array=None,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            z_sources=self.z_sources,
            scale_factors=self.scale_factors,
            comoving_centers=self.comoving_centers,
            density_width=self.density_width,
            observer_position=self.observer_position,
            sharding=self.sharding,
            halo_size=self.halo_size,
            nside=self.nside,
            flatsky_npix=self.flatsky_npix,
            field_size=self.field_size,
            status=self.status,
            unit=self.unit,
        )

    # ------------------------------------------------------------------ Factory
    def __getitem__(self, key) -> Self:
        to_index, not_to_index = eqx.partition(self, lambda x: eqx.is_array(x) and x.ndim >= 1)
        to_index = jax.tree.map(lambda x: x[key], to_index)
        return eqx.combine(to_index, not_to_index)


class FieldMetadata(AbstractField):
    """Concrete metadata-only field (array=None). Produced by ``to_metadata()``."""

    @classmethod
    def full_like(cls, field: AbstractField, fill_value: float = 0.0) -> FieldMetadata:
        raise NotImplementedError("FieldMetadata is metadata-only; use a concrete field class.")

    def is_batched(self) -> bool:
        return False

    def is_multi_batched(self) -> bool:
        return False
