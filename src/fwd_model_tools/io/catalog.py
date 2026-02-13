"""Catalog container for field data and cosmology with HuggingFace serialization."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from .._src.base._core import AbstractField
from .._src.base._enums import ConvergenceUnit, DensityUnit, FieldStatus, PositionUnit
from ..fields import DensityField, FlatDensity, ParticleField, SphericalDensity
from ..fields.lensing_maps import FlatKappaField, SphericalKappaField

try:
    import datasets
except ImportError:
    pass

CATALOG_VERSION = 2

__all__ = ["Catalog", "CATALOG_VERSION"]

# Type variables for decorator
Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")


def requires_datasets(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Decorator that raises ImportError if datasets library is not available."""
    try:
        import datasets  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 'datasets'. Install with: pip install fwd-model-tools[catalog]"
        raise ImportError(msg)

    return deferred_func


def _ensure_batch_dim(array: jnp.ndarray, field_type: str) -> jnp.ndarray:
    """Ensure array has batch dimension."""
    if field_type in ("SphericalDensity", "SphericalKappaField"):  # Check of SphericalDensity or subclass
        # Expected shape: (batch, npix) or (npix,)
        if array.ndim == 1:
            return array[None, ...]
        elif array.ndim != 2:
            raise ValueError(f"Unexpected shape for {field_type}: {array.shape}")
    elif field_type in ("FlatDensity", "FlatKappaField"):  # Check of FlatDensity or subclass
        # Expected shape: (batch, ny, nx) or (ny, nx)
        if array.ndim == 2:
            return array[None, ...]
        elif array.ndim != 3:
            raise ValueError(f"Unexpected shape for FlatDensity: {array.shape}")
    elif field_type == "DensityField":
        # Expected shape: (batch, nx, ny, nz) or (nx, ny, nz)
        if array.ndim == 3:
            return array[None, ...]
        elif array.ndim != 4:
            raise ValueError(f"Unexpected shape for DensityField: {array.shape}")
    elif field_type == "ParticleField":
        # Expected shape: (batch, X, Y, Z, 3) or (X, Y, Z, 3)
        if array.ndim == 4:
            return array[None, ...]
        elif array.ndim != 5:
            raise ValueError(f"Unexpected shape for ParticleField: {array.shape}")
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return array


def _ensure_1d_metadata(value, name: str, batch_size: int) -> np.ndarray:
    """Normalize dynamic metadata to a 1-D array of length ``batch_size``.

    Handles:
    - 0-d scalar -> expand to shape ``(1,)``
    - 1-d array  -> validate length matches ``batch_size``
    - 2-d array with trailing dim 1 (e.g. ``(B, 1)`` from stacking) -> squeeze
    - None       -> raise immediately
    """
    if value is None:
        raise ValueError(f"Dynamic metadata '{name}' cannot be None.")
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr[None]  # scalar -> (1,)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)  # (B, 1) -> (B,) from stacking
    if arr.ndim != 1:
        raise ValueError(f"Dynamic metadata '{name}' must be 0-d or 1-d, got shape {arr.shape}.")
    if arr.shape[0] != batch_size:
        raise ValueError(f"Dynamic metadata '{name}' length {arr.shape[0]} != batch_size {batch_size}.")
    return arr


def _build_features(field: AbstractField) -> datasets.Features:
    """Build HuggingFace Features schema for v2 format (batched arrays per row)."""
    from datasets import Array2D, Array3D, Array4D, Array5D, Features, Sequence, Value

    field_type = type(field).__name__
    array = _ensure_batch_dim(field.array, field_type)
    element_shape = array.shape[1:]  # spatial dims (without batch)
    dtype_str = np.dtype(array.dtype).name

    # Batched array feature: leading dim is None (variable batch size)
    if field_type in ("SphericalDensity", "SphericalKappaField"):
        # (B, npix) -> Array2D with None first dim
        array_feature = Array2D(shape=(None, element_shape[0]), dtype=dtype_str)
    elif field_type in ("FlatDensity", "FlatKappaField"):
        # (B, ny, nx) -> Array3D
        array_feature = Array3D(shape=(None, *element_shape), dtype=dtype_str)
    elif field_type == "DensityField":
        # (B, X, Y, Z) -> Array4D
        array_feature = Array4D(shape=(None, *element_shape), dtype=dtype_str)
    elif field_type == "ParticleField":
        # (B, X, Y, Z, 3) -> Array5D
        array_feature = Array5D(shape=(None, *element_shape), dtype=dtype_str)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    feature_dict = {
        # Main array (full batched shape per row)
        "array": array_feature,
        # Dynamic metadata -- sequences per row (one value per batch element)
        "z_sources": Sequence(Value("float64")),
        "scale_factors": Sequence(Value("float64")),
        "comoving_centers": Sequence(Value("float64")),
        "density_width": Sequence(Value("float64")),
    }

    feature_dict.update({
        # Static metadata - tuples (same every row)
        "mesh_size": Sequence(Value("int32"), length=3),
        "box_size": Sequence(Value("float32"), length=3),
        "observer_position": Sequence(Value("float32"), length=3),
        "halo_size": Sequence(Value("int32"), length=2),
        # Enum fields stored as strings
        "status": Value("string"),
        "unit": Value("string"),
        # Cosmology parameters
        "Omega_c": Value("float32"),
        "Omega_b": Value("float32"),
        "h": Value("float32"),
        "n_s": Value("float32"),
        "sigma8": Value("float32"),
        "w0": Value("float32"),
        "wa": Value("float32"),
        "Omega_k": Value("float32"),
        "Omega_nu": Value("float32"),
        # Version and field type
        "version": Value("int32"),
        "field_type": Value("string"),
    })

    # Optional fields - only add if not None
    if field.nside is not None:
        feature_dict["nside"] = Value("int32")
    if field.flatsky_npix is not None:
        feature_dict["flatsky_npix"] = Sequence(Value("int32"), length=2)
    if field.field_size is not None:
        feature_dict["field_size"] = Sequence(Value("float32"), length=2)

    return Features(feature_dict)


def _catalog_to_row(field: AbstractField, cosmology: jc.Cosmology, version: int) -> dict:
    """Convert a single field + cosmology into a 1-row column-oriented dict.

    The full batched array is stored in a single row. Each key maps to a
    **list of length 1** so that ``Dataset.from_dict`` creates one row.
    """
    field_type = type(field).__name__
    array = np.asarray(_ensure_batch_dim(field.array, field_type))
    batch_size = array.shape[0]

    # Normalize dynamic metadata to 1-D arrays of length batch_size
    z_src = _ensure_1d_metadata(field.z_sources, "z_sources", batch_size)
    scale = _ensure_1d_metadata(field.scale_factors, "scale_factors", batch_size)
    comov = _ensure_1d_metadata(field.comoving_centers, "comoving_centers", batch_size)
    dw = _ensure_1d_metadata(field.density_width, "density_width", batch_size)

    # Static metadata
    static = {
        "mesh_size": list(field.mesh_size),
        "box_size": list(field.box_size),
        "observer_position": list(field.observer_position),
        "halo_size": list(field.halo_size),
        "status": field.status.name,
        "unit": field.unit.name,
        "field_type": field_type,
        "version": int(version),
    }

    # Optional static fields
    if field.nside is not None:
        static["nside"] = int(field.nside)
    if field.flatsky_npix is not None:
        static["flatsky_npix"] = list(field.flatsky_npix)
    if field.field_size is not None:
        static["field_size"] = list(field.field_size)

    # Cosmology
    cosmo = {
        "Omega_c": float(cosmology.Omega_c),
        "Omega_b": float(cosmology.Omega_b),
        "h": float(cosmology.h),
        "n_s": float(cosmology.n_s),
        "sigma8": float(cosmology.sigma8),
        "w0": float(cosmology.w0),
        "wa": float(cosmology.wa),
        "Omega_k": float(cosmology.Omega_k),
        "Omega_nu": float(cosmology.Omega_nu),
    }

    # Build column-oriented dict -- 1-element list per key (1 row)
    data = {
        "array": [array],
        "z_sources": [z_src.tolist()],
        "scale_factors": [scale.tolist()],
        "comoving_centers": [comov.tolist()],
        "density_width": [dw.tolist()],
    }

    # Static metadata and cosmology -- wrap in 1-element lists
    for k, v in static.items():
        data[k] = [v]
    for k, v in cosmo.items():
        data[k] = [v]

    return data


def _row_to_field_cosmo(item: dict) -> tuple[AbstractField, jc.Cosmology, int]:
    """Convert a single v2 dataset row (dict) to a (field, cosmology, version) tuple.

    In v2 format each row stores the full batched array (B, ...) and
    dynamic metadata as sequences of length B.  If B == 1 the leading
    batch dimension is squeezed to recover the original unbatched field.
    """
    field_classes = {
        "SphericalDensity": SphericalDensity,
        "SphericalKappaField": SphericalKappaField,
        "FlatDensity": FlatDensity,
        "FlatKappaField": FlatKappaField,
        "DensityField": DensityField,
        "ParticleField": ParticleField,
    }
    field_cls = field_classes[item["field_type"]]
    if field_cls is ParticleField:
        unit = PositionUnit[item["unit"]]
    elif field_cls in (SphericalKappaField, FlatKappaField):
        unit = ConvergenceUnit[item["unit"]]
    else:
        unit = DensityUnit[item["unit"]]

    def _to_static_tuple(v, _type):
        return tuple(_type(x) for x in v)

    # The array is (B, ...) shaped.  Detect unbatched via B == 1.
    array = jnp.asarray(item["array"])
    unbatched = array.shape[0] == 1

    if unbatched:
        array = array[0]  # squeeze batch dim -> original unbatched shape

    # Dynamic metadata: stored as sequences of length B
    def _read_dynamic(key):
        val = jnp.asarray(item[key], dtype=jnp.float64)
        if unbatched:
            return val[0] if val.ndim >= 1 else val  # scalar
        return val

    # Reconstruct z_sources as plain numeric (same as scale_factors etc.)
    z_sources = _read_dynamic("z_sources")
    if not unbatched:
        z_sources = jnp.atleast_1d(z_sources)

    field = field_cls(
        array=array,
        mesh_size=_to_static_tuple(item["mesh_size"], int),
        box_size=_to_static_tuple(item["box_size"], float),
        observer_position=_to_static_tuple(item["observer_position"], float),
        sharding=None,
        halo_size=_to_static_tuple(item["halo_size"], float),
        nside=int(item["nside"]) if "nside" in item and item["nside"] is not None else None,
        flatsky_npix=_to_static_tuple(item["flatsky_npix"], int)
        if "flatsky_npix" in item and item["flatsky_npix"] is not None else None,
        field_size=_to_static_tuple(item["field_size"], float)
        if "field_size" in item and item["field_size"] is not None else None,
        z_sources=z_sources,
        scale_factors=_read_dynamic("scale_factors") if "scale_factors" in item else None,
        comoving_centers=_read_dynamic("comoving_centers") if "comoving_centers" in item else None,
        density_width=_read_dynamic("density_width") if "density_width" in item else None,
        status=FieldStatus[item["status"]],
        unit=unit,
    )

    cosmology = jc.Cosmology(
        Omega_c=item["Omega_c"],
        Omega_b=item["Omega_b"],
        h=item["h"],
        n_s=item["n_s"],
        sigma8=item["sigma8"],
        w0=item["w0"],
        wa=item["wa"],
        Omega_k=item["Omega_k"],
        Omega_nu=item["Omega_nu"],
    )

    return field, cosmology, int(item["version"].squeeze())


class Catalog(eqx.Module):
    """Container for field data and associated cosmology.

    Stores **lists** of fields and cosmologies. A single field/cosmology
    is auto-wrapped into a length-1 list for a uniform internal representation.

    Attributes
    ----------
    field : list[AbstractField]
        The field data entries.
    cosmology : list[jc.Cosmology]
        The cosmology associated with each field.
    version : int
        Catalog version for compatibility tracking.
    """

    field: list[AbstractField]
    cosmology: list[jc.Cosmology]
    version: int = eqx.field(static=True, default=CATALOG_VERSION)

    def __init__(self, field, cosmology, version=CATALOG_VERSION):
        if isinstance(field, AbstractField):
            field = [field]
        if isinstance(cosmology, jc.Cosmology):
            cosmology = [cosmology]
        self.field = list(field)
        self.cosmology = list(cosmology)
        self.version = int(version)

    def __check_init__(self):
        if len(self.field) != len(self.cosmology):
            raise ValueError(
                f"field and cosmology must have the same length, got {len(self.field)} and {len(self.cosmology)}.")
        if not self.field:
            raise ValueError("Catalog must contain at least one field.")

    def __len__(self) -> int:
        return len(self.field)

    def __getitem__(self, key) -> Catalog:
        if isinstance(key, int):
            return Catalog(field=[self.field[key]], cosmology=[self.cosmology[key]], version=self.version)
        # Slice
        return Catalog(field=self.field[key], cosmology=self.cosmology[key], version=self.version)

    def __repr__(self) -> str:
        n = len(self.field)
        field_types = {type(f).__name__ for f in self.field}
        return f"Catalog(n_entries={n}, field_types={field_types}, version={self.version})"

    @requires_datasets
    def to_dataset(self) -> datasets.Dataset:
        """Convert this Catalog to a HuggingFace Dataset (one row per entry).

        Returns
        -------
        datasets.Dataset
            Dataset with one row per (field, cosmology) pair.
        """
        for f in self.field:
            if not jax.core.is_concrete(f.array):
                raise ValueError("Cannot convert to dataset inside a jit context (arrays are tracers).")

        ds_list = []
        for f, c in zip(self.field, self.cosmology):
            from datasets import Dataset

            features = _build_features(f)
            data = _catalog_to_row(f, c, self.version)
            ds_list.append(Dataset.from_dict(data, features=features))

        return datasets.concatenate_datasets(ds_list)

    @requires_datasets
    def to_parquet(self, path: str) -> None:
        """Save Catalog to parquet file.

        Parameters
        ----------
        path : str
            Path to save the parquet file.
        """
        ds = self.to_dataset()
        ds.to_parquet(path)

    @classmethod
    @requires_datasets
    def from_parquet(cls, path: str) -> Catalog:
        """Load a Catalog from parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.

        Returns
        -------
        Catalog
            A single Catalog containing all entries from the file.
        """
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=path, split="train")
        return cls.from_dataset(ds)

    @classmethod
    @requires_datasets
    def from_dataset(cls, ds: datasets.Dataset) -> Catalog:
        """Reconstruct a Catalog from a HuggingFace Dataset.

        Parameters
        ----------
        ds : datasets.Dataset
            A HuggingFace Dataset (as returned by :meth:`to_dataset`).

        Returns
        -------
        Catalog
            A single Catalog containing all entries from the dataset.
        """
        if len(ds) == 0:
            raise ValueError("Cannot reconstruct Catalog from an empty dataset.")

        ds_jax = ds.with_format("jax")
        fields = []
        cosmologies = []
        version = CATALOG_VERSION
        for i in range(len(ds_jax)):
            f, c, v = _row_to_field_cosmo(ds_jax[i])
            fields.append(f)
            cosmologies.append(c)
            version = v

        return cls(field=fields, cosmology=cosmologies, version=version)
