"""Catalog container for field data and cosmology with HuggingFace serialization."""

from __future__ import annotations

import dataclasses
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from .._src.base._core import AbstractField
from .._src.base._enums import DensityUnit, FieldStatus, PositionUnit
from ..fields import DensityField, FlatDensity, ParticleField, SphericalDensity

try:
    import datasets
except ImportError:
    pass

CATALOG_VERSION = 1

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
    if field_type == "SphericalDensity":
        # Expected shape: (batch, npix) or (npix,)
        if array.ndim == 1:
            return array[None, ...]
        elif array.ndim != 2:
            raise ValueError(f"Unexpected shape for SphericalDensity: {array.shape}")
    elif field_type == "FlatDensity":
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


def _build_features(field: AbstractField, batch_size: int) -> "datasets.Features":
    """Build HuggingFace Features schema based on field type and shapes."""
    from datasets import Array2D, Array3D, Array4D, Array5D, Features, Sequence, Value

    field_type = type(field).__name__
    array = _ensure_batch_dim(field.array, field_type)
    shape = array.shape
    dtype = array.dtype

    batch_size = array.shape[0]

    if field.z_sources is None or field.scale_factors is None or \
       field.comoving_centers is None or field.density_width is None:
        raise ValueError("Dynamic metadata arrays cannot be None when building features.")

    if field.z_sources.size != batch_size or \
       field.scale_factors.size != batch_size or \
       field.comoving_centers.size != batch_size or \
       field.density_width.size != batch_size:
        raise ValueError("Dynamic metadata arrays must match batch size when building features.")

    # Build array feature based on field type
    dtype_str = np.dtype(dtype).name
    if field_type == "SphericalDensity":
        array_feature = Array2D(shape=shape, dtype=dtype_str)
    elif field_type == "FlatDensity":
        array_feature = Array3D(shape=shape, dtype=dtype_str)
    elif field_type == "DensityField":
        array_feature = Array4D(shape=shape, dtype=dtype_str)
    elif field_type == "ParticleField":
        array_feature = Array5D(shape=shape, dtype=dtype_str)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    feature_dict = {
        # Main array
        "array": array_feature,
        # Dynamic metadata - shape (batch,)
        "z_sources": Sequence(Value(np.dtype(field.z_sources.dtype).name), length=batch_size),
        "scale_factors": Sequence(Value(np.dtype(field.scale_factors.dtype).name), length=batch_size),
        "comoving_centers": Sequence(Value(np.dtype(field.comoving_centers.dtype).name), length=batch_size),
        "density_width": Sequence(Value(np.dtype(field.density_width.dtype).name), length=batch_size),
        # Static metadata - tuples
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
    }

    # Optional fields - only add if not None
    if field.nside is not None:
        feature_dict["nside"] = Value("int32")
    if field.flatsky_npix is not None:
        feature_dict["flatsky_npix"] = Sequence(Value("int32"), length=2)
    if field.field_size is not None:
        feature_dict["field_size"] = Sequence(Value("float32"), length=2)

    return Features(feature_dict)


def _field_to_dict(field: AbstractField, cosmology: jc.Cosmology, version: int) -> dict:
    """Convert field and cosmology to dict for dataset creation."""
    field_type = type(field).__name__

    field_data = dataclasses.asdict(field)
    cosmo_data = {
        'h': cosmology.h,
        "Omega_c": cosmology.Omega_c,
        "Omega_b": cosmology.Omega_b,
        'Omega_k': cosmology.Omega_k,
        "w0": cosmology.w0,
        "wa": cosmology.wa,
        'n_s': cosmology.n_s,
        'sigma8': cosmology.sigma8,
        "Omega_nu": cosmology.Omega_nu,
    }

    field_data['status'] = field.status.name
    field_data['unit'] = field.unit.name
    field_data['field_type'] = field_type
    field_data['version'] = version

    field_data.pop('sharding', None)  # Remove sharding info

    data = {**field_data, **cosmo_data}

    # Remove None values (optional fields not present)
    data = {k: v for k, v in data.items() if v is not None}

    return data


def _dict_to_catalog(item: dict) -> "Catalog":
    """Convert dataset item dict to Catalog object."""

    field_classes = {
        "SphericalDensity": SphericalDensity,
        "FlatDensity": FlatDensity,
        "DensityField": DensityField,
        "ParticleField": ParticleField,
    }
    field_cls = field_classes[item["field_type"]]
    if field_cls is ParticleField:
        unit = PositionUnit[item["unit"]]
    else:
        unit = DensityUnit[item["unit"]]

    def _to_static_tuple(v, _type):
        return tuple(_type(x) for x in v)

    field = field_cls(
        array=item["array"],
        mesh_size=_to_static_tuple(item["mesh_size"], int),
        box_size=_to_static_tuple(item["box_size"], float),
        observer_position=_to_static_tuple(item["observer_position"], float),
        sharding=None,
        halo_size=_to_static_tuple(item["halo_size"], float),
        nside=int(item.get("nside")),
        flatsky_npix=_to_static_tuple(item["flatsky_npix"], int),
        field_size=_to_static_tuple(item["field_size"], float),
        z_sources=item.get("z_sources"),
        scale_factors=item.get("scale_factors"),
        comoving_centers=item.get("comoving_centers"),
        density_width=item.get("density_width"),
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

    return Catalog(field=field, cosmology=cosmology, version=item["version"])


class Catalog(eqx.Module):
    """Container for field data and associated cosmology.

    Attributes
    ----------
    field : AbstractField
        The field data (SphericalDensity, FlatDensity, DensityField, or ParticleField).
    cosmology : jc.Cosmology
        The cosmology used to generate the simulation.
    version : int
        Catalog version for compatibility tracking.
    """

    field: AbstractField
    cosmology: jc.Cosmology
    version: int = eqx.field(static=True, default=CATALOG_VERSION)

    def __repr__(self) -> str:
        return (f"Catalog(\n"
                f"  field     ={self.field!r},\n"
                f"  cosmology ={self.cosmology}, "
                f"  version   ={self.version}"
                f")")

    @requires_datasets
    def to_dataset(self) -> "datasets.Dataset":
        """Convert Catalog to HuggingFace Dataset.

        Returns
        -------
        datasets.Dataset
            A single-row dataset containing all field and cosmology data.

        Raises
        ------
        ValueError
            If called inside a jit context (arrays are tracers).
        ImportError
            If datasets library is not installed.
        """
        from datasets import Dataset

        if not jax.core.is_concrete(self.field.array):
            raise ValueError("Cannot convert to dataset inside a jit context (arrays are tracers).")

        field_type = type(self.field).__name__
        array = _ensure_batch_dim(self.field.array, field_type)
        batch_size = array.shape[0]
        features = _build_features(self.field, batch_size)

        data = _field_to_dict(self.field, self.cosmology, self.version)

        # Wrap each value in a list for single-row dataset
        data = {k: [v] for k, v in data.items()}

        return Dataset.from_dict(data, features=features)

    @requires_datasets
    def to_parquet(self, path: str) -> None:
        """Save Catalog to parquet file.

        Parameters
        ----------
        path : str
            Path to save the parquet file.

        Raises
        ------
        ValueError
            If called inside a jit context (arrays are tracers).
        ImportError
            If datasets library is not installed.
        """
        ds = self.to_dataset()
        ds.to_parquet(path)

    @classmethod
    @requires_datasets
    def from_parquet(cls, path: str, format="jax") -> "Catalog":
        """Load Catalog from parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.

        Returns
        -------
        Catalog
            Reconstructed Catalog object.

        Raises
        ------
        ImportError
            If datasets library is not installed.
        """
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=path, split="train").with_format(format)
        return cls.from_dataset(ds[0])

    @classmethod
    @requires_datasets
    def from_dataset(cls, item: dict) -> "Catalog":
        """Convert a dataset item (dict) to Catalog object.

        Parameters
        ----------
        item : dict
            A dictionary from a HuggingFace dataset row.

        Returns
        -------
        Catalog
            Reconstructed Catalog object.

        Raises
        ------
        ImportError
            If datasets library is not installed.
        """
        return _dict_to_catalog(item)
