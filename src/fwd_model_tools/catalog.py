"""
Catalog I/O helpers for density-like fields.

Stores a compact parquet row containing:
    - serialized array payload
    - field metadata (geometry, status, scale_factors, etc.)
    - derived z_near / z_far edges from scale factors
    - cosmology parameters (JSON)

Requires the optional dependency ``pyarrow``. Install via
``pip install fwd_model_tools[io]`` or ``pip install pyarrow``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from fwd_model_tools.fields import DensityField, FieldStatus, FlatDensity, SphericalDensity

PathLike = Union[str, Path]


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pyarrow is required for catalog I/O. Install fwd_model_tools[io] or pip install pyarrow."
        ) from exc
    return pa, pq


def _to_jsonable(obj: Any):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _serialize_array(arr: Any) -> Tuple[bytes, Tuple[int, ...], str]:
    np_arr = np.asarray(arr)
    return np_arr.tobytes(), tuple(np_arr.shape), str(np_arr.dtype)


def _deserialize_array(payload: bytes, shape: Tuple[int, ...], dtype: str) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)


def _compute_z_edges(scale_factors: Any) -> Tuple[list[float], list[float]]:
    a = np.atleast_1d(np.asarray(scale_factors, dtype=float)).ravel()
    if a.size == 0:
        return [], []
    if a.size == 1:
        # symmetric tiny buffer around the single slice
        delta = 1e-3
        edges = np.array([max(a[0] - delta, 1e-6), a[0] + delta])
    else:
        edges = np.empty(a.size + 1, dtype=float)
        mids = 0.5 * (a[:-1] + a[1:])
        edges[1:-1] = mids
        edges[0] = max(a[0] - (mids[0] - a[0]), 1e-8)
        edges[-1] = a[-1] + (a[-1] - mids[-1])
    z_edges = 1.0 / np.clip(edges, 1e-8, None) - 1.0
    return z_edges[:-1].tolist(), z_edges[1:].tolist()


def _status_for_kind(kind: str, value: str):
    if kind == "density":
        return FieldStatus(value)
    return DensityStatus(value)


def to_catalog(field: DensityField, cosmo: jc.Cosmology, path: PathLike) -> Path:
    """
    Write a DensityField (or subclass) plus cosmology to a parquet file.

    Parameters
    ----------
    field : DensityField | FlatDensity | SphericalDensity
        Field instance to serialize.
    cosmo : jc.Cosmology
        Cosmology used to generate the field; stored as JSON.
    path : str or Path
        Destination parquet file.
    """
    pa, pq = _require_pyarrow()

    if isinstance(field, FlatDensity):
        kind = "flat"
    elif isinstance(field, SphericalDensity):
        kind = "spherical"
    elif isinstance(field, DensityField):
        kind = "density"
    else:
        raise TypeError("to_catalog expects DensityField or subclasses.")

    payload, shape, dtype = _serialize_array(field.array)
    z_near, z_far = _compute_z_edges(field.scale_factors)

    cosmo_dict = _to_jsonable(cosmo)
    row: Dict[str, Any] = {
        "field_kind": kind,
        "array": payload,
        "array_shape": list(shape),
        "array_dtype": dtype,
        "mesh_size": list(field.mesh_size),
        "box_size": list(field.box_size),
        "observer_position": list(field.observer_position),
        "sharding": None,
        "nside": field.nside,
        "flatsky_npix": list(field.flatsky_npix) if field.flatsky_npix else None,
        "field_size": field.field_size,
        "halo_size": list(field.halo_size) if isinstance(field.halo_size, tuple) else field.halo_size,
        "z_source": _to_jsonable(field.z_source),
        "status": field.status.value,
        "scale_factors": _to_jsonable(field.scale_factors),
        "z_near": z_near,
        "z_far": z_far,
        "cosmology": json.dumps(cosmo_dict),
    }

    table = pa.Table.from_pydict({k: [v] for k, v in row.items()})
    path = Path(path)
    pq.write_table(table, path)
    return path


def from_catalog(path: PathLike) -> tuple[DensityField, jc.Cosmology, np.ndarray, np.ndarray]:
    """
    Load a field and cosmology from a parquet catalog written by :func:`to_catalog`.

    Returns
    -------
    field : DensityField | FlatDensity | SphericalDensity
    cosmo : jc.Cosmology
    z_near : np.ndarray
    z_far : np.ndarray
    """
    pa, pq = _require_pyarrow()
    table = pq.read_table(path)
    row = table.to_pydict()
    get = lambda key: row[key][0]

    kind = get("field_kind")
    array = _deserialize_array(get("array"), tuple(get("array_shape")), get("array_dtype"))
    status = _status_for_kind(kind, get("status"))

    base_kwargs = dict(
        mesh_size=tuple(get("mesh_size")),
        box_size=tuple(get("box_size")),
        observer_position=tuple(get("observer_position")),
        sharding=None,
        nside=get("nside"),
        flatsky_npix=tuple(get("flatsky_npix")) if get("flatsky_npix") else None,
        field_size=get("field_size"),
        halo_size=tuple(get("halo_size")) if isinstance(get("halo_size"), list) else get("halo_size"),
        z_source=get("z_source"),
        status=status,
        scale_factors=jnp.asarray(get("scale_factors")),
    )

    if kind == "density":
        field = DensityField(array=jnp.asarray(array), **base_kwargs)
    elif kind == "flat":
        density_ref = DensityField(array=jnp.zeros(base_kwargs["mesh_size"]), **base_kwargs)
        field = FlatDensity.FromDensityMetadata(
            array=jnp.asarray(array),
            density_field=density_ref,
            status=status,
            scale_factors=base_kwargs["scale_factors"],
        )
    elif kind == "spherical":
        density_ref = DensityField(array=jnp.zeros(base_kwargs["mesh_size"]), **base_kwargs)
        field = SphericalDensity.FromDensityMetadata(
            array=jnp.asarray(array),
            density_field=density_ref,
            status=status,
            scale_factors=base_kwargs["scale_factors"],
        )
    else:
        raise ValueError(f"Unknown field_kind '{kind}' in catalog.")

    cosmo_dict = json.loads(get("cosmology"))
    cosmo = jc.Cosmology(**cosmo_dict)

    z_near = np.asarray(get("z_near"))
    z_far = np.asarray(get("z_far"))
    return field, cosmo, z_near, z_far
