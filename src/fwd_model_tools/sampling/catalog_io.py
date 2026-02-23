"""Catalog-aware sample loading utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..io.catalog import Catalog

__all__ = ["load_samples"]

# All cosmological parameters stored in each catalog row
COSMO_PARAMS: tuple[str, ...] = ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa", "Omega_k", "Omega_nu")

# THIS FILE SHOULD BE IN CATALOG


def _detect_chains(path: Path) -> list[list[Path]]:
    """Auto-detect chain structure from directory layout.

    Returns a list of chains where each chain is a sorted list of parquet files.

    Layout rules:
    - ``path/*.parquet`` present  → single chain (all files in path/)
    - ``path/subdir_N/*.parquet`` → multi-chain (one chain per sorted subdir)
    """
    parquet_files = sorted(path.glob("*.parquet"))
    if parquet_files:
        return [parquet_files]

    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    chains = []
    for subdir in subdirs:
        chain_files = sorted(subdir.glob("*.parquet"))
        if chain_files:
            chains.append(chain_files)

    if not chains:
        raise FileNotFoundError(f"No parquet files found in {path} (checked top-level and subdirectories).")

    return chains


def _load_chain(
    parquet_files: list[Path],
    cosmo_params: tuple[str, ...],
    collect_fields: bool,
) -> tuple[dict[str, list[float]], list[np.ndarray] | None, object | None]:
    """Load all parquet files for one chain sequentially (one file at a time).

    Returns
    -------
    cosmo_lists : dict mapping param name → list of float values (one per sample)
    field_arrays : list of np.ndarray (one per sample), or None if collect_fields=False
    first_field : the first loaded field object (for metadata cloning), or None
    """
    from datasets import load_dataset

    cosmo_lists: dict[str, list[float]] = {p: [] for p in cosmo_params}
    field_arrays: list[np.ndarray] | None = [] if collect_fields else None
    first_field = None

    for pf in parquet_files:
        ds = load_dataset("parquet", data_files=str(pf), split="train")
        catalog = Catalog.from_dataset(ds)
        for i in range(len(catalog)):
            field = catalog.field[i]
            cosmo = catalog.cosmology[i]
            for p in cosmo_params:
                cosmo_lists[p].append(float(getattr(cosmo, p)))
            if collect_fields:
                field_arrays.append(np.asarray(field.array))
                if first_field is None:
                    first_field = field

    return cosmo_lists, field_arrays, first_field


def load_samples(
    path: str,
    cosmo_params: list[str] | tuple[str, ...] | None = None,
    transforms: list[Callable] | None = None,
) -> dict | tuple[dict, list]:
    """Load MCMC samples from a catalog-format parquet directory.

    Supports two directory layouts (auto-detected):

    - **Single chain**: ``path/*.parquet`` — each file is one saved batch.
    - **Multi-chain**: ``path/subdir_N/*.parquet`` — each subdirectory is one chain.
      Subdirectories are sorted alphabetically.

    Parameters
    ----------
    path : str
        Root directory to load from.
    cosmo_params : list or tuple of str, optional
        Cosmological parameter names to extract. Defaults to all available:
        ``Omega_c``, ``Omega_b``, ``h``, ``n_s``, ``sigma8``, ``w0``, ``wa``,
        ``Omega_k``, ``Omega_nu``.
    transforms : list of Callable, optional
        If provided, each callable receives a ``jnp.ndarray`` of shape
        ``(n_total, *spatial)`` (all chains concatenated along axis 0) and must
        return an array. The result is wrapped into a field (metadata cloned from
        the first loaded sample via ``field.replace(array=...)``).

    Returns
    -------
    cosmo_dict : dict
        Mapping from parameter name to:

        - ``np.ndarray`` of shape ``(n_samples,)`` for single-chain input.
        - ``np.ndarray`` of shape ``(n_chains, n_samples_per_chain)`` for
          multi-chain input (chains are truncated to the shortest).

    If ``transforms`` is not None, returns ``(cosmo_dict, field_list)`` where
    ``field_list[k]`` is the field produced by ``transforms[k]``.
    """
    path = Path(path)
    if cosmo_params is None:
        cosmo_params = COSMO_PARAMS
    else:
        cosmo_params = tuple(cosmo_params)

    collect_fields = transforms is not None
    chains = _detect_chains(path)
    n_chains = len(chains)

    all_chain_cosmo: list[dict[str, np.ndarray]] = []
    all_chain_fields: list[list[np.ndarray]] = []
    first_field = None

    for chain_idx, chain_files in enumerate(chains):
        print(f"  Loading chain {chain_idx + 1}/{n_chains} ({len(chain_files)} file(s))...")
        cosmo_lists, field_arrays, chain_first_field = _load_chain(chain_files, cosmo_params, collect_fields)
        all_chain_cosmo.append({p: np.array(cosmo_lists[p]) for p in cosmo_params})
        if collect_fields:
            all_chain_fields.append(field_arrays)
            if first_field is None:
                first_field = chain_first_field

    # Build cosmo_dict with correct shape per number of chains
    if n_chains == 1:
        cosmo_dict = all_chain_cosmo[0]
    else:
        # Truncate all chains to shortest length for rectangular stacking
        min_len = min(len(c[cosmo_params[0]]) for c in all_chain_cosmo)
        cosmo_dict = {p: np.stack([c[p][:min_len] for c in all_chain_cosmo], axis=0) for p in cosmo_params}

    if not collect_fields:
        return cosmo_dict

    # Concatenate all field arrays across chains and apply transforms
    all_fields_flat: list[np.ndarray] = []
    for chain_fields in all_chain_fields:
        all_fields_flat.extend(chain_fields)

    stacked = jnp.array(np.stack(all_fields_flat, axis=0))

    transformed_fields = []
    for transform in transforms:
        result_array = transform(stacked)
        transformed_fields.append(first_field.replace(array=result_array))

    return cosmo_dict, transformed_fields
