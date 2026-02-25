"""Streaming catalog analysis: per-chain statistics via Welford's algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..fields import DensityField
from .catalog import Catalog

__all__ = ["extract_catalog", "requires_datasets", "CatalogExtract"]

_Param = ParamSpec("_Param")
_Return = TypeVar("_Return")

# pyright: reportOptionalMemberAccess=false


def requires_datasets(func: Callable[_Param, _Return]) -> Callable[_Param, _Return]:
    """Decorator that raises ImportError when the 'datasets' package is not installed."""
    try:
        import datasets  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def _deferred(*args: _Param.args, **kwargs: _Param.kwargs) -> _Return:
        raise ImportError("Missing optional dependency 'datasets'. Install with: pip install fwd-model-tools[catalog]")

    return _deferred


class _RunningStats:
    """Welford's one-pass online algorithm for mean and variance.

    Works with any ``AbstractPytree`` (DensityField, PowerSpectrum, etc.) by
    operating through the object's arithmetic operators and ``apply_fn``.  All
    accumulators are kept in ``float64`` numpy arrays so no JAX memory is
    consumed during the streaming loop.
    """

    def __init__(self, template):
        self.n = 0
        self.mean = template.apply_fn(lambda x: np.zeros_like(np.asarray(x), dtype=np.float64))
        self.M2 = template.apply_fn(lambda x: np.zeros_like(np.asarray(x), dtype=np.float64))

    def update(self, obj):
        x = obj.apply_fn(lambda arr: np.asarray(arr, dtype=np.float64))
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta * (1.0 / self.n)
        delta2 = x - self.mean
        self.M2 = self.M2 + delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self, ddof: int = 0):
        if self.n < 2:
            return self.mean.apply_fn(lambda x: np.zeros_like(x))
        denom = self.n - ddof
        return self.M2.apply_fn(lambda arr: np.sqrt(arr / denom))


def _detect_chains(path: Path) -> list[str]:
    """Detect chain layout under *path* and return glob strings for each chain.

    Two supported layouts (both require a ``samples/`` subdirectory):

    Single-chain layout::

        path/
        └── samples/
            ├── samples_0.parquet
            └── samples_1.parquet

    Multi-chain layout::

        path/
        ├── chain_0/
        │   └── samples/
        │       └── samples_0.parquet
        └── chain_1/
            └── samples/
                └── ...

    Returns
    -------
    list[str]
        One glob string per chain, usable directly as *data_files* in
        ``datasets.load_dataset``.
    """
    samples_dir = path / "samples"
    if samples_dir.exists() and any(samples_dir.glob("*.parquet")):
        return [str(samples_dir / "*.parquet")]

    subdirs = sorted(d for d in path.iterdir() if d.is_dir())
    chains = []
    for subdir in subdirs:
        chain_samples_dir = subdir / "samples"
        if chain_samples_dir.exists() and any(chain_samples_dir.glob("*.parquet")):
            chains.append(str(chain_samples_dir / "*.parquet"))

    if not chains:
        raise FileNotFoundError(
            f"No parquet files found under '{path}'. "
            "Expected either 'path/samples/*.parquet' (single chain) "
            "or 'path/chain_N/samples/*.parquet' (multi-chain)."
        )
    return chains


class CatalogExtract(eqx.Module):
    """Typed container for extract_catalog results.

    All fields default to None; only populated when the corresponding
    option was passed to extract_catalog().  No fields are marked static —
    this module is a plain data container (never JIT-compiled through).
    """

    name: str = eqx.field(static=True)  # for pretty printing
    cosmo: dict  # {key: np.ndarray (n_chains, n_samples)}
    true_ic: DensityField | None = None  # reference IC if provided
    mean_field: DensityField | None = None  # array shape (n_chains, X, Y, Z)
    std_field: DensityField | None = None  # array shape (n_chains, X, Y, Z)
    power_spectra: tuple | None = None  # (mean_tf, std_tf, mean_coh, std_coh)

    @property
    def n_chains(self) -> int:
        vals = list(self.cosmo.values())
        if not vals:
            return 0
        arr = np.asarray(vals[0])
        return arr.shape[0] if arr.ndim == 2 else 1

    @property
    def cosmo_keys(self) -> list[str]:
        return list(self.cosmo.keys())

    def __getitem__(self, idx) -> CatalogExtract:
        """Select chain(s) by index or slice.

        Integer index is normalised to a length-1 slice so cosmo arrays remain
        2-D ``(n_selected_chains, n_samples)`` and ``n_chains`` stays consistent.

        Examples
        --------
        result[0]    # first chain (n_chains=1)
        result[1:3]  # chains 1 and 2
        result[::2]  # every other chain
        """
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        new_cosmo = {k: np.asarray(v)[idx] for k, v in self.cosmo.items()}

        new_mean_field = (
            self.mean_field.replace(array=self.mean_field.array[idx]) if self.mean_field is not None else None
        )
        new_std_field = self.std_field.replace(array=self.std_field.array[idx]) if self.std_field is not None else None
        new_power_spectra = (
            tuple(ps.replace(array=ps.array[idx]) for ps in self.power_spectra)
            if self.power_spectra is not None
            else None
        )

        return CatalogExtract(
            cosmo=new_cosmo,
            true_ic=self.true_ic,
            mean_field=new_mean_field,
            std_field=new_std_field,
            power_spectra=new_power_spectra,
        )


@requires_datasets
def extract_catalog(
    path: str,
    set_name: str,
    cosmo_keys: list[str] | tuple[str, ...],
    true_ic: DensityField | None = None,
    field_statistic: bool = False,
    power_statistic: bool = False,
    ddof: int = 0,
) -> CatalogExtract:
    """Stream MCMC catalog parquet files and compute per-chain statistics.

    Parquet files are read one row at a time (streaming mode) so the full
    sample set never needs to fit in memory.  Statistics are accumulated with
    Welford's online algorithm in ``float64`` precision.

    Parameters
    ----------
    path : str
        Root directory.  Must contain either ``samples/`` (single chain) or
        ``chain_N/samples/`` subdirectories (multi-chain).
    cosmo_keys : list or tuple of str
        Cosmological parameter names to collect (e.g. ``["Omega_c", "sigma8"]``).
    true_ic : DensityField, optional
        Reference initial-condition field used for transfer / coherence spectra.
        Required when ``power_statistic=True``.
    field_statistic : bool
        If *True*, compute per-chain mean and std of the density fields.
    power_statistic : bool
        If *True*, compute per-chain mean and std of the transfer function and
        coherence spectrum relative to ``true_ic``.
    ddof : int
        Delta degrees of freedom for std computation (0 = population, 1 = sample).

    Returns
    -------
    CatalogExtract
        Typed container with attributes ``cosmo``, ``true_ic``, ``mean_field``,
        ``std_field``, and ``power_spectra``.
    """
    if power_statistic and true_ic is None:
        raise ValueError("power_statistic=True requires true_ic to be provided.")

    import datasets as hf_datasets

    path = Path(path)
    chain_globs = _detect_chains(path)
    n_chains = len(chain_globs)

    cosmo_lists: dict[str, list[list[float]]] = {key: [[] for _ in range(n_chains)] for key in cosmo_keys}
    chain_field_stats: list[_RunningStats | None] = [None] * n_chains
    chain_transfer_stats: list[_RunningStats | None] = [None] * n_chains
    chain_coherence_stats: list[_RunningStats | None] = [None] * n_chains

    for chain_idx, chain_glob in enumerate(chain_globs):
        streaming_ds = hf_datasets.load_dataset(
            "parquet",
            data_files=chain_glob,
            split="train",
            streaming=True,
        ).with_format("numpy")

        field_stats: _RunningStats | None = None
        transfer_stats: _RunningStats | None = None
        coherence_stats: _RunningStats | None = None

        for sample in streaming_ds:
            catalog = Catalog.from_dataset(sample)
            cosmo = catalog.cosmology[0]

            # --- cosmological parameters ---
            for key in cosmo_keys:
                cosmo_lists[key][chain_idx].append(float(getattr(cosmo, key)))

            # --- density field (cast to float64 for stable accumulation) ---
            # Welfords online mean calculating is iterative and relies on the arithmetic operators of the object, so we convert to numpy arrays here to ensure all intermediate calculations are done in float64 precision.
            field = catalog.field[0].apply_fn(lambda x: np.asarray(x, dtype=np.float64))

            # --- field statistics ---
            if field_statistic:
                if field_stats is None:
                    field_stats = _RunningStats(field)
                field_stats.update(field)

            # --- power spectra statistics ---
            if power_statistic:
                transfer_ps = field.transfer(true_ic).apply_fn(lambda x: np.asarray(x, dtype=np.float64))
                coherence_ps = field.coherence(true_ic).apply_fn(lambda x: np.asarray(x, dtype=np.float64))
                if transfer_stats is None:
                    transfer_stats = _RunningStats(transfer_ps)
                    coherence_stats = _RunningStats(coherence_ps)
                assert transfer_stats is not None and coherence_stats is not None  # type narrowing for checker
                transfer_stats.update(transfer_ps)
                coherence_stats.update(coherence_ps)

        chain_field_stats[chain_idx] = field_stats
        chain_transfer_stats[chain_idx] = transfer_stats
        chain_coherence_stats[chain_idx] = coherence_stats

    # --- Build cosmo dict: shape (n_chains, n_samples) ---
    cosmo_dict = {key: np.array(cosmo_lists[key]) for key in cosmo_keys}

    result_mean_field = None
    result_std_field = None
    result_power_spectra = None

    # --- Stack field statistics across chains ---
    if field_statistic:
        assert chain_transfer_stats is not None
        chain_means = [chain_field_stats[c].get_mean() for c in range(n_chains)]
        chain_stds = [chain_field_stats[c].get_std(ddof) for c in range(n_chains)]
        stacked_mean = np.stack([np.asarray(f.array) for f in chain_means], axis=0)
        stacked_std = np.stack([np.asarray(f.array) for f in chain_stds], axis=0)
        result_mean_field = chain_means[0].replace(array=jnp.array(stacked_mean))
        result_std_field = chain_stds[0].replace(array=jnp.array(stacked_std))

    # --- Stack power spectra across chains ---
    if power_statistic:

        def _stack_ps(ps_list):
            stacked = np.stack([np.asarray(ps.array) for ps in ps_list], axis=0)
            return ps_list[0].replace(array=jnp.array(stacked))

        mean_transfer = _stack_ps([chain_transfer_stats[c].get_mean() for c in range(n_chains)])
        std_transfer = _stack_ps([chain_transfer_stats[c].get_std(ddof) for c in range(n_chains)])
        mean_coherence = _stack_ps([chain_coherence_stats[c].get_mean() for c in range(n_chains)])
        std_coherence = _stack_ps([chain_coherence_stats[c].get_std(ddof) for c in range(n_chains)])
        result_power_spectra = (mean_transfer, std_transfer, mean_coherence, std_coherence)

    return CatalogExtract(
        cosmo=cosmo_dict,
        name=set_name,
        true_ic=true_ic,
        mean_field=result_mean_field,
        std_field=result_std_field,
        power_spectra=result_power_spectra,
    )
