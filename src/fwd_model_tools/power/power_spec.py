from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .._src.base._core import AbstractPytree
from .._src.fields._plotting import generate_titles

__all__ = ["PowerSpectrum"]


class PowerSpectrum(AbstractPytree):
    """
    Container for power spectrum data (P(k), C_ell, transfer, coherence, ...).

    Supports both single and batched spectra and inherits from AbstractPytree
    for JAX PyTree compatibility and math operations.

    The power spectrum values are stored in `array` (inherited from AbstractPytree).
    A `spectra` property is provided for backwards compatibility.
    """

    # array is inherited from AbstractPytree (the power spectrum values)
    wavenumber: jax.Array
    name: str | None = eqx.field(static=True, default=None)
    scale_factors: Any | None = None

    @property
    def spectra(self) -> jax.Array:
        """Alias for array (backwards compatibility)."""
        return self.array

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        if self.wavenumber.ndim != 1:
            raise ValueError("wavenumber must be 1D")

        n_k = self.wavenumber.shape[0]

        if self.array.ndim == 1:
            if self.array.shape[0] != n_k:
                raise ValueError(f"Spectra length {self.array.shape[0]} does not match wavenumber {n_k}.")
            return

        if self.array.ndim == 2:
            if self.array.shape[1] != n_k:
                raise ValueError(
                    f"Spectra shape {self.array.shape} incompatible with wavenumber {n_k}. Use shape (n_spec, n_k)."
                )
            return

        raise ValueError("Spectra must be 1D or 2D.")

    # ---- Representation ---------------------------------------------------
    def __repr__(self) -> str:
        return (
            "PowerSpectrum("
            f"wavenumber=Array{tuple(self.wavenumber.shape)}, "
            f"array=Array{tuple(self.array.shape)}, "
            f"name={self.name!r}, "
            f"scale_factors={self.scale_factors})"
        )

    def __getitem__(self, key) -> PowerSpectrum:
        """
        Slice spectra while keeping the wavenumber grid aligned.

        Examples
        --------
        ps[:5]          -> first 5 k and spectra entries
        ps[:2, :5]      -> first 2 spectra and first 5 k (for batched spectra)
        """
        # Normalize key into (spec_sel, k_sel)
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("__getitem__ expects key like spectra_sel, k_sel")
            k_sel, spec_sel = key
        else:
            k_sel, spec_sel = key, slice(None)

        k_new = jnp.atleast_1d(self.wavenumber[spec_sel])
        if self.array.ndim == 1:
            array_out = self.array[k_sel]
        else:
            array_out = self.array[k_sel, spec_sel]

        if self.scale_factors is not None and self.array.squeeze().ndim == 1:
            sf_new = jnp.atleast_1d(self.scale_factors[k_sel])
        else:
            sf_new = self.scale_factors

        return PowerSpectrum(wavenumber=k_new, array=array_out, name=self.name, scale_factors=sf_new)

    # ---- Plotting ---------------------------------------------------------
    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        label: Sequence[str] | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
        grid: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes, list[Any]]:
        """
        Overlay all spectra in this object on a single axis.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Target axes; if None, a new figure/axes is created.
        logx, logy : bool
            Whether to use log scaling on x/y.
        label : sequence of str, optional
            One label per spectrum; must match the batch dimension length.
        color : str, optional
            Fixed color to use; otherwise matplotlib cycle is used.
        figsize : tuple, optional
            Used only when creating a new axes.
        grid : bool
            Enable grid lines.
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.plot() outside of a jit context.")

        k_1d = self.wavenumber
        pk_2d = self.array[None, :] if self.array.ndim == 1 else self.array
        n_spec = pk_2d.shape[0]

        if label is None:
            base_name = self.name or "spectrum"
            label = generate_titles(base_name, self.scale_factors, n_spec)

        if isinstance(label, str):
            label = [label] * n_spec

        if not isinstance(label, (list | tuple)):
            raise TypeError("label must be a list/tuple of strings or None.")
        if len(label) != n_spec:
            # Fallback if length mismatch (e.g. if generate_titles produced something different, though it shouldn't)
            # or if user provided wrong length
            raise ValueError(f"label must have length {n_spec}, got {len(label)}.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
        else:
            fig = ax.get_figure()

        artists: list[Any] = []
        for i in range(n_spec):
            lab = label[i]
            (line,) = ax.plot(k_1d, pk_2d[i], label=lab, color=color, **kwargs)
            artists.append(line)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, which="both", ls=":", alpha=0.5)
        if (self.name or "").lower() == "cl":
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$C_\ell$")
        else:
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$P(k)$")
        ax.legend()
        return fig, ax, artists

    def show(
        self,
        *,
        ax: plt.Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        label: Sequence[str] | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
        grid: bool = True,
        **kwargs: Any,
    ):
        fig, ax, artists = self.plot(
            ax=ax,
            logx=logx,
            logy=logy,
            label=label,
            color=color,
            figsize=figsize,
            grid=grid,
            **kwargs,
        )
        plt.show()
        return fig, ax, artists

    def mean_std_plot(
        self,
        *,
        ax: plt.Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        label: Sequence[str] | None = None,
        color: str | None = None,
        alpha: float = 0.25,
        figsize: tuple[float, float] | None = None,
        grid: bool = True,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes, list[Any]]:
        """
        Plot mean and ±1σ band for a batched spectrum.

        If only one spectrum is present, plots it directly without a band.
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.mean_std_plot() outside jit.")

        k_1d = self.wavenumber
        pk_2d = self.array[None, :] if self.array.ndim == 1 else self.array
        n_spec = pk_2d.shape[0]

        if label is not None:
            if not isinstance(label, (list | tuple)):
                raise TypeError("label must be a list/tuple of strings or None.")
            if len(label) != n_spec:
                raise ValueError(f"label must have length {n_spec}, got {len(label)}.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
        else:
            fig = ax.get_figure()

        artists: list[Any] = []
        if n_spec == 1:
            lab = (label[0] if label is not None else self.name) or "spectrum"
            (line,) = ax.plot(k_1d, pk_2d[0], label=lab, color=color, **kwargs)
            artists.append(line)
        else:
            mean_pk = pk_2d.mean(axis=0)
            std_pk = pk_2d.std(axis=0)
            lab = (label[0] if label is not None else self.name) or "mean"
            (line,) = ax.plot(k_1d, mean_pk, label=lab, color=color, **kwargs)
            band = ax.fill_between(k_1d, mean_pk - std_pk, mean_pk + std_pk, color=color, alpha=alpha)
            artists.extend([line, band])

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, which="both", ls=":", alpha=0.5)
        if (self.name or "").lower() == "cl":
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$C_\ell$")
        else:
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$P(k)$")
        ax.legend()
        return fig, ax, artists

    def compare_plot(
        self,
        others: Sequence[PowerSpectrum],
        *,
        ax: plt.Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        grid: bool = True,
        labels: Sequence[Sequence[str]] | None = None,
        colors: Sequence[Sequence[str]] | None = None,
        ratio: bool = True,
        ratio_ylim: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes, list[Any]]:
        """
        Overlay this spectrum and others on one axis; optionally add ratios on a twin y-axis.

        `others` must share the same wavenumber grid.
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.compare_plot() outside jit.")
        if not others:
            raise ValueError("others must be non-empty.")

        k_ref = self.wavenumber
        pk_ref = self.array[None, :] if self.array.ndim == 1 else self.array
        if pk_ref.shape[0] != 1:
            raise ValueError("compare_plot expects the reference spectrum to be unbatched.")
        pk_ref = pk_ref[0]

        if labels is not None:
            if len(labels) != len(others):
                raise ValueError(f"Expected {len(others)} label rows, got {len(labels)}.")
            for i, lbls in enumerate(labels):
                pk_other = others[i].array[None, :] if others[i].array.ndim == 1 else others[i].array
                if len(lbls) != pk_other.shape[0]:
                    raise ValueError(
                        f"labels[{i}] length {len(lbls)} does not match spectra batch {pk_other.shape[0]}."
                    )

        if colors is not None:
            if len(colors) != len(others):
                raise ValueError(f"Expected {len(others)} color rows, got {len(colors)}.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
        else:
            fig = ax.get_figure()

        artists: list[Any] = []
        # plot reference
        (ref_line,) = ax.plot(k_ref, pk_ref, color="k", lw=2, label=self.name or "reference", **kwargs)
        artists.append(ref_line)

        # optional ratio axis
        ax_ratio = ax.twinx() if ratio else None
        if ax_ratio:
            ax_ratio.axhline(1.0, color="gray", lw=1, ls="--")

        for i, other in enumerate(others):
            if not jnp.allclose(other.wavenumber, k_ref):
                raise ValueError("All spectra in compare_plot must share the same wavenumber grid.")
            pk_other = other.array[None, :] if other.array.ndim == 1 else other.array
            for j in range(pk_other.shape[0]):
                color = colors[i][j] if colors else None
                lab = labels[i][j] if labels else (other.name or f"spectrum {i}[{j}]")
                (line,) = ax.plot(k_ref, pk_other[j], label=lab, color=color, **kwargs)
                artists.append(line)
                if ax_ratio:
                    (ratio_line,) = ax_ratio.plot(
                        k_ref, pk_other[j] / pk_ref, label=f"{lab} / ref", color=color, ls="--"
                    )
                    artists.append(ratio_line)

        if logx:
            ax.set_xscale("log")
            if ax_ratio:
                ax_ratio.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, which="both", ls=":", alpha=0.5)
        if (self.name or "").lower() == "cl":
            ax.set_xlabel(r"$\ell$")
            ax.set_ylabel(r"$C_\ell$")
        else:
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$P(k)$")
        ax.legend(loc="upper left")
        if ax_ratio:
            ax_ratio.set_ylabel("ratio")
            if ratio_ylim is not None:
                ax_ratio.set_ylim(*ratio_ylim)
            ax_ratio.legend(loc="lower right")

        return fig, ax, artists

    # ---- Stacking helper ------------------------------------------------
    @classmethod
    def stack(cls, power_spectra: Sequence[PowerSpectrum]) -> PowerSpectrum:
        """Stack multiple PowerSpectrum objects along a new leading axis.

        All wavenumber grids must match (allclose). Arrays are concatenated
        along batch axis (introducing a leading dimension if needed).
        """
        # Make sure that all wavenumber grids match and they have the same name
        ref_k = power_spectra[0].wavenumber
        name = power_spectra[0].name
        for spec in power_spectra[1:]:
            if spec.shape != power_spectra[0].shape:
                raise ValueError("All PowerSpectrum instances must share the same shape to be stacked.")
            if spec.name != name:
                raise ValueError("All PowerSpectrum instances must share the same name to be stacked.")

        stacked_array = jnp.stack([spec.array for spec in power_spectra], axis=0)
        return cls(wavenumber=ref_k, array=stacked_array, name=name)
