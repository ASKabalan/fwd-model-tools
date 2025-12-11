from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .._src.fields._plotting import generate_titles

__all__ = ["PowerSpectrum"]


@jax.tree_util.register_pytree_node_class
class PowerSpectrum:
    """
    Container for power spectrum data (P(k), C_ell, transfer, coherence, ...).

    Supports both single and batched spectra and is registered as a JAX PyTree
    for compatibility with JAX transformations.
    """

    __slots__ = ("wavenumber", "spectra", "name", "scale_factors")

    def __init__(
        self,
        *,
        wavenumber: jax.Array,
        spectra: jax.Array,
        name: str | None = None,
        scale_factors: Any | None = None,
    ):
        """
        Parameters
        ----------
        wavenumber : jax.Array
            Wavenumber (for P(k)) or multipole (for C_ell).
        spectra : jax.Array
            Power spectrum values. Can be 1D (n_k,) for a single spectrum
            or 2D for multiple spectra, in which case one dimension must
            match ``wavenumber.size`` (either (n_spectra, n_k) or (n_k, n_spectra)).
        name : str, optional
            Name/type of the spectrum, e.g. "pk", "cl",
            "transfer", "coherence".
        scale_factors : any, optional
            Scale factors associated with the spectra (for batched inputs).
        """
        self.wavenumber = wavenumber
        self.spectra = spectra
        self.name = name
        self.scale_factors = scale_factors

        self._validate_shapes()

    # ---- PyTree protocol -------------------------------------------------
    def tree_flatten(self):
        children = (self.wavenumber, self.spectra, self.scale_factors)
        aux_data = (self.name,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (name,) = aux_data
        wavenumber, spectra, scale_factors = children
        return cls(wavenumber=wavenumber, spectra=spectra, name=name, scale_factors=scale_factors)

    # ---- Internal helpers -------------------------------------------------
    def _validate_shapes(self) -> None:
        """Validate wavenumber / spectra compatibility."""
        if self.wavenumber.ndim != 1:
            raise ValueError("wavenumber must be 1D")

        n_k = self.wavenumber.shape[0]

        if self.spectra.ndim == 1:
            if self.spectra.shape[0] != n_k:
                raise ValueError(f"Spectra length {self.spectra.shape[0]} does not match wavenumber {n_k}.")
            return

        if self.spectra.ndim == 2:
            if self.spectra.shape[1] != n_k:
                raise ValueError(
                    f"Spectra shape {self.spectra.shape} incompatible with wavenumber {n_k}. Use shape (n_spec, n_k)."
                )
            return

        raise ValueError("Spectra must be 1D or 2D.")

    # ---- Representation ---------------------------------------------------
    def __repr__(self) -> str:
        return (
            "PowerSpectrum("
            f"wavenumber=Array{tuple(self.wavenumber.shape)}, "
            f"spectra=Array{tuple(self.spectra.shape)}, "
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
        if self.spectra.ndim == 1:
            spectra_out = self.spectra[k_sel]
        else:
            spectra_out = self.spectra[k_sel, spec_sel]
            
        if self.scale_factors is not None and self.spectra.squeeze().ndim == 1:
            sf_new = jnp.atleast_1d(self.scale_factors[k_sel])
        else:
            sf_new = self.scale_factors

        return PowerSpectrum(wavenumber=k_new, spectra=spectra_out, name=self.name, scale_factors=sf_new)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the power spectra array."""
        return self.spectra.shape


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
        pk_2d = self.spectra[None, :] if self.spectra.ndim == 1 else self.spectra
        n_spec = pk_2d.shape[0]

        if label is None:
             base_name = self.name or "spectrum"
             label = generate_titles(base_name, self.scale_factors, n_spec)

        if not isinstance(label, (list, tuple)):
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
        pk_2d = self.spectra[None, :] if self.spectra.ndim == 1 else self.spectra
        n_spec = pk_2d.shape[0]

        if label is not None:
            if not isinstance(label, (list, tuple)):
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
        pk_ref = self.spectra[None, :] if self.spectra.ndim == 1 else self.spectra
        if pk_ref.shape[0] != 1:
            raise ValueError("compare_plot expects the reference spectrum to be unbatched.")
        pk_ref = pk_ref[0]

        if labels is not None:
            if len(labels) != len(others):
                raise ValueError(f"Expected {len(others)} label rows, got {len(labels)}.")
            for i, lbls in enumerate(labels):
                pk_other = others[i].spectra[None, :] if others[i].spectra.ndim == 1 else others[i].spectra
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
            pk_other = other.spectra[None, :] if other.spectra.ndim == 1 else other.spectra
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

    # ---- Arithmetic -------------------------------------------------------
    def _binary_op(self, other, op) -> PowerSpectrum:
        """Helper for elementwise binary ops with validation."""
        if isinstance(other, PowerSpectrum):
            if not jnp.allclose(self.wavenumber, other.wavenumber):
                raise ValueError("PowerSpectrum operations require matching wavenumbers")
            rhs = other.spectra
        else:
            rhs = other
        return PowerSpectrum(
            wavenumber=self.wavenumber,
            spectra=op(self.spectra, rhs),
            name=self.name,
        )

    def __mul__(self, other) -> PowerSpectrum:
        """Multiply by scalar, array, or PowerSpectrum."""
        return self._binary_op(other, jnp.multiply)

    def __rmul__(self, other) -> PowerSpectrum:
        return self.__mul__(other)

    def __add__(self, other) -> PowerSpectrum:
        """Add scalar, array, or PowerSpectrum."""
        return self._binary_op(other, jnp.add)

    def __radd__(self, other) -> PowerSpectrum:
        return self.__add__(other)

    def __sub__(self, other) -> PowerSpectrum:
        """Subtract scalar, array, or PowerSpectrum."""
        return self._binary_op(other, jnp.subtract)

    def __rsub__(self, other) -> PowerSpectrum:
        return PowerSpectrum(
            wavenumber=self.wavenumber,
            spectra=jnp.subtract(other, self.spectra),
            name=self.name,
        )

    # ---- Comparison helper ------------------------------------------------
    @classmethod
    def stack(cls, spectra: Sequence[PowerSpectrum]) -> PowerSpectrum:
        """Stack multiple PowerSpectrum objects along a new leading axis.

        All wavenumber grids must match (allclose). Spectra are concatenated
        along batch axis (introducing a leading dimension if needed).
        """
        # Make sure that all wavenumber grids match and they have the same name
        ref_k = spectra[0].wavenumber
        name = spectra[0].name
        for spec in spectra[1:]:
            if spec.shape != spectra[0].shape:
                raise ValueError("All PowerSpectrum instances must share the same wavenumber grid to be stacked.")
            if spec.name != name:
                raise ValueError("All PowerSpectrum instances must share the same name to be stacked.")

        spectra = jnp.stack([spec.spectra for spec in spectra], axis=0)
        return cls(wavenumber=ref_k, spectra=spectra, name=name)
