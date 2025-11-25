from __future__ import annotations

from math import ceil
from typing import Any, Sequence, Tuple

import jax
import jax.core
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["PowerSpectrum"]


@jax.tree_util.register_pytree_node_class
class PowerSpectrum:
    """
    Container for power spectrum data (P(k), C_ell, transfer, coherence, ...).

    Supports both single and batched spectra and is registered as a JAX PyTree
    for compatibility with JAX transformations.
    """

    __slots__ = ("wavenumber", "spectra", "name")

    def __init__(
        self,
        *,
        wavenumber: jax.Array,
        spectra: jax.Array,
        name: str | None = None,
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
        """
        self.wavenumber = jnp.asarray(wavenumber)
        self.spectra = jnp.asarray(spectra)
        self.name = name

        self._validate_shapes()

    # ---- PyTree protocol -------------------------------------------------
    def tree_flatten(self):
        children = (self.wavenumber, self.spectra)
        aux_data = (self.name, )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (name, ) = aux_data
        wavenumber, spectra = children
        return cls(wavenumber=wavenumber, spectra=spectra, name=name)

    # ---- Internal helpers -------------------------------------------------
    def _validate_shapes(self) -> None:
        """Validate wavenumber / spectra compatibility (supports batched spectra)."""
        if self.wavenumber.ndim != 1:
            raise ValueError("wavenumber must be 1D for spectra validation")

        n_k = self.wavenumber.shape[0]

        if self.spectra.shape == self.wavenumber.shape:
            return

        if self.spectra.ndim == 2 and (self.spectra.shape[0] == n_k or self.spectra.shape[1] == n_k):
            return

        raise ValueError("Incompatible shapes for wavenumber and spectra. "
                         f"Got wavenumber={self.wavenumber.shape}, spectra={self.spectra.shape}. "
                         "Expected spectra to be 1D with n_k or 2D with one axis equal to n_k.")

    # ---- Representation ---------------------------------------------------
    def __repr__(self) -> str:
        return ("PowerSpectrum("
                f"wavenumber=Array{tuple(self.wavenumber.shape)}, "
                f"spectra=Array{tuple(self.spectra.shape)}, "
                f"name={self.name!r})")

    def _spectra_batch_first(self) -> tuple[jax.Array, bool]:
        """Return spectra as (n_spec, n_k), transposing if k is leading."""
        pk = self.spectra
        n_k = self.wavenumber.shape[0]
        transposed = False
        if pk.ndim == 1:
            pk_batched = pk[None, :]
        elif pk.ndim == 2:
            if pk.shape[1] == n_k:
                pk_batched = pk
            elif pk.shape[0] == n_k:
                pk_batched = pk.T
                transposed = True
            else:
                raise ValueError("Incompatible shapes for wavenumber and spectra. "
                                 f"Got wavenumber={self.wavenumber.shape}, spectra={pk.shape}.")
        else:
            raise ValueError("PowerSpectrum supports 1D or 2D spectra")
        return pk_batched, transposed

    def __getitem__(self, key) -> "PowerSpectrum":
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
            spec_sel, k_sel = key
        else:
            spec_sel, k_sel = slice(None), key

        k_new = self.wavenumber[k_sel]
        pk_batched, transposed = self._spectra_batch_first()

        pk_sel = pk_batched[spec_sel, k_sel]

        # Restore original orientation
        if pk_sel.ndim == 1:
            spectra_out = pk_sel
        else:
            spectra_out = pk_sel.T if transposed else pk_sel

        return PowerSpectrum(wavenumber=k_new, spectra=spectra_out, name=self.name)

    # ---- Plotting ---------------------------------------------------------
    def plot(
        self,
        *,
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        logx: bool = True,
        logy: bool = True,
        label: str | None = None,
        figsize: Tuple[float, float] | None = None,
        mode: str = "subplots",
        ncols: int = 3,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        alpha: float = 0.3,
        grid: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, Any, list[Any]]:
        """Plot power spectrum(s).

        Modes
        -----
        - "subplots" (default): one subplot per spectrum with ncols heuristic.
        - "overlay": all spectra on a single axis.
        - "mean_std": mean curve with ±1σ band (uses colors[0] or next cycle).
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.plot() outside of a jit context.")

        k_1d = self.wavenumber
        n_k = k_1d.shape[0]
        if self.spectra.ndim == 1:
            if self.spectra.shape[0] != n_k:
                raise ValueError("wavenumber and spectra must match for 1D spectra, "
                                 f"got wavenumber={k_1d.shape}, spectra={self.spectra.shape}.")
            pk_2d = self.spectra[None, :]
        elif self.spectra.ndim == 2:
            if self.spectra.shape[1] == n_k:
                pk_2d = self.spectra
            elif self.spectra.shape[0] == n_k:
                pk_2d = self.spectra
            else:
                raise ValueError("Could not interpret spectra as a stack over wavenumber. "
                                 f"wavenumber.shape={k_1d.shape}, spectra.shape={self.spectra.shape}.")
        else:
            raise ValueError("spectra must be 1D or 2D for plotting")
        n_spec = pk_2d.shape[0]
        artists: list[Any] = []

        if mode not in {"subplots", "overlay", "mean_std"}:
            raise ValueError("mode must be 'subplots', 'overlay', or 'mean_std'.")

        if mode != "overlay" and labels is not None and len(labels) != n_spec:
            raise ValueError(f"Expected {n_spec} labels, got {len(labels)}.")

        def _flatten_axes(axes_obj):
            if axes_obj is None:
                return None
            if isinstance(axes_obj, np.ndarray):
                return axes_obj.ravel()
            if isinstance(axes_obj, Sequence):
                return np.array(axes_obj, dtype=object).ravel()
            return np.array([axes_obj], dtype=object)

        def _get_color(ax_obj, i: int) -> Any:
            if colors is not None and i < len(colors):
                return colors[i]
            return ax_obj._get_lines.get_next_color()

        # Determine axes
        if mode == "subplots" and n_spec > 1:
            axes_flat = _flatten_axes(ax)
            if axes_flat is not None:
                if axes_flat.size < n_spec:
                    raise ValueError("Provided axes array is too small for subplots mode.")
                fig = axes_flat[0].get_figure()
                axes = axes_flat[:n_spec]
            else:
                ncols_eff = max(1, min(ncols or 1, n_spec))
                nrows = ceil(n_spec / ncols_eff)
                fig, axes_grid = plt.subplots(
                    nrows,
                    ncols_eff,
                    figsize=figsize or (5 * ncols_eff, 4 * nrows),
                    squeeze=False,
                )
                axes = axes_grid.ravel()

            for i in range(axes.size):
                if i >= n_spec:
                    axes[i].axis("off")
                    continue
                lab = (labels[i] if labels is not None else (self.name and f"{self.name} {i}") or f"Spectrum {i}")
                color = _get_color(axes[i], i)
                line, = axes[i].loglog(k_1d, pk_2d[i], label=lab, color=color, **kwargs)
                artists.append(line)
                if logx:
                    axes[i].set_xscale("log")
                if logy:
                    axes[i].set_yscale("log")
                if grid:
                    axes[i].grid(True, which="both", ls=":", alpha=0.5)
                axes[i].set_xlabel(r"$k$ or $\ell$")
                axes[i].set_ylabel(r"$P(k)$ or $C_\ell$")
                axes[i].legend()

            # reshape axes to a grid for return when we created them
            if axes_flat is None:
                ncols_eff = max(1, min(ncols or 1, n_spec))
                nrows = ceil(n_spec / ncols_eff)
                axes = np.reshape(axes, (nrows, ncols_eff))
            return fig, axes, artists

        # Overlay or mean_std modes share a single axis
        axes_flat = _flatten_axes(ax)
        if axes_flat is None:
            fig, ax_single = plt.subplots(figsize=figsize or (8, 6))
        else:
            ax_single = axes_flat[0]
            fig = ax_single.get_figure()

        if mode == "overlay":
            for i in range(n_spec):
                lab = ((label if label is not None and n_spec == 1 else None)
                       or (labels[i] if labels is not None else None) or (self.name and f"{self.name} {i}")
                       or f"Spectrum {i}")
                color = _get_color(ax_single, i)
                line, = ax_single.loglog(k_1d, pk_2d[i], label=lab, color=color, **kwargs)
                artists.append(line)
        elif mode == "mean_std":
            if n_spec == 1:
                plot_label = label or self.name
                line, = ax_single.loglog(k_1d, pk_2d[0], label=plot_label, **kwargs)
                artists.append(line)
            else:
                mean_pk = pk_2d.mean(axis=0)
                std_pk = pk_2d.std(axis=0)
                plot_label = label or self.name or "mean"
                color = colors[0] if colors else kwargs.get("color", None)
                if color is None:
                    color = _get_color(ax_single, 0)
                line, = ax_single.loglog(k_1d, mean_pk, label=plot_label, color=color, **kwargs)
                band = ax_single.fill_between(
                    k_1d,
                    mean_pk - std_pk,
                    mean_pk + std_pk,
                    color=color,
                    alpha=alpha,
                )
                artists.extend([line, band])

        if logx:
            ax_single.set_xscale("log")
        if logy:
            ax_single.set_yscale("log")
        if grid:
            ax_single.grid(True, which="both", ls=":", alpha=0.5)
        if title:
            ax_single.set_title(title)

        ax_single.set_xlabel(r"$k$ or $\ell$")
        ax_single.set_ylabel(r"$P(k)$ or $C_\ell$")

        if any((artist.get_label() not in ("", "_nolegend_") for artist in artists)):
            ax_single.legend()

        return fig, ax_single, artists

    def show(
        self,
        *,
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        logx: bool = True,
        logy: bool = True,
        label: str | None = None,
        figsize: Tuple[float, float] | None = None,
        mode: str = "subplots",
        ncols: int = 3,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        alpha: float = 0.3,
        grid: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot and display power spectrum(s).

        Parameters mirror :meth:`PowerSpectrum.plot`.
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.show() outside of a jit context.")

        self.plot(
            ax=ax,
            logx=logx,
            logy=logy,
            label=label,
            figsize=figsize,
            mode=mode,
            ncols=ncols,
            labels=labels,
            colors=colors,
            alpha=alpha,
            grid=grid,
            title=title,
            **kwargs,
        )
        plt.show()

    # ---- Arithmetic -------------------------------------------------------
    def _binary_op(self, other, op) -> "PowerSpectrum":
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

    def __mul__(self, other) -> "PowerSpectrum":
        """Multiply by scalar, array, or PowerSpectrum."""
        return self._binary_op(other, jnp.multiply)

    def __rmul__(self, other) -> "PowerSpectrum":
        return self.__mul__(other)

    def __add__(self, other) -> "PowerSpectrum":
        """Add scalar, array, or PowerSpectrum."""
        return self._binary_op(other, jnp.add)

    def __radd__(self, other) -> "PowerSpectrum":
        return self.__add__(other)

    def __sub__(self, other) -> "PowerSpectrum":
        """Subtract scalar, array, or PowerSpectrum."""
        return self._binary_op(other, jnp.subtract)

    def __rsub__(self, other) -> "PowerSpectrum":
        return PowerSpectrum(
            wavenumber=self.wavenumber,
            spectra=jnp.subtract(other, self.spectra),
            name=self.name,
        )

    # ---- Comparison helper ------------------------------------------------
    def compare(
        self,
        other_spectra: Sequence["PowerSpectrum"],
        *,
        show_ratio: bool = True,
        shaded_regions: Sequence[float] = (0.1, 0.2),
        labels: Sequence[str] | None = None,
        figsize: Tuple[float, float] | None = None,
        logx: bool = True,
        logy: bool = True,
        grid: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, list[plt.Axes], list[Any]]:
        """
        Compare this spectrum to a list of other spectra.
        """
        if not jax.core.is_concrete(self.wavenumber):
            raise ValueError("Cannot plot traced arrays. Use PowerSpectrum.compare() outside of a jit context.")

        if not other_spectra:
            raise ValueError("other_spectra must contain at least one PowerSpectrum.")

        if labels is not None and len(labels) != len(other_spectra):
            raise ValueError(f"Expected {len(other_spectra)} labels, got {len(labels)}.")

        k_ref = self.wavenumber
        n_k = k_ref.shape[0]
        if self.spectra.ndim == 1:
            if self.spectra.shape[0] != n_k:
                raise ValueError("wavenumber and spectra must match for 1D spectra, "
                                 f"got wavenumber={k_ref.shape}, spectra={self.spectra.shape}.")
            pk_ref = self.spectra
        elif self.spectra.ndim == 2 and self.spectra.shape[0] == 1 and self.spectra.shape[1] == n_k:
            pk_ref = self.spectra[0]
        elif self.spectra.ndim == 2 and self.spectra.shape[1] == n_k and self.spectra.shape[0] == 1:
            pk_ref = self.spectra[0]
        else:
            raise ValueError("compare() currently requires a single spectrum aligned with wavenumber.")

        if show_ratio:
            fig, (ax_main, ax_ratio) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={
                    "height_ratios": [3, 1],
                    "hspace": 0.05
                },
                figsize=figsize or (8, 6),
            )
            axes = [ax_main, ax_ratio]
        else:
            fig, ax_main = plt.subplots(figsize=figsize or (8, 5))
            ax_ratio = None
            axes = [ax_main]

        artists: list[Any] = []

        def _plot_main(ax_obj, x, y, **line_kwargs):
            if logx and logy:
                return ax_obj.loglog(x, y, **line_kwargs)
            if logx:
                return ax_obj.semilogx(x, y, **line_kwargs)
            if logy:
                return ax_obj.semilogy(x, y, **line_kwargs)
            return ax_obj.plot(x, y, **line_kwargs)

        def _plot_ratio(ax_obj, x, y, **line_kwargs):
            if logx:
                return ax_obj.semilogx(x, y, **line_kwargs)
            return ax_obj.plot(x, y, **line_kwargs)

        # Plot reference spectrum
        ref_label = self.name or "reference"
        ref_line, = _plot_main(ax_main, k_ref, pk_ref, color="k", lw=2, label=ref_label)
        artists.append(ref_line)

        # ratio = 1 reference line
        if show_ratio:
            one_line = ax_ratio.axhline(1.0, color="k", lw=1, ls="--")
            artists.append(one_line)

        # Plot other spectra and ratios
        for i, other in enumerate(other_spectra):
            k_other = other.wavenumber
            if k_other.shape != k_ref.shape or not jnp.allclose(k_ref, k_other):
                raise ValueError("All spectra in compare() must share the same wavenumber grid.")

            if other.spectra.ndim == 1 and other.spectra.shape[0] == n_k:
                pk_other = other.spectra
            elif other.spectra.ndim == 2 and other.spectra.shape[0] == 1 and other.spectra.shape[1] == n_k:
                pk_other = other.spectra[0]
            elif other.spectra.ndim == 2 and other.spectra.shape[1] == n_k and other.spectra.shape[0] == 1:
                pk_other = other.spectra[0]
            else:
                raise ValueError(
                    "compare() currently supports only single-spectrum PowerSpectrum instances in other_spectra.")

            color = ax_main._get_lines.get_next_color()
            lab = (labels[i] if labels is not None else (other.name or f"spectrum {i}"))

            line_main, = _plot_main(ax_main, k_ref, pk_other, color=color, label=lab, **kwargs)
            artists.append(line_main)

            if show_ratio:
                ratio = pk_other / pk_ref
                line_ratio, = _plot_ratio(ax_ratio, k_ref, ratio, color=color, label=lab)
                artists.append(line_ratio)

        # Grid styling
        if grid:
            ax_main.grid(True, which="both", ls=":", alpha=0.5)
            if show_ratio:
                ax_ratio.grid(True, which="both", ls=":", alpha=0.5)

        ax_main.set_ylabel(r"$P(k)$ or $C_\ell$")
        if show_ratio:
            ax_ratio.set_xlabel(r"$k$ or $\ell$")
        else:
            ax_main.set_xlabel(r"$k$ or $\ell$")

        # Ratio panel styling
        if show_ratio:
            ax_ratio.set_yscale("linear")
            ax_ratio.set_ylabel("ratio")
            for frac in sorted(shaded_regions):
                band = ax_ratio.fill_between(
                    k_ref,
                    1.0 - frac,
                    1.0 + frac,
                    color="gray",
                    alpha=0.1,
                    zorder=0,
                )
                artists.append(band)

        # Legend on main axis
        ax_main.legend()

        if title:
            ax_main.set_title(title)

        return fig, axes, artists

    @classmethod
    def stack(cls, spectra: Sequence["PowerSpectrum"]) -> "PowerSpectrum":
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
