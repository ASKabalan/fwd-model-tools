from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


__all__ = ["PowerSpectrum"]


@jax.tree_util.register_pytree_node_class
class PowerSpectrum:
    """
    Container for power spectrum data (P(k) or C_ell).

    Registered as JAX PyTree for compatibility with JAX transformations.
    """

    __slots__ = ("k", "pk", "label")

    def __init__(
        self,
        *,
        k: jax.Array,
        pk: jax.Array,
        label: str | None = None,
    ):
        """
        Initialize PowerSpectrum.

        Parameters
        ----------
        k : jax.Array
            Wavenumber (for P(k)) or multipole (for C_ell)
        pk : jax.Array
            Power spectrum values
        label : str, optional
            Label for plotting legends
        """
        self.k = jnp.asarray(k)
        self.pk = jnp.asarray(pk)
        self.label = label

        if self.k.shape != self.pk.shape:
            raise ValueError(
                f"k and pk must have same shape, got k={self.k.shape}, pk={self.pk.shape}"
            )

    def tree_flatten(self):
        children = (self.k, self.pk)
        aux_data = (self.label,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (label,) = aux_data
        k, pk = children
        return cls(k=k, pk=pk, label=label)

    def __repr__(self) -> str:
        return (
            f"PowerSpectrum("
            f"k=Array{tuple(self.k.shape)}, "
            f"pk=Array{tuple(self.pk.shape)}, "
            f"label={self.label!r})"
        )

    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        logx: bool = True,
        logy: bool = True,
        label: str | None = None,
        figsize: Tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot power spectrum.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, creates new figure.
        logx : bool, default=True
            Use log scale for x-axis (k or ell)
        logy : bool, default=True
            Use log scale for y-axis (P(k) or C_ell)
        label : str, optional
            Override label for legend. If None, uses self.label.
        figsize : tuple, optional
            Figure size (width, height) in inches. Only used if ax is None.
        **kwargs
            Additional arguments passed to ax.plot()

        Returns
        -------
        fig : plt.Figure
            Figure object
        ax : plt.Axes
            Axes object

        Raises
        ------
        ValueError
            If called within a jit context (array is traced)
        """
        import jax.core
        if not jax.core.is_concrete(self.k):
            raise ValueError(
                "Cannot plot traced arrays. Use outside of jit context."
            )

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
        else:
            fig = ax.get_figure()

        # Determine label
        plot_label = label if label is not None else self.label

        # Plot
        ax.plot(self.k, self.pk, label=plot_label, **kwargs)

        # Set scales
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        # Labels
        ax.set_xlabel(r"$k$ or $\ell$")
        ax.set_ylabel(r"$P(k)$ or $C_\ell$")

        if plot_label:
            ax.legend()

        return fig, ax

    def show(
        self,
        *,
        logx: bool = True,
        logy: bool = True,
        label: str | None = None,
        figsize: Tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot and display power spectrum.

        Parameters mirror :meth:`PowerSpectrum.plot`.

        Raises
        ------
        ValueError
            If called within a jit context (array is traced)
        """
        import jax.core
        if not jax.core.is_concrete(self.k):
            raise ValueError(
                "Cannot plot/show traced arrays. Use outside of jit context."
            )

        self.plot(
            logx=logx,
            logy=logy,
            label=label,
            figsize=figsize,
            **kwargs,
        )
        plt.show()

    def __mul__(self, other: float | jax.Array) -> "PowerSpectrum":
        """Multiply power spectrum by scalar or array (e.g., window function)."""
        return PowerSpectrum(
            k=self.k,
            pk=self.pk * other,
            label=self.label,
        )

    def __rmul__(self, other: float | jax.Array) -> "PowerSpectrum":
        """Right multiplication."""
        return self.__mul__(other)

    def __add__(self, other: "PowerSpectrum" | float) -> "PowerSpectrum":
        """Add two power spectra or add scalar."""
        if isinstance(other, PowerSpectrum):
            if not jnp.allclose(self.k, other.k):
                raise ValueError("Cannot add PowerSpectrum with different k values")
            return PowerSpectrum(
                k=self.k,
                pk=self.pk + other.pk,
                label=self.label,
            )
        else:
            return PowerSpectrum(
                k=self.k,
                pk=self.pk + other,
                label=self.label,
            )

    def __sub__(self, other: "PowerSpectrum" | float) -> "PowerSpectrum":
        """Subtract two power spectra or subtract scalar."""
        if isinstance(other, PowerSpectrum):
            if not jnp.allclose(self.k, other.k):
                raise ValueError("Cannot subtract PowerSpectrum with different k values")
            return PowerSpectrum(
                k=self.k,
                pk=self.pk - other.pk,
                label=self.label,
            )
        else:
            return PowerSpectrum(
                k=self.k,
                pk=self.pk - other,
                label=self.label,
            )
