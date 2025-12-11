"""Sampling-related plotting helpers (posteriors, IC comparisons, chain traces)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples
from getdist import plots as gdplots


def _strip_dollars(label: str) -> str:
    """Remove surrounding $ from LaTeX labels (GetDist adds them internally)."""
    return label.strip("$")


def plot_ic(
    true_ic,
    mean_ic,
    std_ic,
    outdir,
    titles=("True", "Mean", "Std", "Diff"),
    output_format: str = "png",
    dpi: int = 600,
):
    """Plot initial conditions: true, posterior mean, and posterior std.

    Parameters
    ----------
    true_ic : array_like
        Shape (X, Y, Z), the true initial conditions.
    mean_ic : array_like
        Shape (X, Y, Z), posterior mean of initial conditions.
    std_ic : array_like
        Shape (X, Y, Z), posterior standard deviation of initial conditions.
    outdir : str or Path
        Output directory for saved plots.
    titles : tuple of str, optional
        Titles for the four panels. Default is ("True", "Mean", "Std", "Diff").
    output_format : str, optional
        Output format: "png", "pdf", or "show". Default is "png".
    dpi : int, optional
        DPI for saved figures. Default is 600.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    true_ic = np.asarray(true_ic)
    mean_ic = np.asarray(mean_ic)
    std_ic = np.asarray(std_ic)

    diff_ic = mean_ic - true_ic

    slice_idx = 3 * true_ic.shape[-1] // 4

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    data = [
        true_ic[..., slice_idx],
        mean_ic[..., slice_idx],
        std_ic[..., slice_idx],
        diff_ic[..., slice_idx],
    ]

    for ax, d, title in zip(axes, data, titles):
        im = ax.imshow(d, origin="lower", cmap="viridis")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if output_format == "show":
        plt.show()
    else:
        filename = f"ic_comparison.{output_format}"
        plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_posterior(
    param_samples,
    outdir,
    params=("Omega_c", "sigma8"),
    true_values=None,
    labels=None,
    output_format: str = "png",
    dpi: int = 600,
    filled: bool = True,
    contour_colors=None,
    title_limit: int = 1,
    width_inch: float = 7.0,
):
    """Plot posterior distributions using GetDist triangle plot.

    Parameters
    ----------
    param_samples : dict
        Dictionary mapping parameter names to arrays of shape (n_samples,).
    outdir : str or Path
        Output directory for saved plots.
    params : tuple of str, optional
        Parameter names to plot. Default is ("Omega_c", "sigma8").
    true_values : dict, optional
        Dictionary mapping parameter names to true values. These will be
        shown as markers on the plots.
    labels : dict, optional
        Dictionary mapping parameter names to LaTeX labels for plotting.
        If None, uses parameter names directly.
    output_format : str, optional
        Output format: "png", "pdf", or "show". Default is "png".
    dpi : int, optional
        DPI for saved figures. Default is 600.
    filled : bool, optional
        Whether to use filled contours. Default is True.
    contour_colors : list, optional
        List of colors for contours. Default is None (uses GetDist defaults).
    title_limit : int, optional
        Print marginalized limit on diagonal 1D plots. Default is 1.
    width_inch : float, optional
        Width of the plot in inches. Default is 7.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples_array = np.column_stack([param_samples[p] for p in params])
    names = list(params)

    if labels is None:
        labels_list = names
    else:
        labels_list = [_strip_dollars(labels.get(p, p)) for p in params]

    mc_samples = MCSamples(samples=samples_array, names=names, labels=labels_list)

    markers_dict = None
    if true_values is not None:
        markers_dict = {p: true_values[p] for p in params if p in true_values}

    gdplt = gdplots.get_subplot_plotter(width_inch=width_inch)

    plot_kwargs = {
        "filled": filled,
        "title_limit": title_limit,
    }

    if markers_dict:
        plot_kwargs["markers"] = markers_dict

    if contour_colors:
        plot_kwargs["contour_colors"] = contour_colors

    gdplt.triangle_plot([mc_samples], **plot_kwargs)

    if output_format == "show":
        plt.show()
    else:
        filename = f"posterior.{output_format}"
        plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_chain(
    param_samples,
    outdir,
    params=("Omega_c", "sigma8"),
    labels=None,
    output_format: str = "png",
    dpi: int = 600,
):
    """Plot MCMC chain traces for convergence visualization.

    Parameters
    ----------
    param_samples : dict
        Dictionary mapping parameter names to arrays of shape (n_samples,).
    outdir : str or Path
        Output directory for saved plots.
    params : tuple of str, optional
        Parameter names to plot. Default is ("Omega_c", "sigma8").
    labels : dict, optional
        Dictionary mapping parameter names to LaTeX labels for plotting.
        If None, uses parameter names directly.
    output_format : str, optional
        Output format: "png", "pdf", or "show". Default is "png".
    dpi : int, optional
        DPI for saved figures. Default is 600.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for ax, p in zip(axes, params):
        samples = np.asarray(param_samples[p])
        ax.plot(samples, lw=0.5, alpha=0.8)
        if labels is not None and p in labels:
            ylabel = f"${_strip_dollars(labels[p])}$"
        else:
            ylabel = p
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    if output_format == "show":
        plt.show()
    else:
        filename = f"chain_trace.{output_format}"
        plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
    plt.close()
