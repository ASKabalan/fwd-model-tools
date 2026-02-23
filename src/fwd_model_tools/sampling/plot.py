"""Sampling-related plotting helpers (posteriors, IC comparisons, chain traces)."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples
from getdist import plots as gdplots

# THIS FILE IS WIP


def _strip_dollars(label: str) -> str:
    """Remove surrounding $ from LaTeX labels (GetDist adds them internally)."""
    return label.strip("$")


def _resolve_params(cosmo_dict: dict, params: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return params list, defaulting to all keys in cosmo_dict."""
    if params is None:
        return list(cosmo_dict.keys())
    return list(params)


def _build_mcsamples(
    cosmo_dict: dict,
    params: list[str],
    labels: dict | None = None,
) -> MCSamples:
    """Build a GetDist MCSamples from cosmo_dict for the given params.

    For single-chain input (values are 1-D arrays), builds a single MCSamples.
    For multi-chain input (values are 2-D arrays of shape (n_chains, n_samples)),
    passes a list of per-chain arrays to MCSamples so GetDist treats them as
    separate chains for convergence diagnostics.
    """
    sample_0 = np.asarray(cosmo_dict[params[0]])
    is_multi_chain = sample_0.ndim == 2

    if labels is None:
        labels_list = list(params)
    else:
        labels_list = [_strip_dollars(labels.get(p, p)) for p in params]

    if is_multi_chain:
        n_chains = sample_0.shape[0]
        chain_arrays = [np.column_stack([np.asarray(cosmo_dict[p])[c] for p in params]) for c in range(n_chains)]
        return MCSamples(samples=chain_arrays, names=list(params), labels=labels_list)
    else:
        arr = np.column_stack([np.asarray(cosmo_dict[p]) for p in params])
        return MCSamples(samples=arr, names=list(params), labels=labels_list)


def plot_ic(
    true_ic,
    mean_ic,
    std_ic,
    outdir,
    titles=("True", "Mean", "Std", "Diff"),
    output_format: str = "png",
    dpi: int = 600,
):
    """Plot initial conditions: true, posterior mean, posterior std, and residual.

    Parameters
    ----------
    true_ic : DensityField or array_like
        The true initial conditions (shape (X, Y, Z)).
    mean_ic : DensityField or array_like
        Posterior mean of initial conditions.
    std_ic : DensityField or array_like
        Posterior standard deviation of initial conditions.
    outdir : str or Path
        Output directory.
    titles : tuple of str, optional
        Panel titles.
    output_format : str, optional
        "png", "pdf", or "show".
    dpi : int, optional
        DPI for saved figures.
    """
    # TODO: implement IC comparison plot accepting DensityField or raw arrays
    pass


def plot_chains(
    cosmo_dict: dict,
    outdir: str,
    params: list[str] | None = None,
    labels: dict | None = None,
    output_format: str = "png",
    dpi: int = 600,
):
    """Plot MCMC chain traces for convergence visualization.

    Parameters
    ----------
    cosmo_dict : dict
        Mapping from parameter name to array of shape ``(n_samples,)`` (single
        chain) or ``(n_chains, n_samples)`` (multi-chain).
    outdir : str or Path
        Output directory for saved plots.
    params : list of str, optional
        Parameters to plot. Defaults to all keys in ``cosmo_dict``.
    labels : dict, optional
        Mapping from parameter name to LaTeX label string.
    output_format : str, optional
        "png", "pdf", or "show".
    dpi : int, optional
        DPI for saved figures.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = _resolve_params(cosmo_dict, params)
    n_params = len(params)

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for ax, p in zip(axes, params):
        samples = np.asarray(cosmo_dict[p])
        if labels is not None and p in labels:
            ylabel = f"${_strip_dollars(labels[p])}$"
        else:
            ylabel = p
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if samples.ndim == 2:
            # Multi-chain: (n_chains, n_samples) — one colored line per chain
            for c_idx, chain in enumerate(samples):
                ax.plot(chain, lw=0.5, alpha=0.8, label=f"Chain {c_idx}")
            ax.legend(fontsize=6, loc="upper right")
        else:
            ax.plot(samples, lw=0.5, alpha=0.8)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    if output_format == "show":
        plt.show()
    else:
        plt.savefig(outdir / f"chain_trace.{output_format}", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ess(
    cosmo_dict: dict,
    outdir: str,
    params: list[str] | None = None,
    labels: dict | None = None,
    output_format: str = "png",
    dpi: int = 600,
):
    """Plot effective sample size (ESS) per parameter via GetDist autocorrelation.

    ESS is computed as ``n_total / getCorrelationLength(i)`` for each parameter.

    Parameters
    ----------
    cosmo_dict : dict
        Parameter samples, single- or multi-chain (auto-detected from ndim).
    outdir : str or Path
        Output directory.
    params : list of str, optional
        Parameters to include. Defaults to all keys.
    labels : dict, optional
        LaTeX labels per parameter.
    output_format : str, optional
        "png", "pdf", or "show".
    dpi : int, optional
        DPI for saved figures.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = _resolve_params(cosmo_dict, params)
    mc_samples = _build_mcsamples(cosmo_dict, params, labels=labels)

    n_total = len(mc_samples.samples)
    ess_values = {}
    for i, p in enumerate(params):
        corr_len = mc_samples.getCorrelationLength(i)
        ess_values[p] = n_total / corr_len if corr_len > 0 else float("nan")

    param_labels = [f"${_strip_dollars(labels.get(p, p))}$" if labels else p for p in params]
    values = [ess_values[p] for p in params]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(params))))
    ax.barh(param_labels, values)
    ax.set_xlabel("Effective Sample Size (ESS)")
    ax.set_title("ESS per parameter")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if output_format == "show":
        plt.show()
    else:
        plt.savefig(outdir / f"ess.{output_format}", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_rhat(
    cosmo_dict: dict,
    outdir: str,
    params: list[str] | None = None,
    labels: dict | None = None,
    threshold: float = 1.01,
    output_format: str = "png",
    dpi: int = 600,
):
    """Plot per-parameter Gelman-Rubin R-hat convergence statistic.

    Requires multi-chain input (``n_chains >= 2``). Warns and returns early for
    single-chain data.

    R-hat is computed via the standard Gelman-Rubin PSRF formula:
    ``sqrt(V̂ / W)`` where ``V̂ = (n-1)/n * W + B/n``.

    Parameters
    ----------
    cosmo_dict : dict
        Parameter samples of shape ``(n_chains, n_samples)`` per key.
    outdir : str or Path
        Output directory.
    params : list of str, optional
        Parameters to include.
    labels : dict, optional
        LaTeX labels per parameter.
    threshold : float, optional
        Primary reference line (default 1.01).
    output_format : str, optional
        "png", "pdf", or "show".
    dpi : int, optional
        DPI for saved figures.
    """
    sample_0 = np.asarray(cosmo_dict[next(iter(cosmo_dict))])
    if sample_0.ndim != 2:
        warnings.warn("plot_rhat requires multi-chain input (n_chains >= 2). Returning early.", stacklevel=2)
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = _resolve_params(cosmo_dict, params)
    n_chains = sample_0.shape[0]

    rhat_values: dict[str, float] = {}
    for p in params:
        chain_arrays = [np.asarray(cosmo_dict[p])[c] for c in range(n_chains)]
        n = len(chain_arrays[0])
        chain_means = [np.mean(c) for c in chain_arrays]
        chain_vars = [np.var(c, ddof=1) for c in chain_arrays]
        W = float(np.mean(chain_vars))
        B = n * float(np.var(chain_means, ddof=1))
        if W > 0:
            v_hat = ((n - 1) / n) * W + B / n
            rhat_values[p] = float(np.sqrt(v_hat / W))
        else:
            rhat_values[p] = float("nan")

    param_labels = [f"${_strip_dollars(labels.get(p, p))}$" if labels else p for p in params]
    values = [rhat_values[p] for p in params]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(params))))
    ax.barh(param_labels, values)
    ax.axvline(threshold, color="orange", ls="--", label=f"R-hat = {threshold}")
    ax.axvline(1.05, color="red", ls="--", label="R-hat = 1.05")
    ax.set_xlabel("R-hat")
    ax.set_title("Gelman-Rubin R-hat")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if output_format == "show":
        plt.show()
    else:
        plt.savefig(outdir / f"rhat.{output_format}", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_pair(
    cosmo_dict: dict,
    outdir: str,
    params: list[str] | None = None,
    truth: dict | None = None,
    labels: dict | None = None,
    output_format: str = "png",
    dpi: int = 600,
    filled: bool = True,
    contour_colors: list | None = None,
    title_limit: int = 1,
    width_inch: float = 7.0,
):
    """Plot posterior pair (triangle) plot using GetDist.

    Replaces the old ``plot_posterior``. Multi-chain arrays are flattened to a
    single chain before plotting.

    Parameters
    ----------
    cosmo_dict : dict
        Parameter samples, single- or multi-chain.
    outdir : str or Path
        Output directory.
    params : list of str, optional
        Parameters to include.
    truth : dict, optional
        True values per parameter, shown as red dashed markers.
        E.g. ``{"Omega_c": 0.27, "sigma8": 0.8}``.
    labels : dict, optional
        LaTeX labels per parameter.
    output_format : str, optional
        "png", "pdf", or "show".
    dpi : int, optional
        DPI for saved figures.
    filled : bool, optional
        Use filled contours. Default True.
    contour_colors : list, optional
        Colors for contour levels.
    title_limit : int, optional
        Marginalized limit to print on diagonal panels. Default 1.
    width_inch : float, optional
        Figure width in inches. Default 7.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = _resolve_params(cosmo_dict, params)

    # Flatten multi-chain to single chain for pair plot
    flat_dict: dict = {}
    for p in params:
        arr = np.asarray(cosmo_dict[p])
        flat_dict[p] = arr.flatten() if arr.ndim == 2 else arr

    mc_samples = _build_mcsamples(flat_dict, params, labels=labels)

    gdplt = gdplots.get_subplot_plotter(width_inch=width_inch)

    plot_kwargs: dict = {"filled": filled, "title_limit": title_limit}
    if truth is not None:
        markers_dict = {p: truth[p] for p in params if p in truth}
        if markers_dict:
            plot_kwargs["markers"] = markers_dict
            plot_kwargs["marker_args"] = {"color": "red", "ls": "--"}
    if contour_colors:
        plot_kwargs["contour_colors"] = contour_colors

    gdplt.triangle_plot([mc_samples], params, **plot_kwargs)

    if output_format == "show":
        plt.show()
    else:
        plt.savefig(outdir / f"pair_plot.{output_format}", dpi=dpi, bbox_inches="tight")
    plt.close()
