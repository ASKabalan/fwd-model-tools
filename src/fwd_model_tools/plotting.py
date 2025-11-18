"""Lightweight plotting utilities for forward-modeling workflows."""

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples
from getdist import plots as gdplots


def plot_kappa(kappa,
               outdir,
               spherical=False,
               titles=None,
               output_format="png",
               dpi=600):
    """Plot convergence maps (legacy wrapper).

    This helper operates directly on numpy arrays and is kept for backward
    compatibility. New code should prefer calling ``.plot`` / ``.show`` on
    ``FlatDensity`` / ``SphericalDensity`` / kappa field instances from
    :mod:`fwd_model_tools.fields`.

    Parameters
    ----------
    kappa : array_like
        Shape (n_kappa, H, W) for flat geometry, (n_kappa, npix) for full spherical maps,
        requires full maps for HEALPix be sure to reconstruct full map before plotting.
    outdir : str or Path
        Output directory for saved plots.
    spherical : bool, optional
        If True, use HEALPix mollview projection. Default is False (flat).
    titles : list of str, optional
        Custom titles for each kappa map. If None, uses "Kappa 0", "Kappa 1", etc.
    output_format : str, optional
        Output format: "png", "pdf", or "show". Default is "png".
    dpi : int, optional
        DPI for saved figures. Default is 600.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kappa = np.asarray(kappa)
    n_kappa = kappa.shape[0]

    if titles is None:
        titles = [f"Kappa {i}" for i in range(n_kappa)]

    vmin, vmax = np.percentile(kappa[np.isfinite(kappa)], [2, 98])

    if spherical:
        fig = plt.figure(figsize=(4 * n_kappa, 3.5))
        for i in range(n_kappa):
            hp.mollview(
                kappa[i].ravel(),
                sub=(1, n_kappa, i + 1),
                cmap="viridis",
                title=titles[i],
                min=vmin,
                max=vmax,
                cbar=True,
            )
        if output_format == "show":
            plt.show()
        else:
            filename = f"kappa_maps.{output_format}"
            plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        fig, axes = plt.subplots(1, n_kappa, figsize=(5 * n_kappa, 4))
        if n_kappa == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(kappa[i],
                           origin="lower",
                           cmap="viridis",
                           vmin=vmin,
                           vmax=vmax)
            ax.set_title(titles[i])
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if output_format == "show":
            plt.show()
        else:
            filename = f"kappa_maps.{output_format}"
            plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_lightcone(lightcone,
                   outdir,
                   spherical=False,
                   titles=None,
                   output_format="png",
                   dpi=600):
    """Plot lightcone density planes (legacy wrapper).

    This helper operates directly on numpy arrays and is kept for backward
    compatibility. New code should prefer calling ``.plot`` / ``.show`` on
    density field instances from :mod:`fwd_model_tools.fields`.

    Parameters
    ----------
    lightcone : array_like
        Shape (n_planes, H, W) for flat or (n_planes, npix) for spherical.
    outdir : str or Path
        Output directory for saved plots.
    spherical : bool, optional
        If True, use HEALPix mollview projection. Default is False (flat).
    titles : list of str, optional
        Custom titles for each plane. If None, uses "Plane 0", "Plane 1", etc.
    output_format : str, optional
        Output format: "png", "pdf", or "show". Default is "png".
    dpi : int, optional
        DPI for saved figures. Default is 600.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lightcone = np.asarray(lightcone)
    n_planes = lightcone.shape[0]

    if titles is None:
        titles = [f"Plane {i}" for i in range(n_planes)]

    vmin, vmax = np.percentile(lightcone[np.isfinite(lightcone)], [2, 98])

    if spherical:
        rows = (n_planes + 2) // 3
        cols = min(n_planes, 3)
        fig = plt.figure(figsize=(4 * cols, 3.5 * rows))
        for i in range(n_planes):
            plt.subplot(rows, cols, i + 1)
            hp.mollview(
                lightcone[i].ravel(),
                cmap="viridis",
                title=titles[i],
                min=vmin,
                max=vmax,
                cbar=True,
                hold=True,
            )
        if output_format == "show":
            plt.show()
        else:
            filename = f"lightcone.{output_format}"
            plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        rows = (n_planes + 2) // 3
        cols = min(n_planes, 3)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(n_planes):
            im = axes[i].imshow(lightcone[i],
                                origin="lower",
                                cmap="viridis",
                                vmin=vmin,
                                vmax=vmax)
            axes[i].set_title(titles[i])
            plt.colorbar(im, ax=axes[i])
        for j in range(n_planes, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        if output_format == "show":
            plt.show()
        else:
            filename = f"lightcone.{output_format}"
            plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_ic(true_ic,
            mean_ic,
            std_ic,
            outdir,
            titles=("True", "Mean", "Std", "Diff"),
            output_format="png",
            dpi=600):
    """Shim for backwards-compatibility; use fwd_model_tools.sampling.plot.plot_ic instead."""
    from fwd_model_tools.sampling.plot import plot_ic as _plot_ic

    return _plot_ic(
        true_ic=true_ic,
        mean_ic=mean_ic,
        std_ic=std_ic,
        outdir=outdir,
        titles=titles,
        output_format=output_format,
        dpi=dpi,
    )


def plot_gradient_analysis(
    results,
    params_info,
    outdir,
    output_format="png",
    dpi=600,
):
    """Plot gradient analysis results in a 2x2 grid.

    Parameters
    ----------
    results : dict
        Dictionary with parameter names as keys, each containing:
        - 'offsets': array of parameter offsets
        - 'losses': array of MSE loss values
        - 'gradients': array of gradient values
    params_info : dict
        Dictionary with parameter names as keys, each containing:
        - 'fiducial': fiducial parameter value
        - 'offset': offset magnitude
    outdir : str or Path
        Output directory for saved plots.
    output_format : str, optional
        Output format: "png", "pdf", or "show". Default is "png".
    dpi : int, optional
        DPI for saved figures. Default is 600.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    param_names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, param_name in enumerate(param_names):
        data = results[param_name]
        fiducial_val = params_info[param_name]["fiducial"]

        offsets = np.asarray(data["offsets"])
        losses = np.asarray(data["losses"])
        gradients = np.asarray(data["gradients"])
        param_values = fiducial_val + offsets

        ax_loss = axes[i, 0]
        ax_loss.plot(offsets, losses, "o-", linewidth=2, markersize=8)
        ax_loss.axvline(0,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="Fiducial")
        ax_loss.set_xlabel(f"{param_name} offset")
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.set_title(f"Loss vs {param_name} offset")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()

        ax_grad = axes[i, 1]
        ax_grad.plot(offsets,
                     gradients,
                     "s-",
                     linewidth=2,
                     markersize=8,
                     color="orange")
        ax_grad.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax_grad.axvline(0,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="Fiducial")
        ax_grad.set_xlabel(f"{param_name} offset")
        ax_grad.set_ylabel("d(MSE)/d(" + param_name + ")")
        ax_grad.set_title(f"Gradient vs {param_name} offset")
        ax_grad.grid(True, alpha=0.3)
        ax_grad.legend()

    plt.tight_layout()
    if output_format == "show":
        plt.show()
    else:
        filename = f"gradient_analysis.{output_format}"
        plt.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_posterior(
    param_samples,
    outdir,
    params=("Omega_c", "sigma8"),
    true_values=None,
    labels=None,
    output_format="png",
    dpi=600,
    filled=True,
    contour_colors=None,
    title_limit=1,
    width_inch=7,
):
    """Shim for backwards-compatibility; use fwd_model_tools.sampling.plot.plot_posterior instead."""
    from fwd_model_tools.sampling.plot import plot_posterior as _plot_posterior

    return _plot_posterior(
        param_samples=param_samples,
        outdir=outdir,
        params=params,
        true_values=true_values,
        labels=labels,
        output_format=output_format,
        dpi=dpi,
        filled=filled,
        contour_colors=contour_colors,
        title_limit=title_limit,
        width_inch=width_inch,
    )
