"""Lightweight plotting utilities for forward-modeling workflows."""

from pathlib import Path

import arviz as az
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def plot_kappa(kappa, outdir, spherical=False, titles=None):
    """Plot convergence maps.

    Parameters
    ----------
    kappa : array_like
        Shape (n_kappa, H, W) for flat geometry or (n_kappa, npix) for spherical.
    outdir : str or Path
        Output directory for saved plots.
    spherical : bool, optional
        If True, use HEALPix mollview projection. Default is False (flat).
    titles : list of str, optional
        Custom titles for each kappa map. If None, uses "Kappa 0", "Kappa 1", etc.
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
        plt.savefig(outdir / "kappa_maps.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(outdir / "kappa_maps.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_lightcone(lightcone, outdir, spherical=False, titles=None):
    """Plot lightcone density planes.

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
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lightcone = np.asarray(lightcone)
    n_planes = lightcone.shape[0]

    if titles is None:
        titles = [f"Plane {i}" for i in range(n_planes)]

    vmin, vmax = np.percentile(lightcone[np.isfinite(lightcone)], [2, 98])

    if spherical:
        fig = plt.figure(figsize=(4 * n_planes, 3.5))
        for i in range(n_planes):
            hp.mollview(
                lightcone[i].ravel(),
                sub=(1, n_planes, i + 1),
                cmap="viridis",
                title=titles[i],
                min=vmin,
                max=vmax,
                cbar=True,
            )
        plt.savefig(outdir / "lightcone.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(outdir / "lightcone.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_ic(true_ic, samples_ic, outdir, titles=("True", "Mean", "Std")):
    """Plot initial conditions: true, posterior mean, and posterior std.

    Parameters
    ----------
    true_ic : array_like
        Shape (X, Y, Z), the true initial conditions.
    samples_ic : array_like
        Shape (n_samples, X, Y, Z), sampled initial conditions.
    outdir : str or Path
        Output directory for saved plots.
    titles : tuple of str, optional
        Titles for the three panels. Default is ("True", "Mean", "Std").
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    true_ic = np.asarray(true_ic)
    samples_ic = np.asarray(samples_ic)

    mean_ic = samples_ic.mean(axis=0)
    std_ic = samples_ic.std(axis=0)

    slice_idx = true_ic.shape[-1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    data = [
        true_ic[..., slice_idx], mean_ic[..., slice_idx], std_ic[...,
                                                                 slice_idx]
    ]

    for ax, d, title in zip(axes, data, titles):
        im = ax.imshow(d, origin="lower", cmap="viridis")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(outdir / "ic_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def prepare_arviz_data(samples, params=None):
    """Prepare ArviZ InferenceData from sample dictionary.

    Parameters
    ----------
    samples : dict
        Dictionary mapping parameter names to arrays. Arrays can be 1D (n_samples,)
        or 2D (n_chains, n_samples). Higher-dimensional arrays are filtered out.
    params : tuple of str, optional
        Parameter names to include. If None, includes all 1D parameters.

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object ready for plotting.
    """
    scalar_keys = [
        k for k in samples.keys() if np.asarray(samples[k]).ndim == 1
    ]

    if params is not None:
        scalar_keys = [k for k in scalar_keys if k in params]

    posterior_dict = {}
    for k in scalar_keys:
        arr = np.asarray(samples[k])
        if arr.ndim == 1:
            posterior_dict[k] = arr[None, :]
        else:
            posterior_dict[k] = arr

    return az.from_dict(posterior=posterior_dict)


def plot_posterior(param_samples,
                   outdir,
                   params=("Omega_c", "sigma8"),
                   true_values=None):
    """Plot posterior distributions using ArviZ.

    Parameters
    ----------
    param_samples : dict
        Dictionary mapping parameter names to arrays of shape (n_samples,).
    outdir : str or Path
        Output directory for saved plots.
    params : tuple of str, optional
        Parameter names to plot. Default is ("Omega_c", "sigma8").
    true_values : dict, optional
        Dictionary mapping parameter names to true values. If provided,
        true values will be plotted as red stars on the pair plot.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    posterior_dict = {
        k: np.asarray(v)[None, :]
        for k, v in param_samples.items() if k in params
    }

    idata = az.from_dict(posterior=posterior_dict)

    fig, axes = plt.subplots(len(params), 2, figsize=(12, 4 * len(params)))
    if len(params) == 1:
        axes = axes[None, :]

    az.plot_trace(idata, var_names=list(params), axes=axes)

    if true_values is not None:
        for i, param in enumerate(params):
            if param in true_values:
                axes[i, 0].axvline(true_values[param],
                                   color="red",
                                   linestyle="--",
                                   linewidth=2,
                                   label="True value")
                axes[i, 0].legend()

    plt.tight_layout()
    plt.savefig(outdir / "posterior_trace.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 4))
    axis = az.plot_pair(
        idata,
        var_names=list(params),
        kind="kde",
        marginals=True,
        divergences=False,
        reference_values=true_values,
    )

    plt.tight_layout()
    plt.savefig(outdir / "posterior_pair.png", dpi=150, bbox_inches="tight")
    plt.close()
