"""Plotting utilities for forward-modeling workflows.

This module centralizes the visualization routines used by the lensing workflow
so that scripts remain thin orchestrators. Each helper takes NumPy-compatible
arrays and writes plots to disk.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import arviz as az
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np




def plot_lightcones(lightcone, geometry: str, save_path: Path | str, max_planes: int = 6) -> None:
    """Plot a few representative lightcone planes.

    Parameters
    ----------
    lightcone : array-like
        Collection of density planes (num_planes, ...).
    geometry : str
        Either "spherical" (HEALPix) or "flat" (Cartesian).
    save_path : Path or str
        Destination for the PNG file.
    max_planes : int, optional
        Number of planes to show to avoid gigantic figures.
    """

    if lightcone is None:
        return

    save_path = Path(save_path)

    # If stacking failed (object dtype), fall back to list of planes

    planes = lightcone[:max_planes]
    if not planes:
        return

    if geometry == "spherical":
        cols = len(planes)
        plt.figure(figsize=(4 * cols, 3.5))
        for idx, plane in enumerate(planes, start=1):
            plane = np.asarray(plane).ravel()
            finite = plane[np.isfinite(plane)]
            if finite.size == 0:
                vmin, vmax = None, None
            else:
                vmin = np.percentile(finite, 2)
                vmax = np.percentile(finite, 98)
            hp.mollview(
                plane,
                sub=(1, cols, idx),
                cmap="viridis",
                title=f"Lightcone plane {idx}",
                min=vmin,
                max=vmax,
                bgcolor=(0, 0, 0, 0),
                cbar=True,
            )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        rows = math.ceil(len(planes) / 3)
        cols = min(len(planes), 3)
        plt.figure(figsize=(5 * cols, 4 * rows))
        for idx, plane in enumerate(planes, start=1):
            ax = plt.subplot(rows, cols, idx)
            im = ax.imshow(np.asarray(plane), origin="lower", cmap="viridis")
            ax.set_title(f"Lightcone plane {idx}")
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_kappa_maps(
    kappas: dict[str, np.ndarray] | Iterable[np.ndarray],
    geometry: str,
    save_path: Path | str,
    title_prefix: str = "",
) -> None:
    """Visualize convergence maps for all redshift bins."""

    save_path = Path(save_path)

    if isinstance(kappas, dict):
        ordered = [kappas[key] for key in sorted(kappas.keys())]
    else:
        ordered = list(kappas)

    ordered = [np.asarray(k) for k in ordered]
    if not ordered:
        return

    nbins = len(ordered)
    flat_kappas = np.concatenate([k.ravel() for k in ordered])
    valid = flat_kappas[np.isfinite(flat_kappas)]
    if valid.size:
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    else:
        vmin, vmax = None, None

    if geometry == "spherical":
        plt.figure(figsize=(4 * nbins, 3.5))
        for idx, kappa in enumerate(ordered, start=1):
            stats = (
                np.nanmin(kappa),
                np.nanmax(kappa),
                np.nanmean(kappa),
                np.nanstd(kappa),
            )
            hp.mollview(
                kappa.ravel(),
                sub=(1, nbins, idx),
                cmap="viridis",
                min=vmin,
                max=vmax,
                bgcolor=(0, 0, 0, 0),
                title=(
                    f"{title_prefix}Kappa {idx}\n"
                    f"min={stats[0]:.2e} max={stats[1]:.2e}\n"
                    f"mean={stats[2]:.2e} std={stats[3]:.2e}"
                ),
                cbar=True,
            )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(5 * nbins, 4))
        for idx, kappa in enumerate(ordered, start=1):
            ax = plt.subplot(1, nbins, idx)
            im = ax.imshow(kappa, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            stats = (
                np.nanmin(kappa),
                np.nanmax(kappa),
                np.nanmean(kappa),
                np.nanstd(kappa),
            )
            ax.set_title(
                f"{title_prefix}Kappa {idx}\n"
                f"min={stats[0]:.2e} max={stats[1]:.2e}\n"
                f"mean={stats[2]:.2e} std={stats[3]:.2e}"
            )
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_ic_summary(
    true_ic: np.ndarray,
    ic_samples: np.ndarray,
    save_path: Path | str,
    slice_axis: int = -1,
    slice_index: int | None = None,
) -> None:
    """Plot true IC vs posterior mean/std (single slice)."""

    save_path = Path(save_path)
    true_ic_np = _to_numpy(true_ic)
    ic_samples_np = _to_numpy(ic_samples)

    if ic_samples_np.ndim == true_ic_np.ndim:
        # Assume leading axis is samples
        sample_axis = 0
    else:
        raise ValueError("ic_samples must have a leading sample axis")

    mean_ic = np.mean(ic_samples_np, axis=sample_axis)
    std_ic = np.std(ic_samples_np, axis=sample_axis)

    axis = slice_axis if slice_axis >= 0 else true_ic_np.ndim + slice_axis
    if slice_index is None:
        slice_index = true_ic_np.shape[axis] // 2

    slicer = [slice(None)] * true_ic_np.ndim
    slicer[axis] = slice_index
    slicer = tuple(slicer)

    true_slice = np.asarray(true_ic_np[slicer])
    mean_slice = np.asarray(mean_ic[slicer])
    std_slice = np.asarray(std_ic[slicer])
    residual_slice = mean_slice - true_slice

    titles = [
        f"True IC (slice {slice_index})",
        "Posterior mean",
        "Posterior std",
        "Residual (mean - true)",
    ]
    slices = [true_slice, mean_slice, std_slice, residual_slice]

    plt.figure(figsize=(16, 4))
    for idx, (title, data) in enumerate(zip(titles, slices), start=1):
        ax = plt.subplot(1, 4, idx)
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_posterior_pair(
    omega_c_samples,
    sigma8_samples,
    save_path: Path | str,
    true_values: dict[str, float] | None = None,
) -> None:
    """Create an ArviZ pair plot for (Omega_c, sigma8)."""

    omega = np.asarray(omega_c_samples)
    sigma = np.asarray(sigma8_samples)

    if omega.ndim == 1:
        omega = omega[np.newaxis, :]
    if sigma.ndim == 1:
        sigma = sigma[np.newaxis, :]

    az_data = az.from_dict(posterior={"Omega_c": omega, "sigma8": sigma})

    plt.figure(figsize=(6, 6))
    az.plot_pair(
        az_data,
        var_names=["Omega_c", "sigma8"],
        kind="kde",
        marginals=True,
        reference_values=true_values,
        divergences=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
