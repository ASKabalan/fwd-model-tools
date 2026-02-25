"""Plotting utilities for field visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

# String keyword to attribute mapping for particle weights
WEIGHT_KEYWORDS = {
    "redshift": ("z_sources", "Redshift"),
    "z": ("z_sources", "Redshift"),
    "scale": ("scale_factors", "Scale Factor"),
    "a": ("scale_factors", "Scale Factor"),
    "comoving": ("comoving_centers", "Comoving Distance [Mpc/h]"),
    "r": ("comoving_centers", "Comoving Distance [Mpc/h]"),
}


def prepare_axes(
    ax, n_plots: int, ncols: int, projection: str | None = None, figsize: tuple[float, float] | None = None
):
    """
    Prepare matplotlib axes for batch plotting.

    If ax is None, creates new figure with grid layout.
    If ax is provided and projection='3d', replaces 2D axes with 3D axes at same position.

    Parameters
    ----------
    ax : None, Axes, ndarray, or Sequence
        Pre-existing axes or None to create new figure.
    n_plots : int
        Number of plots to create.
    ncols : int
        Number of columns in grid (ignored if ax provided).
    projection : str, optional
        Projection type (e.g., '3d'). If provided and axes exist, they are replaced.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes_flat : ndarray
        1D array of axes.
    """
    if ax is None:
        ncols_eff = max(1, min(ncols, n_plots))
        nrows = ceil(n_plots / ncols_eff)
        if figsize is None:
            figsize = (6 * ncols_eff, 6 * nrows)
        fig, axes = plt.subplots(
            nrows, ncols_eff, figsize=figsize, subplot_kw={"projection": projection}, squeeze=False
        )
        return fig, axes.ravel()
    else:
        # Flatten provided axes
        if isinstance(ax, np.ndarray):
            axes_flat = ax.ravel()
        elif isinstance(ax, Sequence):
            axes_flat = np.array(ax, dtype=object).ravel()
        else:
            axes_flat = np.array([ax], dtype=object)

        if axes_flat.size < n_plots:
            raise ValueError(f"Provided {axes_flat.size} axes but need {n_plots}")

        fig = axes_flat[0].get_figure()

        # If 3D projection requested, verify provided axes are 3D
        if projection == "3d":
            from mpl_toolkits.mplot3d import Axes3D

            for i, ax_i in enumerate(axes_flat[:n_plots]):
                if not isinstance(ax_i, Axes3D):
                    raise TypeError(
                        f"Expected Axes3D for 3D plotting, got {type(ax_i).__name__}. "
                        f"Create axes with projection='3d', e.g.: "
                        f"fig.add_subplot(..., projection='3d')"
                    )

        return fig, axes_flat


def plot_3d_density(
    ax,
    vol,
    project_slices=10,
    crop: tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
    labels: tuple[str, str, str] = ("X", "Y", "Z"),
    ticks: tuple[Sequence[float], Sequence[float], Sequence[float]] = ([], [], []),
    vmin=None,
    vmax=None,
    cmap="magma",
    elev=40,
    azim=-30,
    zoom=0.8,
    edges=True,
    colorbar=True,
    levels=64,
):
    """
    Plot 3 orthogonal faces of a 3D volume (Top, Front, Right) using contourf.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes (must have projection='3d').
    vol : array-like
        3D volume array with shape (Nx, Ny, Nz).
    x_slice, y_slice, z_slice : slice
        Slices to select subvolume.
    vmin, vmax : float, optional
        Color range limits.
    cmap : str
        Colormap name.
    elev, azim : float
        Elevation and azimuth angles for 3D view.
    edges : bool
        Whether to draw wireframe box edges.
    colorbar : bool
        Whether to show colorbar.
    levels : int
        Number of contour levels.

    Returns
    -------
    ax : Axes3D
        The axes with the plot.
    """
    data = np.asarray(vol)
    Nx, Ny, Nz = data.shape

    # Resolve slices to indices
    x_slice, y_slice, z_slice = crop
    x_start, x_stop, _ = x_slice.indices(Nx)
    y_start, y_stop, _ = y_slice.indices(Ny)
    z_start, z_stop, _ = z_slice.indices(Nz)

    # Coordinate system: Z is negative indices
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz), indexing="ij")

    # Global min/max for consistent colorbar
    _vmin = vmin if vmin is not None else data.min()
    _vmax = vmax if vmax is not None else data.max()
    kw = {"vmin": _vmin, "vmax": _vmax, "levels": np.linspace(_vmin, _vmax, levels), "cmap": cmap}

    # Face A: Top (Z plane at z_start)
    z_projection = slice(z_start, min(z_start + project_slices, z_stop))
    ax.contourf(
        X[x_slice, y_slice, z_start],
        Y[x_slice, y_slice, z_start],
        data[x_slice, y_slice, z_projection].mean(axis=2),
        zdir="z",
        offset=Z[0, 0, z_start],
        **kw,
    )

    # Face B: Front (Y plane at y_start)
    y_projection = slice(y_start, min(y_start + project_slices, y_stop))
    ax.contourf(
        X[x_slice, y_start, z_slice],
        data[x_slice, y_projection, z_slice].mean(axis=1),
        Z[x_slice, y_start, z_slice],
        zdir="y",
        offset=y_start,
        **kw,
    )

    # Face C: Right (X plane at x_stop - 1)
    x_face_idx = x_stop - 1
    x_projection = slice(max(x_start, x_face_idx - project_slices + 1), x_stop)
    C = ax.contourf(
        data[x_projection, y_slice, z_slice].mean(axis=0),
        Y[x_face_idx, y_slice, z_slice],
        Z[x_face_idx, y_slice, z_slice],
        zdir="x",
        offset=x_face_idx,
        **kw,
    )

    # Draw edges (wireframe box)
    if edges:
        x0, x1 = X[x_start, 0, 0], X[x_stop - 1, 0, 0]
        y0, y1 = Y[0, y_start, 0], Y[0, y_stop - 1, 0]
        z0, z1 = Z[0, 0, z_start], Z[0, 0, z_stop - 1]

        edge_kw = dict(color="0.4", linewidth=1.5, zorder=1e3)

        # X-lines (Hide ones behind the faces)
        ax.plot([x0, x1], [y0, y0], [z0, z0], **edge_kw)
        ax.plot([x0, x1], [y1, y1], [z0, z0], **edge_kw)
        # ax.plot([x0, x1], [y0, y0], [z1, z1], **edge_kw)
        # ax.plot([x0, x1], [y1, y1], [z1, z1], **edge_kw)

        # Y-lines
        ax.plot([x0, x0], [y0, y1], [z0, z0], **edge_kw)
        ax.plot([x1, x1], [y0, y1], [z0, z0], **edge_kw)
        # ax.plot([x0, x0], [y0, y1], [z1, z1], **edge_kw)
        ax.plot([x1, x1], [y0, y1], [z1, z1], **edge_kw)

        # Z-lines
        ax.plot([x0, x0], [y0, y0], [z0, z1], **edge_kw)
        ax.plot([x1, x1], [y0, y0], [z0, z1], **edge_kw)
        # ax.plot([x0, x0], [y1, y1], [z0, z1], **edge_kw)
        ax.plot([x1, x1], [y1, y1], [z0, z1], **edge_kw)

    # Styling
    ax.grid(False)
    ax.set_xlim(X[x_start, 0, 0], X[x_stop - 1, 0, 0])
    ax.set_ylim(Y[0, y_start, 0], Y[0, y_stop - 1, 0])
    ax.set_zlim(Z[0, 0, z_stop - 1], Z[0, 0, z_start])  # Z is negative
    # Axes labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    # Axes ticks
    ax.set_xticks(ticks[0])
    ax.set_yticks(ticks[1])
    ax.set_zticks(ticks[2])
    # View angle
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)

    if colorbar:
        fig = ax.get_figure()
        fig.colorbar(C, ax=ax, fraction=0.046, pad=0.04)

    return ax


def resolve_particle_weights(
    weights: None | float | str | np.ndarray,
    particles_shape: tuple,
    z_sources=None,
    scale_factors=None,
    comoving_centers=None,
    weights_title: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Resolve weights parameter to array and title for particle plotting.

    Parameters
    ----------
    weights : None, float, str, or array-like
        Weight specification:
        - None: No coloring
        - float: Scalar color for all particles
        - str: Keyword ('redshift', 'z', 'scale', 'a', 'comoving', 'r')
        - array: Direct weight values
    particles_shape : tuple
        Shape of the particles array (Nx, Ny, Nz, 3).
    z_sources : array-like, optional
        Source redshifts (used if weights='redshift' or 'z').
    scale_factors : array-like, optional
        Scale factors (used if weights='scale' or 'a').
    comoving_centers : array-like, optional
        Comoving distances (used if weights='comoving' or 'r').
    weights_title : str, optional
        Override for colorbar title. If None and weights is a string keyword,
        a default title is used.

    Returns
    -------
    resolved_weights : ndarray or None
        Weight values broadcastable to (Nx, Ny, Nz).
    title : str or None
        Colorbar title.

    Raises
    ------
    ValueError
        If string keyword is used but corresponding attribute is None.
    """
    if weights is None:
        return None, None

    # Handle string keywords
    if isinstance(weights, str):
        key = weights.lower()
        if key not in WEIGHT_KEYWORDS:
            raise ValueError(f"Unknown weight keyword '{weights}'. Valid options: {list(WEIGHT_KEYWORDS.keys())}")
        attr_name, default_title = WEIGHT_KEYWORDS[key]
        attr_map = {
            "z_sources": z_sources,
            "scale_factors": scale_factors,
            "comoving_centers": comoving_centers,
        }
        weights = attr_map[attr_name]
        if weights_title is None:
            weights_title = default_title

    # Convert to numpy
    weights = np.asarray(weights)

    # Handle scalar
    if weights.ndim == 0:
        return weights, weights_title

    # Broadcast 1D array (Z,) to (1, 1, Z) for Z-axis coloring
    Nx, Ny, Nz = particles_shape[:3]
    if weights.squeeze().ndim == 1:
        if weights.shape[0] == Nz:
            weights = weights[None, None, :]
        else:
            raise ValueError(f"1D weights array has length {weights.shape[0]}, expected {Nz} (Nz dimension).")

    # Squeeze trailing dimension if present (X, Y, Z, 1) -> (X, Y, Z)
    if weights.ndim == 4 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)

    # Broadcast to full shape
    weights = np.broadcast_to(weights, (Nx, Ny, Nz))

    return weights, weights_title


def generate_titles(
    base_title: str,
    scale_factors: float | Sequence[float] | np.ndarray | None,
    n_plots: int,
) -> list[str]:
    """
    Generate a list of titles for plots, optionally including scale factor information.

    Parameters
    ----------
    base_title : str
        The base text for the title (e.g. "3D Density", "Pk").
    scale_factors : float, sequence, or array, optional
        Scale factor(s) corresponding to the plots.
    n_plots : int
        Total number of plots/titles to generate.

    Returns
    -------
    titles : list[str]
        List of title strings.
    """
    sfs = None
    if scale_factors is not None:
        sfs = np.atleast_1d(np.array(scale_factors))

    titles = []
    for i in range(n_plots):
        if sfs is not None and sfs.ndim == 1 and i < len(sfs):
            titles.append(f"{base_title} at a={sfs[i]:.3f}")
        elif n_plots > 1:
            titles.append(f"{base_title} ({i})")
        else:
            titles.append(base_title)

    return titles


def plot_3d_particles(
    ax,
    particles,
    weights=None,
    thinning=1,
    cmap="viridis",
    point_size=5,
    alpha=0.6,
    elev=40,
    azim=-30,
    zoom=0.8,
    weights_title=None,
    labels=("X", "Y", "Z"),
    ticks=([], [], []),
    colorbar=True,
    vmin=None,
    vmax=None,
):
    """
    Plot a 3D scatter of particles with uniform spatial thinning.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes (must have projection='3d').
    particles : array-like
        Particle positions with shape (Nx, Ny, Nz, 3).
    weights : array-like, optional
        Color values, broadcastable to (Nx, Ny, Nz).
    thinning : int
        Take every n-th particle along each axis.
    cmap : str
        Colormap name.
    point_size : float
        Size of scatter points.
    alpha : float
        Transparency of points.
    elev, azim : float
        Elevation and azimuth angles for 3D view.
    zoom : float
        Zoom factor for the 3D view.
    weights_title : str, optional
        Label for colorbar.
    labels : tuple of str
        Axis labels (X, Y, Z).
    ticks : tuple of sequences
        Tick values for each axis.
    colorbar : bool
        Whether to show colorbar.
    vmin, vmax : float, optional
        Color range limits.

    Returns
    -------
    ax : Axes3D
        The axes with the plot.
    """
    pos = np.asarray(particles)

    # Apply thinning (stride slicing)
    if thinning > 1:
        pos = pos[::thinning, ::thinning, ::thinning, :]
        if weights is not None:
            weights = np.asarray(weights)
            if weights.ndim >= 1:
                weights = weights[::thinning, ::thinning, ::thinning]

    # Flatten for scatter plot
    X_flat = pos[..., 0].flatten()
    Y_flat = pos[..., 1].flatten()
    Z_flat = pos[..., 2].flatten()

    # Handle colors
    c_data = None
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim == 0:
            # Scalar: broadcast to all points
            c_data = np.full(X_flat.shape, float(weights))
        else:
            c_data = weights.flatten()

    # Draw scatter
    scatter_kw = {
        "s": point_size,
        "alpha": alpha,
        "cmap": cmap,
        "edgecolors": "none",
    }
    if c_data is not None:
        scatter_kw["c"] = c_data

    if vmin is not None:
        scatter_kw["vmin"] = vmin
    if vmax is not None:
        scatter_kw["vmax"] = vmax

    sc = ax.scatter(
        X_flat,
        Y_flat,
        Z_flat,
        **scatter_kw,
    )

    # Add colorbar if weights were provided
    if c_data is not None and colorbar:
        fig = ax.get_figure()
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if weights_title is not None:
            cbar.set_label(weights_title)

    # Styling
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    if ticks is not None:
        ax.set_xticks(ticks[0])
        ax.set_yticks(ticks[1])
        ax.set_zticks(ticks[2])

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)

    return ax


def plot_flat_density(
    ax,
    data,
    cmap="magma",
    vmin=None,
    vmax=None,
    show_colorbar=True,
    show_ticks=True,
    title=None,
):
    """Plot a single 2D flat-sky density map using imshow."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.03)
        ax.get_figure().colorbar(im, cax=cax)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def plot_spherical_density(
    ax,
    data,
    cmap="magma",
    vmin=None,
    vmax=None,
    show_colorbar=True,
    show_ticks=True,
    title="",
):
    """Plot a single HEALPix spherical map using healpy.mollview.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    data : array-like
        HEALPix map data.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color range limits.
    show_colorbar : bool
        Whether to show colorbar.
    show_ticks : bool
        Whether to show graticule (coordinate grid lines) on the Mollweide projection.
    title : str
        Title for the plot.
    """
    import healpy as hp

    def _sub_from_ax(ax_obj):
        sp = ax_obj.get_subplotspec()
        gs = sp.get_gridspec()
        row = sp.rowspan.start
        col = sp.colspan.start
        return gs.nrows, gs.ncols, row * gs.ncols + col + 1

    def _attach_delegate(ax_obj, delegate):
        ax_obj._healpy_delegate = delegate
        ax_obj.set_title = delegate.set_title
        ax_obj.get_title = delegate.get_title

    fig = ax.get_figure()
    map_np = np.asarray(data)

    ax.axis("off")
    sub = _sub_from_ax(ax)
    delegate = hp.mollview(
        map_np,
        fig=fig,
        sub=sub,
        cmap=cmap,
        title=title,
        bgcolor=(0.0,) * 4,
        cbar=show_colorbar,
        min=vmin if vmin is not None else 0,
        max=vmax
        if vmax is not None
        else (np.percentile(map_np[map_np > 0], 95) if np.any(map_np > 0) else np.max(map_np)),
    )
    if delegate is None:
        delegate = next(
            (a for a in fig.axes if isinstance(a, hp.projaxes.HpxMollweideAxes)),
            None,
        )
    if delegate is None:
        raise RuntimeError("healpy.mollview did not return a Mollweide axes.")
    _attach_delegate(ax, delegate)

    # Control graticule display
    if not show_ticks:
        hp.graticule(verbose=False, alpha=0)

    return ax
