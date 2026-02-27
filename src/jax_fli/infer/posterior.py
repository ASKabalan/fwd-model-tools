"""GetDist posterior plotting helpers for CatalogExtract results."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np

try:
    from getdist import MCSamples
except ImportError:
    pass

from ..io.extract import CatalogExtract

__all__ = ["build_mcsamples", "plot_posterior", "requires_getdist"]

_Param = ParamSpec("_Param")
_Return = TypeVar("_Return")


def requires_getdist(func: Callable[_Param, _Return]) -> Callable[_Param, _Return]:
    """Decorator that raises ImportError when 'getdist' is not installed."""
    try:
        import getdist  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def _deferred(*args: _Param.args, **kwargs: _Param.kwargs) -> _Return:
        raise ImportError("Missing optional dependency 'getdist'. Install with: pip install fwd-model-tools[plot]")

    return _deferred


def _strip_dollars(label: str) -> str:
    return label.strip("$")


@requires_getdist
def build_mcsamples(
    catalog_extracts: CatalogExtract | list[CatalogExtract],
    params: list[str] | None = None,
    labels: dict | None = None,
    model_labels: list[str] | None = None,
) -> list[MCSamples]:
    """Build one GetDist MCSamples per CatalogExtract.

    Chains within each extract are **concatenated** (flattened) so GetDist
    draws KDE from all draws collectively.  The ``label`` attribute of each
    MCSamples is set to the corresponding entry in ``model_labels``, which
    GetDist uses to build the triangle-plot legend.

    Parameters
    ----------
    catalog_extracts : CatalogExtract or list[CatalogExtract]
        One or more extract results from ``extract_catalog()``.
    params : list of str, optional
        Parameters to include.  Defaults to all keys in the first extract.
    labels : dict, optional
        LaTeX labels per parameter, e.g. ``{"Omega_c": r"\\Omega_c"}``.
    model_labels : list of str, optional
        Legend labels per model.  Defaults to ``None`` per model (no legend entry).

    Returns
    -------
    list[MCSamples]
        One ``MCSamples`` object per extract.
    """
    from getdist import MCSamples

    if isinstance(catalog_extracts, CatalogExtract):
        catalog_extracts = [catalog_extracts]
    if model_labels is None:
        model_labels = [None] * len(catalog_extracts)
    if params is None:
        params = catalog_extracts[0].cosmo_keys

    result = []
    for ce, lbl in zip(catalog_extracts, model_labels):
        param_labels = [_strip_dollars(labels.get(p, p) if labels else p) for p in params]
        flat_samples = np.column_stack([np.asarray(ce.cosmo[p]).flatten() for p in params])
        mc = MCSamples(samples=flat_samples, names=params, labels=param_labels, label=lbl)
        result.append(mc)
    return result


@requires_getdist
def plot_posterior(
    catalog_extracts: CatalogExtract | list[CatalogExtract],
    outpath: str | Path | None = None,
    params: list[str] | None = None,
    labels: dict | None = None,
    model_labels: list[str] | None = None,
    truth: dict | None = None,
    filled: bool = True,
    shaded: bool = False,
    title_limit: int = 1,
    width_inch: float = 7.0,
    fontsize: int = 14,
    dpi: int = 300,
    output_format: str = "png",
) -> None:
    """Plot a GetDist triangle (pair) plot for one or more CatalogExtracts.

    Parameters
    ----------
    catalog_extracts : CatalogExtract or list[CatalogExtract]
        One or more extract results.  Multiple models are overlaid in one plot.
    outpath : str or Path, optional
        Full output file path.  If *None*, calls ``plt.show()``.
    params : list of str, optional
        Parameters to include.  Defaults to all keys in the first extract.
    labels : dict, optional
        LaTeX labels per parameter, e.g. ``{"Omega_c": r"\\Omega_c"}``.
    model_labels : list of str, optional
        Legend labels per model.  Pass a list when overlaying multiple models.
    truth : dict, optional
        True cosmology values shown as red dashed markers,
        e.g. ``{"Omega_c": 0.27, "sigma8": 0.8}``.
    filled : bool, optional
        Use filled contours.  Default ``True``.
    shaded : bool, optional
        Use shaded 1-D marginals.  Default ``False``.
    title_limit : int, optional
        Marginalized limit to print on diagonal panels.  Default ``1``.
    width_inch : float, optional
        Figure width in inches.  Default ``7.0``.
    fontsize : int, optional
        Base font size.  Default ``14``.
    dpi : int, optional
        Resolution for saved figures.  Default ``300``.
    output_format : str, optional
        Image format used when ``outpath`` has no extension.  Default ``"png"``.
    """
    import matplotlib.pyplot as plt
    from getdist import plots as gdplots
    from getdist.plots import GetDistPlotSettings

    if isinstance(catalog_extracts, CatalogExtract):
        catalog_extracts = [catalog_extracts]
    if params is None:
        params = catalog_extracts[0].cosmo_keys

    all_mc = build_mcsamples(catalog_extracts, params=params, labels=labels, model_labels=model_labels)

    settings = GetDistPlotSettings()
    settings.fontsize = fontsize
    settings.axes_fontsize = fontsize - 2
    settings.legend_fontsize = fontsize - 2
    settings.title_limit_fontsize = fontsize - 2
    settings.linewidth = 1.5
    settings.linewidth_contour = 1.5
    settings.tight_layout = True

    gdplt = gdplots.get_subplot_plotter(width_inch=width_inch, settings=settings)

    try:
        plot_kwargs = {
            "filled": filled,
            "shaded": shaded,
            "title_limit": title_limit,
            "legend_labels": model_labels,
        }
        if truth:
            markers = {p: truth[p] for p in params if p in truth}
            if markers:
                plot_kwargs["markers"] = markers
                plot_kwargs["marker_args"] = {"color": "red", "ls": "--", "lw": 1.5}

        gdplt.triangle_plot(all_mc, params, **plot_kwargs)

        if outpath is None:
            plt.show()
        else:
            outpath = Path(outpath)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        warnings.warn(f"plot_posterior skipped — GetDist raised: {exc}", stacklevel=2)
        plt.close()
