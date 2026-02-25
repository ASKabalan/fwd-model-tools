"""High-level analysis orchestrator: produce a complete report folder from a CatalogExtract."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np

from ..io.extract import CatalogExtract

__all__ = ["analyze", "requires_arviz"]

_Param = ParamSpec("_Param")
_Return = TypeVar("_Return")


def requires_arviz(func: Callable[_Param, _Return]) -> Callable[_Param, _Return]:
    """Decorator that raises ImportError when 'arviz' is not installed."""
    try:
        import arviz  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def _deferred(*args: _Param.args, **kwargs: _Param.kwargs) -> _Return:
        raise ImportError("Missing optional dependency 'arviz'. Install with: pip install fwd-model-tools[plot]")

    return _deferred


@requires_arviz
def analyze(
    catalog_extract: list[CatalogExtract] | CatalogExtract,
    outfolder: str,
    outformat: str = "png",
    extract_labels: list[str] | None = None,
    dpi: int = 300,
    truth: dict | None = None,
) -> None:
    """Produce a complete report folder from one or more CatalogExtracts.

    Writes per-model diagnostic outputs to *outfolder*:

    * ``field_projections_{label}.{fmt}`` — per-chain field projection panels (if ``mean_field`` available)
    * ``power_spectra_{label}.{fmt}``     — transfer function and coherence plots (if ``power_spectra`` available)
    * ``rank_plot_{label}.{fmt}``         — ArviZ rank diagnostic
    * ``trace_plot_{label}.{fmt}``        — ArviZ chain trace plot
    * ``summary.md``                      — Markdown table with a section per model

    For the GetDist triangle/pair plot call ``plot_posterior()`` separately, which
    supports multi-model overlay:

    .. code-block:: python

        analyze([result_a, result_b], "reports/")
        plot_posterior([result_a, result_b], outpath="reports/pair_plot.png",
                       model_labels=["Model A", "Model B"])

    Parameters
    ----------
    catalog_extract : CatalogExtract or list[CatalogExtract]
        Output(s) from ``extract_catalog()``.
    outfolder : str
        Output directory.  Created if absent.
    outformat : str, optional
        Image format: ``"png"``, ``"pdf"``, etc.  Default ``"png"``.
    extract_labels : list of str, optional
        Human-readable label per model, used in filenames and the summary.
        Defaults to ``["model_0", "model_1", ...]``.
    dpi : int, optional
        Resolution for saved figures.  Default 300.
    labels : dict, optional
        LaTeX labels per cosmo parameter, e.g. ``{"Omega_c": r"\\Omega_c"}``.
    truth : dict, optional
        True cosmology values (currently unused in diagnostics; pass to
        ``plot_posterior()`` for triangle-plot markers).
    """
    import arviz as az
    import matplotlib.pyplot as plt
    import tabulate as tb

    # --- Normalise to list ---
    if isinstance(catalog_extract, CatalogExtract):
        catalog_extract = [catalog_extract]

    outfolder = Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)

    summary_parts: list[str] = []

    for ce in catalog_extract:
        safe = ce.name.replace(" ", "_").replace("/", "_")
        n_chains = ce.n_chains

        # Build ArviZ InferenceData (reused for rank, trace, and summary)
        idata = az.from_dict(posterior={k: np.asarray(v) for k, v in ce.cosmo.items()})

        # --------------------------------------------------------------
        # Output 1: Field projections
        # --------------------------------------------------------------
        if ce.mean_field is not None:
            has_true_ic = ce.true_ic is not None
            n_cols = 4 if has_true_ic else 2
            col_names = ["True IC", "Mean", "Std", "Diff"] if has_true_ic else ["Mean", "Std"]

            fig, axes = plt.subplots(n_chains, n_cols, figsize=(4 * n_cols, 4 * n_chains))
            if n_chains == 1:
                axes = axes[np.newaxis, :]
            if n_cols == 1:
                axes = axes[:, np.newaxis]

            for col_i, col_name in enumerate(col_names):
                axes[0, col_i].set_title(col_name)

            for c in range(n_chains):
                axes[c, 0].set_ylabel(f"Chain {c}")

                if has_true_ic:
                    ce.true_ic.project().plot(ax=axes[c, 0])
                    mean_c = ce.mean_field.replace(array=ce.mean_field.array[c])
                    mean_c.project().plot(ax=axes[c, 1])
                    std_c = ce.std_field.replace(array=ce.std_field.array[c])
                    std_c.project().plot(ax=axes[c, 2])
                    diff_c = ce.true_ic - mean_c
                    diff_c.project().plot(ax=axes[c, 3])
                else:
                    mean_c = ce.mean_field.replace(array=ce.mean_field.array[c])
                    mean_c.project().plot(ax=axes[c, 0])
                    std_c = ce.std_field.replace(array=ce.std_field.array[c])
                    std_c.project().plot(ax=axes[c, 1])

            fig.tight_layout()
            fig.savefig(outfolder / f"field_projections_{safe}.{outformat}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        # --------------------------------------------------------------
        # Output 2: Power spectra
        # --------------------------------------------------------------
        if ce.power_spectra is not None:
            mean_tf, std_tf, mean_coh, std_coh = ce.power_spectra
            k = np.asarray(mean_tf.wavenumber)

            fig, axes = plt.subplots(n_chains, 2, figsize=(10, 4 * n_chains))
            if n_chains == 1:
                axes = axes[np.newaxis, :]

            for c in range(n_chains):
                ax_tf = axes[c, 0]
                m = np.asarray(mean_tf.array[c])
                s = np.asarray(std_tf.array[c])
                ax_tf.plot(k, m, label="mean")
                ax_tf.fill_between(k, m - s, m + s, alpha=0.3, label="±1σ")
                ax_tf.set_xscale("log")
                ax_tf.set_yscale("log")
                ax_tf.set_title(f"Transfer — Chain {c}")
                ax_tf.set_xlabel("k")
                ax_tf.legend(fontsize=8)

                ax_coh = axes[c, 1]
                m_c = np.asarray(mean_coh.array[c])
                s_c = np.asarray(std_coh.array[c])
                ax_coh.plot(k, m_c, label="mean")
                ax_coh.fill_between(k, m_c - s_c, m_c + s_c, alpha=0.3, label="±1σ")
                ax_coh.set_xscale("log")
                ax_coh.set_title(f"Coherence — Chain {c}")
                ax_coh.set_xlabel("k")
                ax_coh.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(outfolder / f"power_spectra_{safe}.{outformat}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        # --------------------------------------------------------------
        # Output 3: ArviZ rank plot
        # --------------------------------------------------------------
        az.plot_rank(idata, var_names=ce.cosmo_keys)
        plt.suptitle(f"Rank Plots — {safe}", y=1.02)
        plt.savefig(outfolder / f"rank_plot_{safe}.{outformat}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # --------------------------------------------------------------
        # Output 4: ArviZ trace plot
        # --------------------------------------------------------------
        az.plot_trace(idata, var_names=ce.cosmo_keys)
        plt.suptitle(f"Chain Traces — {safe}", y=1.02)
        plt.savefig(outfolder / f"trace_plot_{safe}.{outformat}", dpi=dpi, bbox_inches="tight")
        plt.close()

        # --------------------------------------------------------------
        # Summary section for this model
        # --------------------------------------------------------------
        summary_df = az.summary(idata, var_names=ce.cosmo_keys)[["mean", "sd", "ess_bulk", "r_hat"]]
        agg_table = tb.tabulate(summary_df, headers="keys", tablefmt="github", floatfmt=".4f")

        ess_rows = []
        for p in ce.cosmo_keys:
            row = [p]
            for c in range(n_chains):
                chain_idata = az.from_dict(posterior={k: np.asarray(v)[c : c + 1] for k, v in ce.cosmo.items()})
                chain_ess = az.ess(chain_idata, method="bulk")
                row.append(f"{float(np.asarray(chain_ess[p]).flat[0]):.1f}")
            ess_rows.append(row)

        ess_table = tb.tabulate(
            ess_rows,
            headers=["param"] + [f"chain_{c}" for c in range(n_chains)],
            tablefmt="github",
        )

        summary_parts.append(
            f"## {safe}\n\n### Aggregate Statistics\n\n{agg_table}\n\n### Per-Chain ESS (bulk)\n\n{ess_table}\n"
        )

    # --- Write combined summary.md ---
    summary_path = outfolder / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# MCMC Summary\n\n")
        for part in summary_parts:
            f.write(part)
            f.write("\n")
