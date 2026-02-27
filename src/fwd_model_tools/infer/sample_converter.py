"""Utilities for converting MCMC sample dicts to Catalog format."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ..fields import DensityField
from ..fields.lensing_maps import FlatKappaField, SphericalKappaField
from ..io import Catalog

if TYPE_CHECKING:
    from ..probabilistic_models.config import Configurations

__all__ = ["sample2catalog"]


def _append_metrics_row(metrics: dict, batch_id: int, base_path: str) -> None:
    """Append one row to {base_path}/metrics.md, creating headers on first write."""
    md_path = os.path.join(base_path, "metrics.md")
    headers = (
        "| Batch | Avg Steps | Divergences | Mean Accept |",
        "|-------|-----------|-------------|-------------|",
    )
    row = "| {batch} | {steps} | {div} | {acc} |".format(
        batch=batch_id,
        steps=f"{metrics['mean_num_steps']:.1f}" if metrics["mean_num_steps"] is not None else "N/A",
        div=str(metrics["total_divergences"]) if metrics["total_divergences"] is not None else "N/A",
        acc=f"{metrics['mean_accept_prob']:.3f}" if metrics["mean_accept_prob"] is not None else "N/A",
    )
    write_headers = not os.path.exists(md_path)
    with open(md_path, "a") as f:
        if write_headers:
            f.write("\n".join(headers) + "\n")
        f.write(row + "\n")


def default_save(samples, metrics, path, batch_id):
    """Default save callback that just saves the samples as an orbax checkpoint."""
    os.makedirs(path, exist_ok=True)
    base_path = os.path.dirname(path)
    _append_metrics_row(metrics, batch_id, base_path)
    np.savez(os.path.join(path, f"samples_batch_{batch_id}.npz"), **samples)


#
def sample2catalog(config: Configurations):
    """Build a save callback that writes both orbax checkpoints and parquet Catalogs.

    The returned callback is suitable for the ``save_callback`` parameter of
    :func:`~fwd_model_tools.sampling.batched_sampling`.

    Parameters
    ----------
    config : Configurations
        The same configuration object used to build the probabilistic model.

    Returns
    -------
    cb : Callable[[dict, dict, str, int], None]
        Callback with signature ``(samples, metrics, path, batch_id)``.
    """

    is_spherical = config.geometry == "spherical"
    KappaFieldCls = SphericalKappaField if is_spherical else FlatKappaField

    def cb(samples, metrics, path, batch_id):
        """Save orbax checkpoint and parquet Catalogs for one batch.

        Parameters
        ----------
        samples : dict
            Sample dictionary from the MCMC run.
        metrics : dict
            Aggregated sampler diagnostics (mean_num_steps, total_divergences, mean_accept_prob).
        path : str
            Full orbax checkpoint path (e.g. ``"output/run/samples"``).
        batch_id : int
            Integer batch index.
        """
        base_path = os.path.dirname(path)
        _append_metrics_row(metrics, batch_id, base_path)

        cosmo = samples["cosmo"]
        initial_conditions = samples.get("initial_conditions")

        if initial_conditions is None:
            # Power-spectrum model: no IC field, save cosmo params as npz instead
            print("No initial conditions found, saving cosmo parameters to npz.")
            cosmo_dir = os.path.join(path, "samples")
            os.makedirs(cosmo_dir, exist_ok=True)
            cosmo_dict = {
                "Omega_c": cosmo.Omega_c,
                "Omega_b": cosmo.Omega_b,
                "h": cosmo.h,
                "n_s": cosmo.n_s,
                "sigma8": cosmo.sigma8,
                "w0": cosmo.w0,
                "wa": cosmo.wa,
                "Omega_k": cosmo.Omega_k,
                "Omega_nu": cosmo.Omega_nu,
            }
            np.savez(os.path.join(cosmo_dir, f"cosmo_{batch_id}.npz"), **cosmo_dict)
        else:
            # Save the IC samples as a parquet Catalog
            ic_dir = os.path.join(path, "samples")
            os.makedirs(ic_dir, exist_ok=True)

            intitial_condition_meta_data = samples["initial_conditions_meta_data"]
            if not isinstance(initial_conditions, DensityField):
                initial_conditions = DensityField.FromDensityMetadata(
                    array=initial_conditions,
                    field=intitial_condition_meta_data,
                )

            initial_conditions = initial_conditions.replace(
                z_sources=jnp.zeros(initial_conditions.shape[0]),
                comoving_centers=jnp.zeros(initial_conditions.shape[0]),
                scale_factors=jnp.zeros(initial_conditions.shape[0]),
                density_width=jnp.zeros(initial_conditions.shape[0]),
            )

            sample_catalog = Catalog(field=initial_conditions, cosmology=cosmo)
            sample_catalog.to_parquet(os.path.join(ic_dir, f"samples_{batch_id}.parquet"))

        # Check if samples has kappa
        if "kappa_0" not in samples:
            print("No kappa samples found, skipping kappa catalog saving.")
            return
        fields_dir = os.path.join(path, "kappa_fields")
        os.makedirs(fields_dir, exist_ok=True)
        # find out how many kappa bins there are by counting keys
        kappa_keys = [k for k in samples if k.startswith("kappa_")]
        n_bins = len(kappa_keys)
        # Create the kappa fields class
        kappa_meta_data = samples["kappa_meta_data"]
        kappa_array = jnp.stack([samples[f"kappa_{i}"] for i in range(n_bins)], axis=0)
        kappa_field = KappaFieldCls.FromDensityMetadata(
            array=kappa_array,
            field=kappa_meta_data,
        )

        kappa_catalog = Catalog(field=kappa_field, cosmology=cosmo)
        kappa_catalog.to_parquet(os.path.join(fields_dir, f"fields_{batch_id}.parquet"))

        if "lightcone" in samples:
            lightcone_dir = os.path.join(path, "lightcones")
            os.makedirs(lightcone_dir, exist_ok=True)
            lightcone = samples["lightcone"]
            lightcone_catalog = Catalog(field=lightcone, cosmology=cosmo)
            lightcone_catalog.to_parquet(os.path.join(lightcone_dir, f"lightcone_{batch_id}.parquet"))

    return cb
