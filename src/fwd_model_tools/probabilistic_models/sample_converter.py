"""Utilities for converting MCMC sample dicts to Catalog format."""

from __future__ import annotations

import os

import jax.numpy as jnp

from .._src.base import ConvergenceUnit
from ..fields import DensityField, DensityUnit, FieldStatus
from ..fields.lensing_maps import FlatKappaField, SphericalKappaField
from ..io import Catalog, save_sharded
from .config import Configurations


__all__ = ["sample2catalog"]


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
    cb : Callable[[dict, str, int], None]
        Callback with signature ``(samples, path, batch_id)``.
    """

    n_bins = len(config.nz_shear)
    is_spherical = config.geometry == "spherical"
    KappaFieldCls = SphericalKappaField if is_spherical else FlatKappaField

    def cb(samples, path, batch_id):
        """Save orbax checkpoint and parquet Catalogs for one batch.

        Parameters
        ----------
        folder : str
            Folder containing the samples.
        path : str
            Full orbax checkpoint path (e.g. ``"output/run/samples_0"``).
        batch_id : int
            Integer batch index.
        """

        # Save the samples
        ic_dir = os.path.join(path, "samples")
        os.makedirs(ic_dir, exist_ok=True)

        cosmo = samples["cosmo"]
        initial_conditions = samples["initial_conditions"]
        intitial_condition_meta_data = samples["initial_conditions_meta_data"]
        if not isinstance(initial_conditions, DensityField):
            initial_conditions = DensityField.FromDensityMetadata(
                array=initial_conditions,
                field=intitial_condition_meta_data,)

        
        initial_conditions = initial_conditions.replace(z_sources=jnp.zeros(initial_conditions.shape[0]) , comoving_centers=jnp.zeros(initial_conditions.shape[0]),scale_factors=jnp.zeros(initial_conditions.shape[0]), density_width=jnp.zeros(initial_conditions.shape[0]))

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
            field=kappa_meta_data,)
        
        kappa_catalog = Catalog(field=kappa_field, cosmology=cosmo)
        kappa_catalog.to_parquet(os.path.join(fields_dir, f"fields_{batch_id}.parquet"))

        if "lightcone" in samples:
            lightcone_dir = os.path.join(path, "lightcones")
            os.makedirs(lightcone_dir, exist_ok=True)
            lightcone = samples["lightcone"]
            lightcone_catalog = Catalog(field=lightcone, cosmology=cosmo)
            lightcone_catalog.to_parquet(os.path.join(lightcone_dir, f"lightcone_{batch_id}.parquet"))

    return cb
