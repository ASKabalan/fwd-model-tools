"""
fwd_model_tools: Forward-modeling and sampling on top of JAXPM + JAX-Decomp.

This package exposes a curated top-level API organized around:
    - fields and initial conditions
    - particle-mesh (PM) evolution
    - lensing
    - probabilistic models
    - power-spectrum utilities
    - sampling and persistency
    - lightweight plotting helpers
"""

from fwd_model_tools.catalog import from_catalog, to_catalog
from fwd_model_tools.fields import (DensityField, DensityStatus, FieldStatus,
                                    FlatDensity, FlatKappaField,
                                    FlatShearField, ParticleField,
                                    SphericalDensity, SphericalKappaField,
                                    SphericalShearField, stack)
from fwd_model_tools.initial import (gaussian_initial_conditions,
                                     interpolate_initial_conditions)
from fwd_model_tools.lensing import born, raytrace
from fwd_model_tools.plotting import plot_gradient_analysis, plot_ic
from fwd_model_tools.pm import lpt, nbody
from fwd_model_tools.power import (PowerSpectrum, angular_cl_flat,
                                   angular_cl_spherical, compute_theory_cl,
                                   power)
from fwd_model_tools.probabilistic_models import (Configurations, Planck18,
                                                  full_field_probmodel,
                                                  make_2pt_model,
                                                  make_full_field_model,
                                                  pixel_window_function,
                                                  powerspec_probmodel)
from fwd_model_tools.sampling import (DistributedNormal, batched_sampling,
                                      load_samples)
from fwd_model_tools.sampling.plot import plot_posterior
from fwd_model_tools.utils import (compute_box_size_from_redshift,
                                   compute_lightcone_shells,
                                   compute_lpt_lightcone_scale_factors,
                                   compute_max_redshift_from_box_size,
                                   compute_snapshot_scale_factors,
                                   reconstruct_full_sphere)

__version__ = "0.1.0"

__all__ = [
    # Fields
    "DensityField",
    "ParticleField",
    "FlatDensity",
    "SphericalDensity",
    "FlatKappaField",
    "SphericalKappaField",
    "FlatShearField",
    "SphericalShearField",
    "FieldStatus",
    "DensityStatus",
    "stack",
    # Initial conditions
    "gaussian_initial_conditions",
    "interpolate_initial_conditions",
    # PM
    "lpt",
    "nbody",
    # Lensing
    "born",
    "raytrace",
    # Probabilistic models
    "Configurations",
    "Planck18",
    "make_full_field_model",
    "full_field_probmodel",
    "reconstruct_full_sphere",
    "powerspec_probmodel",
    "make_2pt_model",
    "pixel_window_function",
    # Power
    "PowerSpectrum",
    "power",
    "angular_cl_flat",
    "angular_cl_spherical",
    "compute_theory_cl",
    # Utilities
    "compute_box_size_from_redshift",
    "compute_lightcone_shells",
    "compute_max_redshift_from_box_size",
    "compute_snapshot_scale_factors",
    "compute_lpt_lightcone_scale_factors",
    # Sampling
    "DistributedNormal",
    "batched_sampling",
    "load_samples",
    # Plotting
    "plot_posterior",
    "plot_ic",
    "plot_gradient_analysis",
    # Catalog I/O
    "to_catalog",
    "from_catalog",
]
