"""
fwd_model_tools: Forward-modeling and sampling on top of JAXPM + JAX-Decomp

Tagline: Forward-modeling and sampling on top of JAXPM + JAX-Decomp (no painting).
Scope: IC priors → JAXPM evolution → user-supplied summaries/likelihoods; optional sampling.

This package provides clean, JAX-native primitives to sample whitened ICs and run JAXPM,
with BlackJAX and NumPyro support for HMC/NUTS/MCLMC sampling with whitening reparametrization.

Installation
------------
    pip install fwd-model-tools

For development:
    pip install fwd-model-tools[dev]
"""

from fwd_model_tools.config import Configurations
from fwd_model_tools.lensing_model import (E, Planck18, full_field_probmodel,
                                           linear_field, make_full_field_model)
from fwd_model_tools.plotting import (plot_ic, plot_kappa, plot_lightcone,
                                      plot_posterior, prepare_arviz_data)

__version__ = "0.1.0"

__all__ = [
    "Configurations",
    "E",
    "Planck18",
    "linear_field",
    "make_full_field_model",
    "full_field_probmodel",
    "plot_lightcone",
    "plot_kappa",
    "plot_ic",
    "plot_posterior",
    "prepare_arviz_data",
]
