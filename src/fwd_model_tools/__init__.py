"""
fwd_model_tools: Forward-modeling and sampling on top of JAXPM + JAX-Decomp.
"""

from . import fields, initial, io, lensing, pm
from . import probabilistic_models as ppl
from . import sampling, utils

# From fields
from .fields import (
    AbstractField,
    DensityField,
    DensityUnit,
    FieldMetadata,
    FieldStatus,
    FlatDensity,
    FlatKappaField,
    FlatShearField,
    PaintingOptions,
    ParticleField,
    PhysicalUnit,
    PositionUnit,
    SphericalDensity,
    SphericalKappaField,
    SphericalShearField,
    convert_units,
    units,
)

# From initial
from .initial import gaussian_initial_conditions, interpolate_initial_conditions

# From lensing
from .lensing import born, raytrace

# From pm
from .pm import (
    AbstractNBodySolver,
    DriftInterp,
    EfficientDriftDoubleKick,
    InterpTilerState,
    NoCorrection,
    NoInterp,
    OnionTiler,
    PGDKernel,
    ReversibleDoubleKickDrift,
    SharpeningKernel,
    TelephotoInterp,
    lpt,
    nbody,
)

# From power
from .power import (
    PowerSpectrum,
    angular_cl_flat,
    angular_cl_spherical,
    coherence,
    compute_theory_cl,
    power,
    tophat_z,
    transfer,
)

# From utils
from .utils import (
    centers,
    compute_box_size_from_redshift,
    compute_lightcone_shells,
    compute_max_redshift_from_box_size,
    compute_particle_scale_factors,
    distances,
    edges,
)

__version__ = "0.1.0"

__all__ = [
    # Submodules
    "fields",
    "initial",
    "io",
    "lensing",
    "pm",
    "ppl",
    "sampling",
    "utils",
    # From initial
    "gaussian_initial_conditions",
    "interpolate_initial_conditions",
    # From fields
    "AbstractField",
    "FieldMetadata",
    "DensityField",
    "DensityUnit",
    "FieldStatus",
    "FlatDensity",
    "FlatKappaField",
    "FlatShearField",
    "PaintingOptions",
    "ParticleField",
    "PhysicalUnit",
    "PositionUnit",
    "SphericalDensity",
    "SphericalKappaField",
    "SphericalShearField",
    "convert_units",
    "units",
    # From pm
    "lpt",
    "nbody",
    "EfficientDriftDoubleKick",
    "ReversibleDoubleKickDrift",
    "AbstractNBodySolver",
    "PGDKernel",
    "SharpeningKernel",
    "NoCorrection",
    "OnionTiler",
    "TelephotoInterp",
    "NoInterp",
    "DriftInterp",
    "InterpTilerState",
    # From power
    "PowerSpectrum",
    "power",
    "transfer",
    "coherence",
    "angular_cl_flat",
    "angular_cl_spherical",
    "compute_theory_cl",
    "tophat_z",
    # From lensing
    "born",
    "raytrace",
    # From utils
    "compute_box_size_from_redshift",
    "compute_max_redshift_from_box_size",
    "compute_lightcone_shells",
    "compute_particle_scale_factors",
    "edges",
    "distances",
    "centers",
]
