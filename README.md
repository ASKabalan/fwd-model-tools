# fwd-model-tools

**Differentiable cosmological forward modeling on JAX**

<p align="center">
  <img src="assets/lightcone_shells.png" alt="HEALPix spherical lightcone shells from an N-body simulation" width="700">
</p>

## Overview

`fwd-model-tools` is a JAX toolkit for end-to-end differentiable cosmological simulations. It chains initial conditions, Lagrangian Perturbation Theory, Particle-Mesh N-body integration, lightcone painting (3D, flat-sky, HEALPix), gravitational lensing (Born and ray-tracing), and angular power spectrum estimation into a single differentiable pipeline. The library supports multi-GPU distribution via JAX sharding, reversible solvers for memory-efficient backpropagation, and probabilistic inference with BlackJAX/NumPyro.

```
ICs ──> LPT ──> PM N-body ──> Lightcone Painting ──> Lensing ──> Power Spectra
                   │                  │
             symplectic KDK/KKD    3D / flat-sky / HEALPix
             + PGD correction      + interpolation kernels
```

## Key Features

- **N-body solvers** -- Symplectic DKD (`EfficientDriftDoubleKick`) and reversible KKD (`ReversibleDoubleKickDrift`) integrators
- **Painting targets** -- 3D density, flat-sky 2D projection, and HEALPix spherical maps with CIC / bilinear / NGP / RBF schemes
- **Interpolation kernels** -- `DriftInterp`, `OnionTiler`, and `TelephotoInterp` for on-the-fly lightcone construction beyond the box boundary
- **Correction kernels** -- PGD (position-based) and Sharpening (velocity-based, reversible) for sub-grid halo correction
- **Gravitational lensing** -- Born approximation (fully JIT-able) and ray-tracing (via Dorian) convergence maps
- **Multi-GPU** -- Distributed simulations via JAX sharding with automatic halo exchange
- **Immutable Field PyTrees** -- `DensityField`, `ParticleField`, `FlatDensity`, `SphericalDensity` carrying arrays + metadata through the pipeline
- **Power spectra** -- 3D P(k), angular C_ell, transfer functions, and theory predictions with Halofit
- **Probabilistic inference** -- Deterministic forward model builder + NumPyro wrappers + BlackJAX batched sampling
- **I/O** -- Orbax checkpointing, Parquet serialization, HuggingFace Dataset integration, CosmoGrid/GowerStreet loaders

## Installation

```bash
pip install -e ".[all]"
```

This installs all optional dependencies (lensing, ray-tracing, catalogs, sampling). For specific extras:

```bash
pip install -e ".[dev]"        # Development tools (pytest, ruff, pre-commit)
pip install -e ".[raytrace]"   # Ray-tracing via Dorian
pip install -e ".[catalog]"    # Parquet / HuggingFace catalog support
```

> **Note:** This package depends on custom forks of `jaxpm` and `jax_cosmo` that are git-pinned in `pyproject.toml`.

## Quick Start

```python
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import fwd_model_tools as ffi

key = jax.random.PRNGKey(42)
cosmo = jc.Planck18()

# 1. Gaussian initial conditions
initial_field = ffi.gaussian_initial_conditions(
    key, mesh_size=(256, 256, 256), box_size=(1000.0, 1000.0, 1000.0),
    cosmo=cosmo, nside=256,
)

# 2. LPT displacement + momentum
dx, p = ffi.lpt(cosmo, initial_field, ts=0.1, order=1)

# 3. PM N-body with spherical lightcone output
solver = ffi.ReversibleDoubleKickDrift(
    interp_kernel=ffi.NoInterp(painting=ffi.PaintingOptions(target="spherical")),
)
lightcone = ffi.nbody(cosmo, dx, p, t1=1.0, dt0=0.05, nb_shells=4, solver=solver)

# 4. Born lensing convergence
nz = [ffi.tophat_z(0.0, 0.5, gals_per_arcmin2=1.0)]
kappa = ffi.born(cosmo, lightcone, nz_shear=nz)

# 5. Angular power spectrum
cl = kappa.angular_cl(method="healpy")
```

<p align="center">
  <img src="assets/kappa_born.png" alt="Born approximation convergence kappa maps" width="700">
</p>

## Tutorials

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Basics](notebooks/01-basics.ipynb) | Core objects, field types, painting targets, power spectra, and I/O |
| 02 | [LPT Simulation](notebooks/02-LPT-Simulation.ipynb) | Full lightcone using LPT with arrays of scale factors and spherical shells |
| 03 | [PM Simulation](notebooks/03-PM-Simulation.ipynb) | PM N-body solvers, correction kernels, and painting targets |
| 04 | [Distributed PM](notebooks/04-Distributed-PM.ipynb) | Multi-GPU distributed PM using JAX sharding and device meshes |
| 05 | [PM Interpolation](notebooks/05-PM-Interpolation.ipynb) | Lightcone painting with TelephotoInterp and OnionTiler kernels |
| 06 | [Advanced PM](notebooks/06-Advanced-PM.ipynb) | Production pipeline combining PGD correction, OnionTiler, and theory validation |
| 07 | [Lensing](notebooks/07-Lensing.ipynb) | Born approximation and ray-tracing convergence maps |
| 08 | [External Catalogs](notebooks/08-External-Catalog.ipynb) | Loading CosmoGrid and GowerStreet simulation data |
| 09 | [Comparison](notebooks/09-Comparison.ipynb) | Validation against CosmoGrid through power spectra and lensing maps |
| 10 | [Probabilistic Modeling](notebooks/10-Probabilistic-Modeling.ipynb) | Custom MCMC distributions and reparameterization transforms |
| 11 | [Full-Field Inference](notebooks/11-FullFieldInference.ipynb) | Bayesian inference of initial conditions and cosmological parameters |

## Module Map

| Module | Purpose |
|--------|---------|
| `fields/` | Immutable PyTree containers (`DensityField`, `ParticleField`, `FlatDensity`, `SphericalDensity`) |
| `initial.py` | Gaussian initial conditions and interpolation to mesh |
| `pm/` | Particle-mesh engine: `lpt()`, `nbody()`, symplectic solvers, PGD correction, integration loop |
| `lensing/` | Born approximation and ray-tracing convergence maps |
| `power/` | Power spectrum estimation (P(k), C_ell, transfer, coherence), theory predictions |
| `probabilistic_models/` | Deterministic forward model builder, NumPyro wrappers, `Configurations` dataclass |
| `sampling/` | BlackJAX batched sampling, `DistributedNormal`, chain plotting |
| `io/` | Checkpoint persistence, HuggingFace catalog, CosmoGrid/GowerStreet loaders |
| `utils.py` | Lightcone geometry helpers, comoving distances, scale factors |
| `parameters.py` | Predefined cosmologies (e.g. `Planck18`) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Format + import sort (pre-commit uses yapf + isort)
pre-commit run --all-files
```

## License

MIT
