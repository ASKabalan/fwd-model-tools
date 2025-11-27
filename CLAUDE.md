# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fwd-model-tools` is a JAX-based package for forward-modeling and Bayesian inference in cosmological simulations. It provides primitives for sampling initial conditions, running N-body simulations via JAXPM, and performing MCMC sampling for parameter inference.

**Project Structure:**
- `src/fwd_model_tools/` - Main package code
  - `config.py` - Configuration dataclass (Configurations NamedTuple)
  - `lensing_model.py` - Forward model and probabilistic model (make_full_field_model, full_field_probmodel, Planck18)
  - `field.py` - Field container PyTrees (DensityField, ParticleField, FlatDensity, SphericalDensity)
  - `normal.py` - Gaussian initial conditions (gaussian_initial_conditions, interpolate_initial_conditions)
  - `pm.py` - Particle mesh operations (lpt wrapper for JAXPM)
  - `sampling.py` - MCMC sampling (batched_sampling, load_samples, DistributedNormal)
  - `distributed.py` - Distributed computing utilities (save_sharded, load_sharded)
  - `plotting.py` - Plotting utilities (plot_kappa, plot_lightcone, plot_ic, plot_posterior, prepare_arviz_data)
  - `solvers/` - ODE integration (integrate.py, semi_implicit_euler.py)
- `scripts/` - Executable scripts (run_distributed_full_field_inference.py, run_simple_sampling.py, run_full_field_vs_powerspec.py)
- `tests/` - Test suite (test_sampling.py)
- Uses src/ layout (PEP 517/518)

**Key Design Principles:**
- Pure JAX functions compatible with `jit`, `grad`, and `vmap`
- Explicit `PRNGKey` handling and static shapes
- No painting operations (delegated to JAXPM)
- Whitened reparametrization for efficient HMC/NUTS sampling
- Distributed computing via JAX sharding and jaxDecomp

## Development Commands

### Testing
```bash
pytest
pytest tests/test_specific.py  # run single test file
pytest -k test_function_name   # run specific test
```

### Code Quality
```bash
ruff check .           # lint code
ruff check . --fix     # auto-fix linting issues
ruff format .          # format code
pre-commit run --all-files  # run all pre-commit hooks (yapf, isort, etc.)
```

### Installation
```bash
pip install -e .        # full installation (includes all dependencies)
pip install -e .[dev]   # with development tools
```

### Running Example Scripts
```bash
python scripts/run_distributed_full_field_inference.py --output-dir output --geometry spherical
python scripts/run_simple_sampling.py --output-dir output/simple_run
python scripts/run_full_field_vs_powerspec.py --output-dir output/comparison
```

## Architecture

### Core Components

**Configuration Layer (`config.py`):**
- `Configurations` is a `NamedTuple` that encapsulates all simulation parameters
- Includes box geometry, time evolution, cosmology, observation parameters, and computational options
- Central configuration object passed to all high-level functions

**Forward Model (`lensing_model.py`):**
- `make_full_field_model()`: Factory function that returns a JIT-compiled forward model
  - Takes whitened initial conditions → generates linear field → runs N-body simulation → computes lensing convergence
  - Returns function: `(cosmo, nz_shear, initial_conditions) -> (convergence_maps, lightcone, lin_field)`
  - Supports both spherical (HEALPix) and flat (Cartesian) geometries
- `full_field_probmodel()`: Creates a NumPyro probabilistic model for Bayesian inference
  - Samples cosmological parameters from priors
  - Samples whitened initial conditions using `DistributedNormal`
  - Runs forward model and conditions on observations
  - Pre-computes spherical visibility masks outside model to avoid dynamic shapes under JIT
- `Planck18()`: Returns JAX-Cosmo cosmology object with Planck 2018 parameters
- `compute_box_size_from_redshift()`: Compute simulation box size from max redshift and observer position
- `compute_max_redshift_from_box_size()`: Compute max redshift from box size and observer position
  - `reconstruct_full_sphere()`: Reconstruct full HEALPix map from visible pixels only

**Field Containers (`field.py`):**
- `DensityField`: PyTree container for 3D volumetric simulation arrays with metadata
  - Stores array data, mesh/box shape, observer position, sharding, nside, halo_size
  - Tracks field lifecycle status (RAW, INITIAL_FIELD, LPT1, LPT2, DENSITY_FIELD)
  - Supports arithmetic operations on fields (+, -, *, /)
  - Registered as JAX PyTree with proper flatten/unflatten
- `ParticleField`: Subclass of DensityField for particle positions/displacements
  - Array shape must be (X, Y, Z, 3) for 3D particle data
  - Provides `paint()` method for CIC painting (relative/absolute modes)
  - Provides `read_out()` for interpolation from density mesh
  - Provides `paint_2d()` for flat-sky projections
  - Provides `paint_spherical()` for HEALPix projections
- `FlatDensity`: 2D flat-sky density/shear maps with plotting support
- `SphericalDensity`: HEALPix density/shear maps with mollview plotting
- All field classes support stacking and visualization

**Initial Conditions (`normal.py`):**
- `gaussian_initial_conditions()`: Generate Gaussian ICs and package as DensityField
  - Samples whitened field using `jaxpm.distributed.normal_field()`
  - Applies power spectrum via `interpolate_initial_conditions()`
- `interpolate_initial_conditions()`: Apply power spectrum to whitened field
  - Takes whitened field → FFT → multiply by sqrt(P(k)) → IFFT
  - Returns DensityField with INITIAL_FIELD status
  - Supports both cosmology object and explicit power spectrum function

**Particle Mesh Operations (`pm.py`):**
- `lpt()`: Wrapper for JAXPM's LPT implementation
  - Takes DensityField, returns (ParticleField displacements, ParticleField momenta)
  - Supports order=1 (Zel'dovich) and order=2 (2LPT)
  - Automatically propagates metadata (sharding, observer position, etc.)

**MCMC Sampling (`sampling.py`):**
- `DistributedNormal`: Extends NumPyro's Normal distribution for distributed sampling
  - Uses `jaxpm.distributed.normal_field()` for sharded sampling
  - Whitened parameterization (unit Gaussian prior)
  - Compatible with NumPyro's probabilistic modeling
- `batched_sampling()`: Main sampling function with warmup state persistence
  - Supports BlackJAX (NUTS, HMC, MCLMC) and NumPyro (NUTS, HMC) backends
  - Saves warmup state to disk for resumable sampling
  - Batches samples to manage memory
  - BlackJAX backend recommended for distributed/sharded workflows
- `load_samples()`: Efficiently loads and concatenates samples from batch files
  - Supports loading specific parameters, last N batches, and on-the-fly transforms (mean/std)

**Distributed Computing (`distributed.py`):**
- `save_sharded()`: Saves sharded arrays using orbax checkpointing
- `load_sharded()`: Loads sharded arrays from checkpoints
- Enables efficient storage and loading of large distributed arrays without gathering to single device

**Plotting Utilities (`plotting.py`):**
- `plot_kappa()`: Plot convergence maps (supports both spherical HEALPix and flat geometries)
- `plot_lightcone()`: Plot density planes from lightcone
- `plot_ic()`: Compare true vs. posterior mean/std of initial conditions
- `plot_posterior()`: Plot posterior distributions using ArviZ (trace and pair plots)
- `prepare_arviz_data()`: Convert sample dictionaries to ArviZ InferenceData format
- Requires `healpy` for HEALPix visualization and `arviz` for posterior analysis

**ODE Integration (`solvers/`):**
- `integrate.py`: Custom reverse-mode adjoint integrator with memory-efficient checkpointing
  - Implements custom VJP for gradient computation
  - Uses `jax.lax.scan` and `jax.lax.while_loop` for forward and backward passes
  - Saves only at specified snapshot times (not every step)
- `semi_implicit_euler.py`: Semi-implicit Euler solver for symplectic integration
  - Compatible with JAXPM's `symplectic_ode`

### Data Flow

1. **Prior Sampling**: `DistributedNormal` samples whitened initial conditions (3D array)
2. **Linear Field**: `interpolate_initial_conditions()` applies power spectrum → DensityField
3. **LPT**: `lpt()` computes displacements and momenta → ParticleField(dx), ParticleField(p)
4. **N-body Evolution**: `symplectic_ode()` evolves particles forward in time
5. **Density Planes**: `density_plane_fn()` or `spherical_density_fn()` saves lightcone snapshots
6. **Lensing**: `convergence_Born()` computes convergence from density planes
7. **Likelihood**: Gaussian likelihood conditions on observed maps (visible pixels for spherical)

### Key Patterns

**Sharding Consistency:**
All JAXPM functions must receive the same `sharding` parameter. Pass through from `Configurations.sharding`:
```python
forward_model = make_full_field_model(..., sharding=config.sharding)
```

**PRNGKey Management:**
All functions use explicit `PRNGKey` arguments:
```python
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
sample = distribution.sample(subkey)
```

**Cosmology Workspace:**
Clear JAX-Cosmo's internal cache before each use to avoid tracing issues:
```python
cosmo._workspace = {}
```

**Observer Position:**
The `observer_position` parameter in `Configurations` specifies the observer location as a fraction of box size (x, y, z) with values between 0 and 1:
```python
config = Configurations(..., observer_position=(0.5, 0.5, 0.5))  # center of box
config = Configurations(..., observer_position=(0.5, 0.5, 0.0))  # edge observer
```
For spherical geometry, this determines which HEALPix pixels are visible. Use `spherical_visibility_mask()` to get visible pixel indices.

**Field PyTree Pattern:**
All simulation data flows through Field containers (DensityField, ParticleField, etc.) which are JAX PyTrees:
```python
lin_field = interpolate_initial_conditions(white_noise, mesh_size, box_size, pk_fn=pk_fn)
dx_field, p_field = lpt(cosmo, lin_field, a=0.1)
density = dx_field.paint(mode="relative")  # ParticleField → DensityField
```
Fields automatically carry metadata (sharding, observer position, halo_size) through computations.

**Adjoint Selection:**
Two options for gradient computation in ODE integration:
- `RecursiveCheckpointAdjoint(n)`: Diffrax's checkpointing (default)
- Custom adjoint from `solvers/integrate.py`: More memory-efficient for long integrations

## Common Tasks

### Adding a New Likelihood

1. Create likelihood function in `lensing_model.py` or a new module
2. Accept observations and predictions as JAX arrays
3. Return log-likelihood (scalar)
4. Integrate into `full_field_probmodel()` via `numpyro.sample()`

### Adding a New Initial Condition Prior

1. Create a new distribution class in `sampling.py` extending `numpyro.distributions.Distribution`
2. Override `sample()` method to use distributed sampling (e.g., `jaxpm.distributed.normal_field()`)
3. Ensure compatibility with whitened parameterization
4. Use in `full_field_probmodel()` via `numpyro.sample()`

Example:
```python
class DistributedLogNormal(LogNormal):
    def __init__(self, loc=0.0, scale=1.0, sharding=None, *, validate_args=None):
        self.sharding = sharding
        super().__init__(loc, scale, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = normal_field(key, sample_shape + self.batch_shape, self.sharding)
        return jnp.exp(self.loc + eps * self.scale)
```

### Modifying the Forward Model

1. Edit `make_full_field_model()` in `lensing_model.py`
2. Maintain JIT compatibility (no dynamic shapes or control flow)
3. Ensure all JAXPM calls receive consistent `sharding` and `halo_size`
4. Test gradients with `jax.grad()` or `jax.value_and_grad()`

### Debugging Integration Issues

- Use `scan_integrate()` from `solvers/integrate.py` for comparison (stores all steps)
- Check snapshot times are within `[t0, t1]`
- Verify `dt0` is small enough for solver stability
- Inspect gradients with `jax.grad(forward_model)`

## Dependencies and Imports

**All dependencies are installed by default:**
- `jax`, `equinox`
- `jaxpm`, `jax_cosmo`, `diffrax`
- `blackjax`, `numpyro`, `jaxtyping`
- `healpy`, `arviz` (for plotting utilities)

**Import patterns:**
All packages can be imported directly without guards:
```python
import jaxpm
import jax_cosmo as jc
import blackjax
import numpyro
```

## Code Style

- Line length: 120 characters (enforced by ruff)
- Use double quotes for strings
- Type hints encouraged but not required
- Docstrings: NumPy style with Parameters/Returns/Notes sections
- Auto-formatting via `ruff format` (primary) and yapf (pre-commit)
- Import sorting via isort (pre-commit hook)

## Testing Notes

- Tests located in `tests/` directory
- Use `pytest` conventions: `test_*.py`, `Test*` classes, `test_*` functions
- Coverage reporting enabled via `pytest-cov`
- Existing tests:
  - `test_sampling.py`: Tests `load_samples()` functionality with different transform modes
- When writing tests:
  - Mock JAXPM dependencies when testing core logic without cosmology
  - Use small mesh shapes (e.g., 8x8x8) to speed up tests
  - Test Field PyTree flatten/unflatten behavior
  - Verify sharding preservation across operations

## Important Implementation Details

### Semi-Implicit Euler Solver
The `SemiImplicitEuler` solver in `solvers/semi_implicit_euler.py` is specifically designed for symplectic integration and includes a `reverse()` method for adjoint computation. This is critical for gradient-based inference.

### Custom Adjoint Integration
The custom adjoint in `solvers/integrate.py` uses `jax.lax.while_loop` for inner time-stepping and `jax.lax.scan` for snapshot iteration. This reduces memory usage compared to Diffrax's `RecursiveCheckpointAdjoint` by only storing snapshots, not intermediate steps.

### Distributed Array I/O
When working with large distributed arrays, use `distributed.save_sharded()` and `distributed.load_sharded()` to efficiently save/load arrays across multiple devices without gathering to a single device. These functions use orbax checkpointing under the hood.

### Field PyTree System
The Field classes (DensityField, ParticleField, FlatDensity, SphericalDensity) are registered JAX PyTrees that carry both data and metadata. This design ensures:
- Automatic metadata propagation through JAX transformations (jit, grad, vmap)
- Type safety: ParticleField enforces shape (X, Y, Z, 3), painting operations only available on ParticleField
- Lifecycle tracking: FieldStatus enum tracks transformations (RAW → INITIAL_FIELD → LPT1 → DENSITY_FIELD)
- Arithmetic operations preserve metadata: `field1 + field2` returns a new Field with the same metadata

### NumPyro Backend Sharding Limitation
**CRITICAL**: NumPyro does NOT preserve sharding across checkpoint resumption. When resuming from a saved state, sharding information is lost, causing all data to collapse to a single device or replicated sharding. This severely impacts performance and memory in distributed settings.

**Always use BlackJAX backend for distributed/sharded workflows.** BlackJAX correctly preserves sharding across checkpoint resumption.
