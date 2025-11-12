# fwd-model-tools

**Forward-modeling and sampling on top of JAXPM + JAX-Decomp (no painting)**

`fwd-model-tools` provides clean, JAX-native primitives for sampling initial conditions and running JAXPM N-body simulations, with optional BlackJAX/NumPyro wrappers for Bayesian inference.

**Scope**: IC priors → JAXPM evolution → user-supplied summaries/likelihoods; optional sampling

## Key Features

- **Pure JAX Implementation**: All functions use explicit PRNGKey, static shapes, and optional sharding
- **Distributed-Ready**: Supports single-device and multi-GPU/multi-host execution via jaxDecomp
- **Dual Geometry Support**: Both spherical (HEALPix) and flat (Cartesian) geometries with proper visibility masking
- **Flexible Field Generation**: Gaussian and lognormal initial conditions with distributed sampling
- **Memory-Efficient Integration**: Custom reverse-mode adjoint for ODE integration with checkpointing
- **Sampling Backend Flexibility**: BlackJAX (HMC/NUTS/MCLMC) and NumPyro (NUTS/HMC) support with warmup state persistence
- **Batched MCMC**: Resumable sampling with automatic checkpoint management
- **No Painting Operations**: Focuses on positions, velocities, and displacements (painting is delegated to JAXPM)

## Installation

```bash
pip install fwd-model-tools
```

For development:
```bash
pip install fwd-model-tools[dev]
```

## Quick Start

### 1. Generate Initial Conditions

```python
import jax
import jax.numpy as jnp
from fwd_model_tools.fields import DistributedNormal, linear_field
from fwd_model_tools import Planck18

# Setup
key = jax.random.PRNGKey(42)
mesh_size = (128, 128, 128)
box_size = [500.0, 500.0, 500.0]  # Mpc/h

# Define power spectrum
cosmo = Planck18()
k = jnp.logspace(-4, 1, 128)
import jax_cosmo as jc
pk_fn = lambda x: jc.power.linear_matter_power(cosmo, x)

# Sample Gaussian initial conditions
ic_dist = DistributedNormal(jnp.zeros(mesh_size), jnp.ones(mesh_size))
white_noise = ic_dist.sample(key)

# Generate linear field
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.kernels import fftk
lin_field = linear_field(mesh_size, box_size, pk_fn, white_noise, fft3d, ifft3d, fftk)
```

### 2. Run Forward Model

```python
from fwd_model_tools import make_full_field_model, Configurations
import numpyro.distributions as dist

# Configure simulation
config = Configurations(
    field_size=5.0,        # degrees
    field_npix=64,
    box_size=(128, 128, 128),
    box_size=[500.0, 500.0, 500.0],
    density_plane_width=50,
    density_plane_npix=256,
    nside=512,
    density_plane_smoothing=0.1,
    nz_shear=[...],        # redshift distributions
    fiducial_cosmology=Planck18,
    sigma_e=0.3,
    priors={"Omega_c": dist.Uniform(0.2, 0.35)},
    t0=0.1, dt0=0.05, t1=1.0,
)

# Create forward model
forward_model = make_full_field_model(
    config.field_size,
    config.field_npix,
    config.box_size,
    config.box_size,
    config.density_plane_width,
    config.density_plane_npix,
    nside=config.nside,
    t0=config.t0,
    t1=config.t1,
    dt0=config.dt0,
    geometry="spherical"  # or "flat"
)

# Run forward model
convergence_maps, lightcone, ic = forward_model(cosmo, config.nz_shear, white_noise)
```

### 3. Bayesian Inference with MCMC

```python
from fwd_model_tools import full_field_probmodel
from fwd_model_tools.sampling import batched_sampling, load_samples

# Create probabilistic model
model = full_field_probmodel(config)

# Run MCMC sampling (BlackJAX recommended for distributed/sharded workflows)
batched_sampling(
    model,
    path="output/mcmc_run",
    rng_key=jax.random.PRNGKey(0),
    num_warmup=500,
    num_samples=200,     # samples per batch
    batch_count=5,        # total: 1000 samples
    sampler="NUTS",       # "HMC", "MCLMC" (BlackJAX only)
    backend="blackjax",   # "numpyro" (see Important Notes below)
    save=True,            # enables checkpoint resumption
)

# Load and analyze samples
samples = load_samples("output/mcmc_run")
print(f"Omega_c: {samples['Omega_c'].mean():.4f} ± {samples['Omega_c'].std():.4f}")

# Plot posteriors
from fwd_model_tools.plotting import plot_posterior
plot_posterior(samples, "output/plots", params=("Omega_c", "sigma8"))
```
## Design Principles

### 1. Pure JAX Functions
All core functions are pure JAX transformations compatible with `jit`, `grad`, and `vmap`:
- Explicit `PRNGKey` arguments
- Static shapes (no dynamic control flow dependent on traced values)
- Optional sharding passthrough for distributed operations

### 2. No Painting by Design
Painting (particle-to-mesh interpolation) is intentionally out-of-scope. Users should:
- Use JAXPM's `cic_paint`/`cic_read` directly if needed
- Focus on particle positions/velocities for summaries
- Keep forward models independent of mass assignment details

### 3. Distributed Computing
Supports single-device and distributed runs with deterministic per-device RNG:
- Use `sharding` parameter consistently across JAXPM calls
- Use `DistributedNormal`/`DistributedLogNormal` for initial conditions
- See JAXPM documentation for multi-GPU/multi-host setup

### 4. Whitened Reparametrization
Initial conditions are sampled in whitened space (unit Gaussian):
- Better HMC/NUTS performance (no mass matrix tuning needed)
- Power spectrum applied in forward model (not in prior)
- Supports both Gaussian and lognormal transformations

## Example Workflows

### Command-Line Scripts

**Complete lensing inference workflow:**
```bash
python scripts/run_lensing_model.py \
  --box-shape 128 128 128 \
  --geometry spherical \
  --observer-position 0.5 0.5 0.0 \
  --max-redshift 1.0 \
  --num-warmup 500 \
  --num-samples 200 \
  --batch-count 10 \
  --sampler NUTS \
  --backend blackjax \
  --output-dir output/lensing_run
```

**Simple field-based inference (pedagogical example):**
```bash
python scripts/run_simple_sampling.py \
  --num-warmup 100 \
  --num-samples 50 \
  --batch-count 5 \
  --sampler NUTS \
  --backend blackjax \
  --output-dir output/simple_run
```

**Plot-only mode (re-plot without re-running inference):**
```bash
python scripts/run_lensing_model.py --plot-only --n-samples-plot -1
```

### Interactive Jupyter Notebook

See `notebooks/lensing_inference_workflow.ipynb` for an interactive workflow including:
- Gradient analysis (parameter sensitivity)
- Synthetic observation generation
- Resumable MCMC sampling
- Posterior visualization and diagnostics

## Alignment with JAXPM and JAX-Decomp

`fwd-model-tools` is designed to work seamlessly with:
- **JAXPM**: Uses JAXPM's `lpt()`, `pm_forces()`, and lensing functions
- **JAX-Decomp**: Compatible with jaxDecomp's distributed FFTs and sharding
- **JAX-Cosmo**: Uses JAX-Cosmo for cosmology calculations and power spectra

## License

MIT License - see LICENSE file for details

## Citation

If you use `fwd-model-tools` in your research, please cite:

```bibtex
@software{fwd_model_tools,
  title = {fwd-model-tools: Forward-modeling and sampling on top of JAXPM},
  author = {Forward Model Tools Contributors},
  year = {2024},
  url = {https://github.com/DifferentiableUniverseInitiative/fwd-model-tools}
}
```

Also cite JAXPM:
```bibtex
@software{jaxpm,
  title = {JaxPM: A Particle-Mesh N-body solver in JAX},
  author = {JaxPM developers},
  year = {2024},
  url = {https://github.com/DifferentiableUniverseInitiative/JaxPM}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests with `pytest`
4. Submit a pull request

## Important Notes

### ⚠️ NumPyro Sharding Limitation

**NumPyro does NOT support distributed sharding with checkpoint resumption.** When resuming from a saved state, sharding information is lost, causing all data to collapse to a single device (`SingleDeviceSharding`) or replicated sharding (`GSPMDSharding({replicated})`). This severely impacts performance and memory usage in distributed settings.
Please avoid using NumPyro backend for distributed/sharded workflows that require checkpoint resumption.

**For distributed/sharded workflows, use BlackJAX backend.** BlackJAX correctly preserves sharding across checkpoint resumption.

Example:
```python
# ✓ RECOMMENDED for distributed/sharded runs
batched_sampling(..., backend="blackjax", sampler="NUTS", sharding=sharding)

# ✗ NOT RECOMMENDED: NumPyro loses sharding on resumption
batched_sampling(..., backend="numpyro", sampler="NUTS", sharding=sharding)
```

### Spherical Geometry

When using `geometry="spherical"`:
- The forward model returns convergence maps for **visible pixels only** (determined by observer position and HEALPix visibility mask)
- Use `reconstruct_full_kappa()` to convert visible-pixel maps to full HEALPix maps for plotting
- Likelihood automatically handles visible-pixel sampling with correct HEALPix pixel area scaling

### Batched Sampling

The `batched_sampling` function:
- Saves warmup state to `{path}/sampling_state.pkl` for resumption
- Saves samples to `{path}/samples_{i}.npz` (one file per batch)
- Can be interrupted and resumed by re-running with increased `batch_count`
- Use `load_samples(path)` to efficiently load and concatenate all batches

## Troubleshooting

**CUDA plugin errors on CPU-only runs:**
```bash
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
python scripts/run_lensing_model.py ...
```

**Out of memory with large box sizes:**
- Use `batched_sampling` with smaller `num_samples` and more `batch_count`
- Enable distributed sharding (requires multiple devices)
- Reduce `box_size` or `density_plane_npix`

**Posterior plots look odd (too narrow/spiky):**
- For small sample counts (n < 200), try `plot_posterior(..., pair_kind="scatter")`
- Check that likelihood noise levels are reasonable for your setup
- Verify forward model gradients with gradient analysis (see notebook)

## Support

- **Issues**: https://github.com/DifferentiableUniverseInitiative/fwd-model-tools/issues
- **JAXPM Documentation**: https://github.com/DifferentiableUniverseInitiative/JaxPM
