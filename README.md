# fwd-model-tools

**Forward-modeling and sampling on top of JAXPM + JAX-Decomp (no painting)**

`fwd-model-tools` provides clean, JAX-native primitives for sampling initial conditions and running JAXPM N-body simulations, with optional BlackJAX/NumPyro wrappers for Bayesian inference.

**Scope**: IC priors → JAXPM evolution → user-supplied summaries/likelihoods; optional sampling

## Key Features

- **Pure JAX Implementation**: All functions use explicit PRNGKey, static shapes, and optional sharding
- **Distributed-Ready**: Supports single-device and multi-GPU/multi-host execution via jaxDecomp
- **Flexible Field Generation**: Gaussian and lognormal initial conditions with distributed sampling
- **Memory-Efficient Integration**: Custom reverse-mode adjoint for ODE integration with checkpointing
- **Sampling Backend Flexibility**: BlackJAX (HMC/NUTS/MCLMC) and NumPyro support with warmup state persistence
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
mesh_shape = (128, 128, 128)
box_size = [500.0, 500.0, 500.0]  # Mpc/h

# Define power spectrum
cosmo = Planck18()
k = jnp.logspace(-4, 1, 128)
import jax_cosmo as jc
pk_fn = lambda x: jc.power.linear_matter_power(cosmo, x)

# Sample Gaussian initial conditions
ic_dist = DistributedNormal(jnp.zeros(mesh_shape), jnp.ones(mesh_shape))
white_noise = ic_dist.sample(key)

# Generate linear field
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.kernels import fftk
lin_field = linear_field(mesh_shape, box_size, pk_fn, white_noise, fft3d, ifft3d, fftk)
```

### 2. Run Forward Model

```python
from fwd_model_tools import make_full_field_model, Configurations
import numpyro.distributions as dist

# Configure simulation
config = Configurations(
    field_size=5.0,        # degrees
    field_npix=64,
    box_shape=(128, 128, 128),
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
    config.box_shape,
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
from fwd_model_tools.sampling import batched_sampling

# Create probabilistic model
model = full_field_probmodel(config)

# Run MCMC sampling
batched_sampling(
    model,
    path="output/mcmc_run",
    rng_key=jax.random.PRNGKey(0),
    num_warmup=500,
    num_samples=1000,
    batch_count=5,
    sampler="NUTS",       # or "HMC", "MCLMC"
    backend="blackjax",   # or "numpyro"
)

# Load samples
from fwd_model_tools.sampling import load_samples
samples = load_samples("output/mcmc_run")
```

## Package Structure

```
fwd_model_tools/
├── __init__.py              # Main exports: Configurations, Planck18, make_full_field_model
├── config.py                # Configuration dataclass
├── lensing_model.py         # Forward model and probabilistic model
├── fields/                  # Initial condition generators
│   ├── gaussian.py          # Gaussian fields + DistributedNormal
│   └── lognormal.py         # Lognormal fields + DistributedLogNormal
├── likelihood/              # Likelihood functions
│   ├── gaussian.py          # Gaussian likelihoods
│   └── poisson.py           # Poisson likelihoods
├── sampling/                # MCMC sampling utilities
│   └── sampling.py          # Batched sampling with BlackJAX/NumPyro
├── distributed/             # Distributed computing utilities
│   └── rng.py               # Sharding helpers, array save/load
└── solvers/                 # ODE integration
    ├── integrate.py         # Custom reverse-mode adjoint integration
    └── semi_implicit_euler.py  # Semi-implicit Euler solver
```

## Dependencies

All dependencies are installed by default:
- `jax>=0.4.35`
- `equinox>=0.11.0`
- `blackjax>=1.0.0`
- `numpyro>=0.13.0`
- `jaxtyping>=0.2.0`
- `jaxpm>=0.1.0`
- `jax-cosmo>=0.1.0`
- `diffrax>=0.4.0`

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

## Examples

See `examples/` directory for:
- `01-gaussian-ics.py`: Generating Gaussian initial conditions
- `02-lognormal-ics.py`: Generating lognormal initial conditions
- `03-forward-model.py`: Running JAXPM forward model
- `04-hmc-inference.py`: Bayesian inference with HMC/NUTS

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

## Support

- **Issues**: https://github.com/DifferentiableUniverseInitiative/fwd-model-tools/issues
- **Documentation**: https://fwd-model-tools.readthedocs.io (coming soon)
- **JAXPM Documentation**: https://github.com/DifferentiableUniverseInitiative/JaxPM
