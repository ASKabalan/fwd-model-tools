# CLAUDE.md — fwd-model-tools

## Build & Development

```bash
# Install (editable, with all extras)
pip install -e ".[all]"

# Install dev tools only
pip install -e ".[dev]"

# Run tests
pytest                    # uses testpaths=["tests"], runs with --cov

# Lint & format
ruff check .              # lint (pycodestyle, pyflakes, pyupgrade, isort, debugger)
ruff format .             # format (double quotes, 120 char line length)

# Pre-commit (yapf + isort + trailing whitespace)
pre-commit run --all-files
```

Note: the pre-commit config uses **yapf** (not ruff) for formatting and **isort** for import sorting. `ruff` is used separately for linting. Line length is 120 everywhere.

## Architecture

### Pipeline flow

```
initial conditions → LPT → N-body (PM) → lightcone painting → lensing (Born/raytrace) → power spectra
```

The probabilistic layer wraps this pipeline into a differentiable forward model sampled with BlackJAX/NumPyro.

### Module map

| Module | Purpose |
|---|---|
| `fields/` | Immutable PyTree containers (`DensityField`, `ParticleField`, lightcone maps). All inherit `AbstractField → AbstractPytree → eqx.Module`. |
| `_src/base/` | Core base classes (`AbstractPytree`, `AbstractField`), enums (`FieldStatus`, units), `tri_map`. |
| `initial.py` | Gaussian initial conditions and interpolation to mesh. |
| `pm/` | Particle-mesh engine: `lpt()`, `nbody()`, symplectic solvers, PGD correction, interpolation/tiling, integration loop. |
| `power/` | Power spectrum estimation (`power`, `transfer`, `coherence`, `angular_cl_*`), theory predictions. |
| `lensing/` | Born approximation and ray-tracing convergence maps. |
| `probabilistic_models/` | Deterministic forward model builder (`make_full_field_model`), NumPyro probabilistic wrappers, `Configurations` dataclass. |
| `sampling/` | BlackJAX batched sampling, distributed priors (`DistributedNormal`), chain plotting. |
| `io/` | Checkpoint persistence (`save_sharded`/`load_sharded`), HuggingFace catalog, CosmoGrid/GowerStreet loaders. |
| `utils.py` | Lightcone geometry helpers (shell computation, comoving distances, scale factors). |
| `parameters.py` | Predefined cosmologies (e.g. `Planck18`). |
| `units.py` | Unit conversion logic for density fields. |

### Key abstractions

- **Field PyTrees**: `AbstractPytree` → `AbstractField` → `DensityField` / `ParticleField` / lightcone types. Fields carry both the JAX array (`.array`) and static metadata (`mesh_size`, `box_size`, `sharding`, `status`, `unit`). Immutable — use `.replace(**kwargs)` to produce modified copies.
- **N-body solvers**: `AbstractNBodySolver` (equinox Module) with `init/step/save_at/reverse`. Two implementations: `EfficientDriftDoubleKick` (FastPM, not reversible) and `ReversibleDoubleKickDrift` (supports adjoint reverse mode).
- **Interpolation kernels**: `AbstractInterp` → `NoInterp`, `OnionTiler`, `TelephotoInterp`. Handle lightcone painting during integration.
- **Correction kernels**: `AbstractCorrection` → `NoCorrection`, `PGDKernel`, `SharpeningKernel`. Position/velocity corrections at each step.
- **Integration**: `integrate()` in `pm/integrate.py` drives the time-stepping loop with checkpointed or reversible adjoint support.

## Coding Conventions

- **Equinox modules everywhere**: solvers, fields, interp kernels, corrections are all `eqx.Module` subclasses. Mark non-traced metadata with `eqx.field(static=True)`.
- **`__check_init__`**: use for validation in equinox modules (not `__post_init__`).
- **`replace()` not mutation**: fields and modules are immutable PyTrees. Use `self.replace(array=new_arr)` or `eqx.tree_at()` to update.
- **`jax.jit` with `static_argnames`**: top-level functions (`lpt`, `nbody`) use `@partial(jax.jit, static_argnames=[...])` for arguments that affect tracing (shapes, enums, orders).
- **Type hints**: use `jaxtyping.Array`, `Optional`, standard typing. `F722` (jaxtyping syntax) is suppressed in ruff.
- **Imports**: `from __future__ import annotations` in every file. Relative imports within the package.
- **`__all__`**: every public module defines `__all__`.
- **No `__pow__`**: deliberately omitted on `AbstractPytree` to avoid breaking `equinox.Omega`.

## Key Dependencies (custom forks)

| Package | Source | Branch |
|---|---|---|
| `jaxpm` | `DifferentiableUniverseInitiative/JaxPM` | `41-spherical-lensing` |
| `jax_cosmo` | `ASKabalan/jax_cosmo` | `ALL-MODIF` |
| `dorian-raytrace` (optional) | `ASKabalan/Dorian` | `main` |

These are git-pinned in `pyproject.toml`. Standard PyPI for everything else (`jax>=0.6`, `equinox`, `blackjax`, `numpyro`, `healpy`, `jax-healpy`, `orbax-checkpoint`).
