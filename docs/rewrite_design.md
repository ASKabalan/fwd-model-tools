# Forward Model Tools Rewrite

## Vision
- Provide a single, composable stack for forward-modeling cosmological observables: start from statistically consistent initial conditions and end at survey-ready shear, convergence, and summary statistics.
- Maintain the project’s strengths (pure JAX, distributed-friendly, IC sampling) while unifying downstream simulation and weak-lensing tooling now scattered across prototypes.
- Ship self-contained modules that can be imported independently yet compose into an “opinionated default” workflow.

## Current Status Snapshot
| Module / Capability | Description | Status |
| --- | --- | --- |
| Field structure core | Canonicalizes meshes, cosmology configs, shared dataclasses (`config.py`, field helpers). | ⬜️ Planned |
| Gaussian/lognormal priors | `normal.py` rewrite with deterministic RNG plumbing. | ⬜️ Planned |
| Particle-mesh solvers | New `pm.py` housing LPT + N-body building blocks. | ⬜️ Planned |
| Weak-lensing propagation | `weaklensing.py` with Born + ray-tracing engines. | ⬜️ Planned |
| in `weaklensing.py` have `kappa→shear` utilities | Geometry-specific transforms and map hygiene. | ⬜️ Planned |
| Power spectrum utilities | Unified `powerspec` package for flat-sky and spherical $C_\ell$. | ⬜️ Planned |
| Probablistic Forward model | Update `lensing_model.py` to use new modules end-to-end. | ⬜️ Planned |
| Sampling / likelihoods | Update `sampling.py` to consume new shear maps. | ⬜️ Planned | (this is probably already very good)
| plotting | plot kappas density flat maps slices of 3D fields and posteriors | ⬜️ Planned |

## End-to-End Architecture
1. **Field structure**: Configure meshes, cosmology, noise seeds once; feed consistent dataclasses downstream.
2. **Initial conditions (`normal.py`)**: Sample whitened Gaussian/lognormal fields and attach metadata (mesh, cosmology, seeds).
3. **PM solvers (`pm.py`)**: Transform ICs into evolved density/lightcone planes with interchangeable LPT or N-body integrators.
4. **Weak lensing (`weaklensing.py`)**: Convert 3D matter distribution into convergence/deflection planes via Born or multi-plane ray tracing.
5. **Kappa → shear utilities**: Run final geometric transforms for science-ready shear maps, masking, and consistency checks.
6. **Power spectrum utilities (`powerspec`)**: Provide fast P(k) and $C_\ell$ evaluators (flat sky and spherical) plus theoretical predictions for validation/likelihoods.

All modules expose stateless, JAX-friendly functions with explicit `jax.random.PRNGKey` and mesh descriptors; high-level builders live in `lensing_model.py` and `sampling.py`.

## Module Designs

### `pm.py`
**Responsibilities**
- Wrap JAXPM kernels for 1LPT/2LPT initialization and full N-body evolution.
- Provide composable stages (`lpt_displacements`, `nbody_evolve`, `lightcone_planes`) so advanced users can insert custom physics.

**Key APIs**
- `generate_particles(field: FieldState, scheme: Literal["1LPT","2LPT"]) -> ParticleState`
- `evolve_pm(particles, *, pm_config, integrator: Literal["kick-drift-kick","symplectic"]) -> ParticleState`
- `density_planes(particles, *, plane_spec: PlaneSpec, reduce_fn=cic)` returning stacked convergence-ready slices.

**Implementation Notes**
- Depend on `config.py` dataclasses (`PMConfig`, `PlaneSpec`) for mesh/step control.
- Keep sharding hints optional but plumbed through to JAXPM utilities.
- Integrate with `solvers/` ODE helpers for custom time stepping; expose `jit`-ready entrypoints that accept PyTrees only.

### `weaklensing.py`
**Responsibilities**
- Operate on density/deflection planes to produce convergence `κ` and shear `γ` maps.
- Support both Born approximation (single integral along LoS) and multi-plane ray tracing.

**Key APIs**
- `born_convergence(planes, nz_shear, geometry, cosmo, *, caching=None)`
- `raytrace(planes, nz_shear, *, solver="rk4", interpolator="cubic") -> RaytraceResult`
- Shared dataclasses: `SourceDistribution`, `GeometrySpec`, `RaytraceConfig`.

**Implementation Notes**
- Accept outputs from `pm.py` or externally supplied planes.
- Factor numerical kernels (kernel weighting, interpolation) into private helpers for reuse in `kappa2shear`.
- Hook into `distributed.py` for device sharding awareness; ensure consistent chunking of source redshift bins.

### `kappa2shear`
**Responsibilities**
- Convert convergence maps to shear (γ1, γ2) respecting survey geometry.
- Offer FFT-based pipeline for flat sky and HEALPix/SHT-based path for spherical maps.

**Key APIs**
- `kappa_to_shear_flat(kappa_map, kvecs, *, taper=None)`
- `kappa_to_shear_spherical(kappa_map, lmax, *, alm_backend="jax_healpy")`
- Validation helpers: `check_e_mode_purity`, `apply_mask_inplace`.

**Implementation Notes**
- Lives either as `kappa2shear.py` or `weaklensing.kappa2shear` submodule; imports `powerspec` for consistency checks.
- Provide minimal wrappers returning structured dataclasses to keep downstream typing clean.

### `powerspec` Package
**Responsibilities**
- Compute matter power spectrum P(k) (using provided cosmology or tabulated inputs) and angular power spectra $C_\ell$.
- Offer both estimator utilities (measuring from maps/fields) and theory curves for comparisons.

**Layout Proposal**
- `powerspec/__init__.py`
- `powerspec/pk.py`: wrappers for JAX-cosmo linear/nonlinear P(k), FFT-based estimators from 3D grids.
- `powerspec/cl_flat.py`: flat-sky `C_\ell` via FFTs and window corrections.
- `powerspec/cl_spherical.py`: spherical-harmonic estimators + theoretical $C_\ell$ using Limber / beyond-Limber kernels.
- `powerspec/models.py`: convenience functions `theory_cl_shear`, `theory_cl_kappa`.

**Implementation Notes**
- Keep estimators JAX-compatible; optionally return metadata (bin edges, shot noise).
- House validation scripts under `tests/test_powerspec*.py` with deterministic seeds.

## Data & Control Flow
```
Initial Conditions (normal.py)
        │
        ▼
Particles + Displacements (pm.py: LPT)
        │
        ▼
N-body Evolution (pm.py integrators)
        │
        ▼
Density / Potential Planes (pm.py)
        │
        ├──► Power spectrum checks (powerspec.*)
        ▼
Weak Lensing Propagation (weaklensing.py)
        │
        ▼
κ maps
        │
        ├──► Theory comparison (powerspec.models)
        ▼
γ maps (kappa2shear) → samplers/likelihoods
```

## Implementation Plan
| Phase | Deliverables | Tests / Validation | Status |
| --- | --- | --- | --- |
| Field structure + priors cleanup | `config.py` refactor, `normal.py` rewrite, baseline dataclasses. | Unit coverage on IC sampling; deterministic RNG tests. | ✅ Done |
| PM core (`pm.py`) | LPT + N-body API, lightcone plane generators, docstrings. | Compare against JAXPM reference runs; energy conservation smoke tests. | ⬜️ Planned |
| Weak-lensing core | Born + ray-tracing kernels, new configs, integration into `lensing_model.py`. | Regression vs existing scripts; numerical convergence harness. | ⬜️ Planned |
| Kappa→shear utilities | Flat/spherical pipelines, masks, validation helpers. | Round-trip κ→γ→κ tests; E/B leakage metrics. | ⬜️ Planned |
| Power spectrum module | Flat-sky & spherical estimators + theory calculators. | Analytical benchmarks, `pytest` covering estimators + models. | ⬜️ Planned |
| API integration & docs | Update `lensing_model.py`, scripts, README; add gallery notebook. | End-to-end smoke test via `scripts/run_lensing_model.py`. | ⬜️ Planned |

## Open Questions / Follow-Ups
1. **Geometry abstraction**: single `GeometrySpec` dataclass vs. separate flat/spherical classes?
2. **Power spectrum dependencies**: keep purely in JAX (jax-cosmo) or allow optional CAMB/CLASS inputs?
3. **Distributed story**: what is the minimum supported mesh/sharding configuration for PM + weak lensing to guarantee reproducibility?
4. **Testing strategy**: need lightweight fixtures (e.g., 32³ box) to keep CI runtime acceptable while validating new modules.
5. **Interoperability**: how should users plug in custom summary statistics (e.g., peak counts) without modifying core modules?

This document is the starting point; as modules land we can mark the corresponding rows with green check marks and expand the API sections with concrete signatures/examples.
