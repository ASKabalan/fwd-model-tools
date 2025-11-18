# Plan

## Goal
- Reshape `fwd_model_tools` into a clear, modular API organized around fields, PM, lensing, probabilistic models, and sampling.
- Move legacy monolithic pieces (e.g. `lensing_model.py`, `sampling.py`) into smaller, well-named modules with `_src` mirrors for internal code.
- Modernize configuration and probabilistic models while updating tests, scripts, docs, and notebooks to the new layout (backward-compatibility not required).

## Files to modify and modifications

- [x] `src/fwd_model_tools/normal.py`
  - Rename module to `src/fwd_model_tools/initial.py`.
  - Keep `gaussian_initial_conditions` and `interpolate_initial_conditions` as the public IC API.
  - Ensure public APIs use explicit `jax.random.PRNGKey` and return `DensityField`-based PyTrees (no hidden RNG/global state).
  - Update imports in code, tests, scripts, README, and notebooks from `fwd_model_tools.normal` to `fwd_model_tools.initial`.

- [x] `src/fwd_model_tools/field.py`, `src/fwd_model_tools/kappa.py`, `src/fwd_model_tools/_src/*`
  - Introduce a single public fields module:
    - Add `src/fwd_model_tools/fields.py` (module, not package) as the canonical public entry point for all field types.
  - Move core field implementations under `_src`:
    - Create `src/fwd_model_tools/_src/fields/_density.py` and move implementations of:
      - `FieldStatus`, `DensityStatus`, `DensityField`, `ParticleField`, `FlatDensity`, `SphericalDensity`, `particle_from_density`, `stack`.
      - Plus supporting logic currently in `field.py` that is not already in `_checks`, `_painting`, or `core`.
    - Adjust imports in `_density.py` to use `_src._checks`, `_src._painting`, and `_src.core` as today.
  - Wire the public `fields` module:
    - Implement `fields.py` as a thin wrapper importing these types from `_src.fields._density` and re-exporting them.
    - Update all imports in the repo (`tests`, `scripts`, notebooks, internal modules) to use `from fwd_model_tools.fields import ...` instead of `from fwd_model_tools.fields import ...`.
  - Kappa and shear fields:
    - Create `src/fwd_model_tools/_src/fields/_kappa_shear.py` and move the implementations of `FlatKappaField` and `SphericalKappaField` there as subclasses of `FlatDensity` and `SphericalDensity`.
    - In the same module, introduce `FlatShearField` and `SphericalShearField`:
      - Same parents (`FlatDensity` / `SphericalDensity`).
      - Carry a `z_source` attribute (like the kappa fields).
      - Methods such as `compute_power_spectrum` and `get_shear` remain `NotImplementedError` for now.
    - Re-export `FlatKappaField`, `SphericalKappaField`, `FlatShearField`, `SphericalShearField` from `fields.py` so users always write:
      - `from fwd_model_tools.fields import FlatKappaField, SphericalKappaField, ...`.
  - Clean-up:
    - Once all imports are updated to `fwd_model_tools.fields`, remove `field.py` and `kappa.py`.
    - All kappa/shear implementations live in `_src/fields/_kappa_shear.py` with public access exclusively via `fields.py`.

- [x] `src/fwd_model_tools/pm.py` → `src/fwd_model_tools/pm/`
  - Restructure PM code as a small package:
    - Create `src/fwd_model_tools/pm/__init__.py`.
    - Move current `lpt` implementation into `src/fwd_model_tools/pm/lpt.py`, importing field types from `fwd_model_tools.fields`.
    - Move current `nbody` implementation into `src/fwd_model_tools/pm/nbody.py`, preserving support for both single-geometry and tuple-of-geometries (as already tested in `tests/test_pm_geometry.py`).
    - Optionally move closely related utilities (e.g. `compute_lpt_lightcone_scale_factors`) to `pm/` or keep them in `utils` and re-export as needed; the public entry-point will be via `__init__.py`.
  - Public API:
    - `pm/__init__.py` re-exports `lpt` and `nbody`.
    - Update imports to use `from fwd_model_tools.pm import lpt, nbody`.
  - Clean-up:
    - Remove the old `pm.py` once all imports are updated.

- [x] `src/fwd_model_tools/solvers/`
  - Keep `integrate.py`, `ode.py`, and `semi_implicit_euler.py` in place.
  - Update imports from `..field` to `..fields` and pull `particle_from_density` from `fwd_model_tools.fields` (or `_src` as appropriate).
  - Ensure `src/fwd_model_tools/solvers/__init__.py` still exposes the intended public solver API.

- [x] `src/fwd_model_tools/lensing.py` → `src/fwd_model_tools/lensing/`
  - Turn lensing into a package:
    - Create `src/fwd_model_tools/lensing/born.py`:
      - Move the public `born(...)` function here, preserving signature and semantics.
      - Move internal helpers into `src/fwd_model_tools/_src/lensing/_born.py`, hosting:
        - `_born_flat`, `_born_spherical`, `_born_core_impl`.
      - Implement `born(...)` as a thin dispatcher that:
        - Accepts `FlatDensity` or `SphericalDensity` (and kappa/shear derivatives) from `fwd_model_tools.fields`.
        - Delegates to the appropriate helper in `_src.lensing._born`.
    - Create `src/fwd_model_tools/lensing/raytrace.py`:
      - Add a placeholder `raytrace(...)` that raises `NotImplementedError`, with a clear docstring describing intended arguments and return values.
    - Add `src/fwd_model_tools/lensing/__init__.py` re-exporting `born` and `raytrace`.
  - Update imports:
    - Replace uses of `fwd_model_tools.lensing` (monolithic module) to import from `fwd_model_tools.lensing` (package) instead.
  - Clean-up:
    - Delete the old monolithic `src/fwd_model_tools/lensing.py` once all imports are updated.

- [x] `src/fwd_model_tools/probabilistic_models/config.py` (moved from `src/fwd_model_tools/config.py`)
  - Replace `Configurations` `NamedTuple` with a single dataclass-like configuration used by both full-field and power-spectrum models:
    - Keep the name `Configurations` but implement as a dataclass-equivalent with fields:
      - `density_plane_smoothing: float`
      - `nz_shear: list`
      - `fiducial_cosmology: Any`
      - `sigma_e: float`
      - `priors: dict`
      - `t0: float`
      - `dt0: float`
      - `t1: float`
      - `adjoint: RecursiveCheckpointAdjoint = RecursiveCheckpointAdjoint(5)`
      - `min_redshift: float = 0.01`
      - `max_redshift: float = 3.0`
      - `geometry: str = "spherical"`
      - `log_lightcone: bool = False`
      - `log_ic: bool = False`
      - `ells: Array = jnp.arange(2, 2048)`
    - Also add simulation hyperparameters:
      - `number_of_shells: int = 8`
      - `lensing: str = "born"`
      - `lpt_order: int = 2`
  - Remove fields that belong naturally to `DensityField` (mesh/box geometry, `nside`, `field_npix`, `observer_position`, `sharding`, `halo_size`) and instead obtain those from field objects or separate arguments.
  - Update all call sites in probabilistic models, scripts, and tests to use this streamlined `Configurations` dataclass.

- `src/fwd_model_tools/utils.py`
  - Add general cosmology helpers from `lensing_model.py`:
    - `Planck18(...)`.
    - `reconstruct_full_sphere(...)`.
  - Keep existing geometry helpers:
    - `compute_box_size_from_redshift`, `compute_max_redshift_from_box_size`, `compute_lightcone_shells`, `compute_snapshot_scale_factors`, `compute_lpt_lightcone_scale_factors`.
  - Update imports across the repo to use `from fwd_model_tools.utils import Planck18, reconstruct_full_sphere, ...` instead of importing them from `lensing_model.py`.

- `src/fwd_model_tools/lensing_model.py` and `src/fwd_model_tools/powerspec_model.py`
  - [x] Introduce a dedicated `probabilistic_models/` package:
    - `probabilistic_models/forward_model.py` houses `Planck18` and the deterministic `make_full_field_model` that consumes user-provided `DensityField` templates and uses explicit `nz_shear` lensing.
    - `probabilistic_models/full_field_model.py` now wraps the forward model with NumPyro, records arrays at deterministic sites, and requires callers to pass the template field explicitly.
  - [x] Move the two-point / power spectrum helpers into `probabilistic_models/power_spec_model.py` (currently a copy of the legacy module) and update it to the new config/layout.
  - [ ] Update README/docs/scripts/notebooks to import from `fwd_model_tools.probabilistic_models` and describe the new template-field requirement.

- [x] `src/fwd_model_tools/sampling.py` → `src/fwd_model_tools/sampling/`
  - Restructure sampling into a package:
    - Created `src/fwd_model_tools/sampling/__init__.py` exposing:
      - `DistributedNormal`
      - `batched_sampling`
      - `load_samples`
      - `save_sharded`
      - `load_sharded`.
    - `src/fwd_model_tools/sampling/dist.py`:
      - Moved `DistributedNormal` here from the old `sampling.py`.
      - Added missing imports: `is_prng_key` (`numpyro.util`) and `normal_field` (`jaxpm.distributed`).
    - `src/fwd_model_tools/sampling/batched_sampling.py`:
      - Moved `batched_sampling` and `load_samples` here.
      - Removed emojis from all print/log statements and kept behavior identical for NumPyro and BlackJAX backends (including MCLMC).
    - `src/fwd_model_tools/sampling/persistency.py`:
      - Moved Orbax-based `save_sharded` and `load_sharded` from `distributed.py`.
      - Kept API compatible with existing tests and scripts.
    - (Later) `src/fwd_model_tools/sampling/plot.py`:
      - Will host sampling-specific plotting helpers (e.g. `plot_posterior`, IC comparison) moved from `plotting.py`.
  - Update `tests/test_sampling.py`:
    - Import `save_sharded` from `fwd_model_tools.sampling.persistency`.
    - Import `batched_sampling`, `load_samples` from `fwd_model_tools.sampling`.
  - Clean-up:
    - The legacy `sampling.py` module has been removed; the package layout is now the single source of truth.

- [x] `src/fwd_model_tools/distributed.py`
  - Moved the Orbax checkpointing logic to `sampling.persistency` as above.
  - Implemented `save_sharded` / `load_sharded` in `distributed.py` as thin shims that delegate to `fwd_model_tools.sampling.persistency`, so external imports remain valid.
  - Updated call sites (tests, scripts) to import `save_sharded` from `fwd_model_tools.sampling.persistency` instead of `fwd_model_tools.distributed`.

- `src/fwd_model_tools/plotting.py`
  - Narrow this module’s responsibilities:
    - [ ] Eventually remove `plot_kappa` and `plot_lightcone` (and similar field-map helpers). Users should rely on `.plot(...)` / `.show(...)` methods of `FlatDensity`, `SphericalDensity`, kappa, and shear fields from `fwd_model_tools.fields`.
    - For now, mark `plot_kappa` and `plot_lightcone` as legacy wrappers in docstrings, and stop exporting them from the top-level `fwd_model_tools` namespace.
  - [x] Move sampling-related plotting helpers (e.g. `plot_posterior`, IC comparison) into `sampling/plot.py`:
    - `plot_posterior` and `plot_ic` now live in `src/fwd_model_tools/sampling/plot.py`.
    - `plotting.py` keeps thin shims that delegate to `sampling.plot` to avoid breaking existing imports.
  - After the scripts/notebooks are fully migrated, `plotting.py` may only need to keep `plot_gradient_analysis` and any other non-sampling, non-field-specific utilities; if nothing remains, we can later remove it entirely.
  - Update scripts and docs so:
    - Field map visualization uses the `fields` API directly.
    - Posterior/IC plots use `fwd_model_tools.sampling.plot` (or `fwd_model_tools.sampling` re-exports).

- [x] `src/fwd_model_tools/__init__.py`
  - Redefined the top-level public API as a curated surface:
    - Fields (from `fields` package):
      - `DensityField`, `ParticleField`, `FlatDensity`, `SphericalDensity`,
        `FlatKappaField`, `SphericalKappaField`, `FlatShearField`, `SphericalShearField`,
        `FieldStatus`, `DensityStatus`, `stack`.
    - Initial conditions (from `initial.py`):
      - `gaussian_initial_conditions`, `interpolate_initial_conditions`.
    - PM (from `pm` package):
      - `lpt`, `nbody`.
    - Lensing (from `lensing` package):
      - `born`, `raytrace`.
    - Probabilistic models (from `probabilistic_models` + `utils.py`):
      - `Configurations`, `Planck18`, `make_full_field_model`, `full_field_probmodel`,
        `reconstruct_full_sphere`, `powerspec_probmodel`, `make_2pt_model`, `pixel_window_function`.
    - Power (from `power` package):
      - `PowerSpectrum`, `compute_pk`, `compute_flat_cl`, `compute_spherical_cl`, `compute_theory_cl`.
    - Utilities (from `utils.py`):
      - `compute_box_size_from_redshift`, `compute_lightcone_shells`,
        `compute_max_redshift_from_box_size`, `compute_snapshot_scale_factors`,
        `compute_lpt_lightcone_scale_factors`.
    - Sampling (from `sampling` package):
      - `DistributedNormal`, `batched_sampling`, `load_samples`.
    - Plotting:
      - Re-export a small, focused subset:
        - `plot_posterior` (from `sampling.plot`) and `plot_ic`, `plot_gradient_analysis` (from `plotting.py`).
  - Updated `__all__` to this curated public surface; legacy exports are no longer part of the top-level namespace.

- `scripts/*.py`
  - `scripts/run_simple_sampling.py`
    - Update imports to:
      - `from fwd_model_tools.sampling import DistributedNormal`.
      - `from fwd_model_tools.sampling import batched_sampling, load_samples`.
      - `from fwd_model_tools.sampling.plot import plot_posterior`.
    - Remove imports from `fwd_model_tools.fields`, old `fwd_model_tools.sampling` layout, and `fwd_model_tools.distributed`.
  - `scripts/run_distributed_full_field_inference.py`, `scripts/run_full_field_vs_powerspec.py`, `scripts/generate_pm_densities.py`
    - Update imports to use:
      - `initial` for ICs.
      - `fields` for field objects.
      - `pm` for LPT/N-body.
      - `lensing` for Born lensing.
      - `probabilistic_models` for full-field and power-spectrum models.
      - `sampling` and `sampling.plot` for sampling and plotting.
    - Update usage of `Configurations`, `full_field_probmodel`, and power-spectrum models to match the new dataclass and module.
    - Ensure no emojis remain in script logging.

- `tests/`
  - `tests/test_pm_geometry.py`
    - Import:
      - `FlatDensity`, `SphericalDensity` from `fwd_model_tools.fields`.
      - `Planck18`, `compute_snapshot_scale_factors` from `fwd_model_tools.utils`.
      - `gaussian_initial_conditions` from `fwd_model_tools.initial`.
      - `lpt`, `nbody` from `fwd_model_tools.pm`.
    - Keep assertions verifying tuple-of-geometries behavior.
  - `tests/test_power.py`, `tests/test_plot_axes.py`
    - Replace imports from `fwd_model_tools.fields` with `fwd_model_tools.fields`.
    - Confirm `compute_power_spectrum` methods still behave as before.
  - `tests/test_sampling.py`
    - Import `save_sharded` from `fwd_model_tools.sampling.persistency`.
    - Import `batched_sampling`, `load_samples` from `fwd_model_tools.sampling`.
    - Keep current behavior and assertions intact.
  - New tests:
    - Add a small smoke test for `fwd_model_tools.lensing.born`:
      - Minimal `FlatDensity` input (small 2D grid).
      - Minimal `SphericalDensity` input (small `nside`) and check for correct shapes / no runtime errors.
    - Add a smoke test for `make_full_field_model` and `full_field_probmodel`:
      - Construct a tiny `Configurations` instance and run one forward pass, verifying output shapes/dtypes.
    - Add a minimal test for `Configurations` dataclass construction and default values (`ells`, `min_redshift`, `max_redshift`).

- `docs/rewrite_design.md`, `README.md`, and `notebooks/`
  - Update design docs and README to reflect the new layout:
    - `from fwd_model_tools.fields import DensityField, FlatDensity, SphericalDensity, FlatKappaField, ...`
    - `from fwd_model_tools.initial import gaussian_initial_conditions`
    - `from fwd_model_tools.pm import lpt, nbody`
    - `from fwd_model_tools.lensing import born`
    - `from fwd_model_tools.probabilistic_models import full_field_probmodel, powerspec_probmodel`
    - `from fwd_model_tools.sampling import DistributedNormal, batched_sampling, load_samples`
    - `from fwd_model_tools.sampling.plot import plot_posterior`
  - For key notebooks (`01-basics`, `02-lpt-lightcone`, `03-nbody-simulation`, `04-lensing-example`):
    - When we later modify them, update imports and narrative to the new modules, keep cells small and modular, and re-execute via `nbconvert` per AGENTS instructions.
  - Replace outdated references to `lensing_model.py`, `normal.py`, and `field.py` with the new APIs.

## Summary of plan
1. Introduce and wire the new module layout (`initial`, `fields`, `pm/`, `lensing/`, `probabilistic_models`, `sampling/`), moving implementations into `_src` mirrors where appropriate and simplifying public imports.
2. Refactor configuration and probabilistic models around a single `Configurations` dataclass and the `probabilistic_models` module, ensuring NumPyro sites operate on arrays (or simple distributions) rather than custom classes.
3. Split sampling, persistency, and plotting concerns into the `sampling` package, remove emojis and redundant plotting wrappers, update imports in tests, scripts, docs, and notebooks, and add targeted smoke tests for the reorganized modules.
