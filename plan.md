# Plan

## Goal
- Simplify and fix plotting for `DensityField`, `FlatDensity`, and `SphericalDensity`, ensuring 3D fields project through `FlatDensity` helpers, consistent `plot/show` signatures, and top-level `jax.core` imports.
- Fix `PowerSpectrum` plotting so grids appear, log–log scaling is used consistently across modes (including `compare`), and `show` mirrors `plot`.
- Re-run `notebooks/lpt_power_compare.ipynb` to validate individual plots, overlays, and `mean_std` power-spectrum visuals after code changes.

## Files to modify and modifications

- `src/fwd_model_tools/fields/density.py`
  - Add top-level `import jax.core`; remove per-function imports.
  - Refactor `DensityField.plot` to project via `project(...)` and delegate to `FlatDensity.plot`; accept/forward `ax`, `nz_slices`, `show_colorbar`, `show_ticks`, `apply_log`, etc.; return delegated fig/axes.
  - Align `DensityField.show` signature with `plot` and forward all args before `plt.show()`.
  - Align `FlatDensity.show` signature with `FlatDensity.plot` (include `ax`) and forward args.
  - Ensure `SphericalDensity.plot/show` accept matching signatures (including `apply_log`, `show_colorbar` passthrough if applicable) while keeping Mollweide behavior.

- `src/fwd_model_tools/power/power_spec.py`
  - Simplify axis/grid handling; ensure log–log plotting across modes and that grids render.
  - Make `show` signature identical to `plot` (including `grid`, `title`, etc.) and forward args.
  - Update `compare` to use log–log for main panel and consistent grid handling; ratio panel shares the x-scale.

- `notebooks/lpt_power_compare.ipynb`
  - Re-execute via nbconvert workflow (execute to `_executed` then rename back) to refresh outputs showing individual plots, overlays, and `mean_std` power-spectrum visuals.
  - Cell structure: intro markdown; imports; data generation/loading; compute spectra; plot individual; overlay multiple; mean/std mode; (if present) comparison/transfer checks.

## Summary of plan
1. Adjust density plotting APIs: top-level `jax.core`, unified `plot/show`, delegate 3D plotting through `FlatDensity` projection; keep spherical plotting consistent.
2. Streamline `PowerSpectrum` plotting: log–log everywhere, grid visibility fixed, `show` mirrors `plot`, `compare` uses log–log main panel.
3. Re-execute `notebooks/lpt_power_compare.ipynb` with nbconvert (execute -> rename) to validate plotting changes; skip tests as requested.

## Questions (none)
